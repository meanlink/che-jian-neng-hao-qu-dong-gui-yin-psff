"""厂站代理：原始日样本不出站，只向外提交充分统计量与留出误差。

职责：
- 本地加载并校验日样本（与单机版相同的数据质量规则），计算样本版本指纹；
- 轮询协调方，按冻结清单与开放阶段计算贡献并提交；
- 断线/超时自动重试：同一内容同一序号，协调方幂等不重复累计；
- 控制端支持断线演练（暂停提交N秒）与恢复。

运行：python -m federated.site_agent --site S1 --data S1.csv --coord http://127.0.0.1:8310 \
        --state-dir state/S1 --control-port 8411
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import protocol as proto
from .algorithm import parse_records, predict, stats_from_rows, zero_stats


def load_records(path):
    """加载CSV或JSON日样本；返回原始记录列表（数值列保持文件原样，校验在parse阶段）。"""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sample_version(records):
    """样本版本指纹：规范化（日期排序、数值列float）后的内容哈希。"""
    normalized = []
    for item in records:
        normalized.append({
            "day": str(item["day"]),
            "line": str(item.get("line", "")),
            "units": float(item["units"]),
            "run_hours": float(item["run_hours"]),
            "temp_c": float(item["temp_c"]),
            "idle_hours": float(item["idle_hours"]),
            "kwh": float(item["kwh"]),
        })
    normalized.sort(key=lambda r: r["day"])
    return proto.content_hash(normalized)


def train_payload(manifest, parsed, site_id, version):
    """阶段一贡献：训练窗累计统计量(4个折边界)+情景窗统计+训练窗特征极值。"""
    windows = manifest["windows"]
    start = date.fromisoformat(windows["boundary"]["start"])
    end = date.fromisoformat(windows["boundary"]["end"])
    bounds = windows["cum_bounds"]
    in_boundary = [r for r in parsed if start <= r[0] <= end]
    cum = []
    for bound in bounds:
        cum.append(stats_from_rows([r for r in in_boundary if (r[0] - start).days < bound]))
    scen_start = date.fromisoformat(windows["scenario"]["start"])
    scenario = stats_from_rows([r for r in in_boundary if r[0] >= scen_start])
    train_rows = [r for r in in_boundary if (r[0] - start).days < bounds[-1]]
    if train_rows:
        train_min = [min(r[1][j] for r in train_rows) for j in range(4)]
        train_max = [max(r[1][j] for r in train_rows) for j in range(4)]
    else:
        train_min = train_max = [None] * 4
    return {
        "site_id": site_id,
        "sample_version": version,
        "boundary_records": len(in_boundary),
        "coverage": {"first": in_boundary[0][0].isoformat() if in_boundary else None,
                     "last": in_boundary[-1][0].isoformat() if in_boundary else None},
        "cum": cum,
        "scenario": scenario,
        "train_min": train_min,
        "train_max": train_max,
    }


def holdout_payload(manifest, parsed, site_id, version, model):
    """阶段二贡献：用全局系数在本厂留出窗上计算的误差与目标汇总。"""
    windows = manifest["windows"]
    start = date.fromisoformat(windows["boundary"]["start"])
    end = date.fromisoformat(windows["boundary"]["end"])
    train_days = windows["train"]["days"]
    rows = [r for r in parsed if start <= r[0] <= end and (r[0] - start).days >= train_days]
    sse = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    for _, vector, target in rows:
        resid = target - predict(model, vector)
        sse += resid * resid
        sum_y += target
        sum_y2 += target * target
    return {"site_id": site_id, "sample_version": version,
            "n": len(rows), "sse": sse, "sum_y": sum_y, "sum_y2": sum_y2}


class SiteAgent:
    def __init__(self, site_id, data_path, coord, state_dir, control_port=None, interval=1.0):
        self.site_id = site_id
        self.data_path = Path(data_path)
        self.coord = coord.rstrip("/")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.control_port = control_port
        self.paused_until = 0.0
        self.stop_event = threading.Event()
        self.control_server = None
        self.log_lines = []
        raw = load_records(self.data_path)
        self.version = sample_version(raw)
        self.local_error = None
        try:
            self.parsed = parse_records(raw)
        except ValueError as exc:
            self.parsed = []
            self.local_error = str(exc)
        self.records_count = len(self.parsed)
        self.coverage = {"first": self.parsed[0][0].isoformat() if self.parsed else None,
                         "last": self.parsed[-1][0].isoformat() if self.parsed else None}
        self.state_path = self.state_dir / f"{self.site_id}.json"
        self.state = self._load_state()

    # ------------------------------------------------------------ 状态与日志
    def _load_state(self):
        if self.state_path.exists():
            with open(self.state_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {"submitted": {}}

    def _save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False)
        tmp.replace(self.state_path)

    def log(self, message):
        line = f"[{self.site_id}] {message}"
        self.log_lines.append(line)
        if len(self.log_lines) > 200:
            del self.log_lines[:-200]
        print(line, flush=True)

    # ------------------------------------------------------------ HTTP
    def _request(self, method, path, body=None, timeout=5):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.coord + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _post(self, path, body):
        return self._request("POST", path, body)

    def _get(self, path):
        return self._request("GET", path)

    # ------------------------------------------------------------ 主循环
    def tick(self):
        if time.time() < self.paused_until:
            return  # 断线演练：暂停提交，模拟厂站断线
        try:
            self._post("/api/sites/register", {
                "site_id": self.site_id, "control_url": self.control_url(),
                "sample_version": self.version, "records": self.records_count,
                "coverage": self.coverage})
            _, overview = self._get("/api/overview")
        except (OSError, urllib.error.URLError):
            return  # 协调方不可达：下个周期重试
        for run in overview.get("runs", []):
            if run["status"] != "collecting" or self.site_id not in run["participants"]:
                continue
            try:
                self._serve_run(run["run_id"])
            except (OSError, urllib.error.URLError):
                return

    def _serve_run(self, run_id):
        _, brief = self._get(f"/api/runs/{run_id}/brief?site_id={self.site_id}")
        stage = brief["open_stage"]
        if stage is None or stage in brief["my"]:
            return  # 已确认或无可提交阶段
        frozen = None
        for p in brief["manifest"]["participants"]:
            if p["site_id"] == self.site_id:
                frozen = p["sample_version"]
        if self.local_error:
            self._post(f"/api/runs/{run_id}/fail",
                       {"site_id": self.site_id, "reason": f"本地数据校验失败：{self.local_error}"})
            self.log(f"运行{run_id}标记失败：{self.local_error}")
            return
        if frozen != self.version:
            self._post(f"/api/runs/{run_id}/fail",
                       {"site_id": self.site_id, "reason": "样本版本与冻结清单不符，本地数据疑似变更"})
            self.log(f"运行{run_id}样本版本不符，已报告失败")
            return
        if stage == proto.STAGE_TRAIN:
            payload = train_payload(brief["manifest"], self.parsed, self.site_id, self.version)
        else:
            model = brief["stage_brief"]["model"]
            payload = holdout_payload(brief["manifest"], self.parsed, self.site_id, self.version, model)
        sent = self.state["submitted"].setdefault(run_id, {})
        seq = sent.get(stage, {}).get("seq", 1)
        env = proto.build_envelope(run_id, stage, self.site_id, seq, payload)
        try:
            _, resp = self._post(f"/api/runs/{run_id}/contributions", env)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("error", {})
            except Exception:
                detail = {}
            code = detail.get("code", "")
            if code == "stage_closed":
                return  # 阶段已推进，下轮按新阶段重算
            if code == "seq_conflict":
                self.log(f"运行{run_id}阶段{stage}同号异内容被拒，检查本地数据是否变更")
                time.sleep(self.interval * 5)
                return
            raise
        sent[stage] = {"seq": seq, "hash": env["content_hash"][:12]}
        self._save_state()
        self.log(f"运行{run_id}阶段{stage}贡献已确认（seq={seq} hash={env['content_hash'][:12]}）")

    def control_url(self):
        if self.control_port:
            return f"http://127.0.0.1:{self.control_port}"
        return ""

    # ------------------------------------------------------------ 控制端（断线演练）
    def start_control(self):
        agent = self

        class Control(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _send(self, obj):
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                if self.path == "/control/disconnect":
                    seconds = float(body.get("seconds", 30))
                    agent.paused_until = time.time() + seconds
                    agent.log(f"断线演练：暂停提交{seconds:g}秒")
                    self._send({"paused_seconds": seconds})
                elif self.path == "/control/resume":
                    agent.paused_until = 0.0
                    agent.log("断线演练结束：恢复提交")
                    self._send({"resumed": True})
                else:
                    self._send({"error": "未知控制指令"})

            def do_GET(self):
                if self.path == "/control/status":
                    self._send({"site_id": agent.site_id, "paused": time.time() < agent.paused_until,
                                "version": agent.version, "records": agent.records_count})
                else:
                    self._send({"error": "未知路径"})

        self.control_server = ThreadingHTTPServer(("127.0.0.1", self.control_port), Control)
        thread = threading.Thread(target=self.control_server.serve_forever, daemon=True)
        thread.start()

    def run(self, once=False):
        if self.control_port:
            self.start_control()
        self.log(f"代理启动：样本版本{self.version[:12]} 记录{self.records_count}条 协调方{self.coord}")
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:  # 单轮异常不终止代理
                self.log(f"轮询异常：{exc}")
            if once:
                break
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        if self.control_server:
            self.control_server.shutdown()
            self.control_server.server_close()


def main():
    parser = argparse.ArgumentParser(description="厂站代理（原始日样本不出站）")
    parser.add_argument("--site", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--coord", default="http://127.0.0.1:8310")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--control-port", type=int, default=None)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    agent = SiteAgent(args.site, args.data, args.coord, args.state_dir,
                      control_port=args.control_port, interval=args.interval)
    try:
        agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
