"""本地演练启动器：一条命令拉起协调方与多个独立厂站代理。

用法：
  python -m federated.demo init  --dir .demo/fed --sites 3        # 生成各厂CSV
  python -m federated.demo up    --dir .demo/fed --port 8310      # 起协调方+厂站代理
  python -m federated.demo run   --dir .demo/fed --port 8310      # 用已注册版本发起一次运行
演练（断线、迟到、同号异包、中断恢复）在页面 http://127.0.0.1:8310/ 上操作。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .sample_data import FACTORIES, factory_records
from .site_agent import sample_version

FIELDS = ["day", "line", "units", "run_hours", "temp_c", "idle_hours", "kwh"]


def cmd_init(args):
    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    for site_id, line, seed in FACTORIES[: args.sites]:
        rows = factory_records(seed, line, days=args.days)
        path = root / f"{site_id}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{site_id} {line} -> {path}（{len(rows)}天，样本版本{sample_version(rows)[:12]}）")
    print("下一步：python -m federated.demo up --dir", args.dir)


def cmd_up(args):
    root = Path(args.dir)
    procs = []
    coord_cmd = [sys.executable, "-m", "federated.coordinator",
                 "--port", str(args.port), "--data-dir", str(root / "coordinator")]
    procs.append(("coordinator", subprocess.Popen(coord_cmd)))
    for i, (site_id, _, _) in enumerate(FACTORIES[: args.sites]):
        csv_path = root / f"{site_id}.csv"
        if not csv_path.exists():
            continue
        agent_cmd = [sys.executable, "-m", "federated.site_agent",
                     "--site", site_id, "--data", str(csv_path),
                     "--coord", f"http://127.0.0.1:{args.port}",
                     "--state-dir", str(root / "state" / site_id),
                     "--control-port", str(args.port + 101 + i),
                     "--interval", "0.5"]
        procs.append((site_id, subprocess.Popen(agent_cmd)))
    print(f"协调方页面: http://127.0.0.1:{args.port}/  （Ctrl+C 全部停止）")
    try:
        for _, proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        for _, proc in procs:
            proc.terminate()


def cmd_run(args):
    coord = f"http://127.0.0.1:{args.port}"
    with urllib.request.urlopen(coord + "/api/sites", timeout=5) as resp:
        sites = {s["site_id"]: s for s in json.loads(resp.read().decode("utf-8"))["sites"]}
    participants = []
    for site_id, _, _ in FACTORIES[: args.sites]:
        if site_id not in sites:
            raise SystemExit(f"站点{site_id}尚未注册，请先 up 并等待心跳")
        participants.append({"site_id": site_id, "sample_version": sites[site_id]["sample_version"]})
    body = {"params": {"max_features": args.max_features, "scenario_pct": args.scenario_pct},
            "boundary": {"start": args.start, "end": args.end},
            "participants": participants, "created_by": "demo-cli"}
    req = urllib.request.Request(coord + "/api/runs", method="POST",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    print(f"已发起并冻结: {out['run_id']}  清单指纹 {out['manifest_hash'][:16]}…")
    print(f"在页面待汇总中查看进度，齐备后进入待批准：{coord}/")


def main():
    parser = argparse.ArgumentParser(description="联合归因本地演练")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init", help="生成各厂演示CSV")
    p_init.add_argument("--dir", required=True)
    p_init.add_argument("--sites", type=int, default=3, choices=(2, 3))
    p_init.add_argument("--days", type=int, default=160)
    p_init.set_defaults(func=cmd_init)
    p_up = sub.add_parser("up", help="启动协调方与厂站代理")
    p_up.add_argument("--dir", required=True)
    p_up.add_argument("--port", type=int, default=8310)
    p_up.add_argument("--sites", type=int, default=3, choices=(2, 3))
    p_up.set_defaults(func=cmd_up)
    p_run = sub.add_parser("run", help="按已注册样本版本发起一次联合运行")
    p_run.add_argument("--dir", required=True)
    p_run.add_argument("--port", type=int, default=8310)
    p_run.add_argument("--sites", type=int, default=3, choices=(2, 3))
    p_run.add_argument("--start", default="2026-05-01")
    p_run.add_argument("--end", default="2026-10-07")
    p_run.add_argument("--max-features", dest="max_features", type=int, default=4)
    p_run.add_argument("--scenario-pct", dest="scenario_pct", type=float, default=10)
    p_run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
