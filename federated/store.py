"""协调方状态机：冻结清单、阶段屏障、幂等贡献、恢复与批准门。

持久化：每次状态变更整体原子落盘（临时文件+rename），协调服务中断后
从数据目录恢复；已确认阶段的快照不可变，恢复只重放未完成的屏障判定。

幂等与防串（同阶段各厂齐备才推进，缺席不当零）：
- 贡献按 (run, stage, site) 归键存储，聚合只对冻结参与厂全集做一次，
  结构上不存在重复累计；
- 同序号同内容重试 → 幂等接受；同序号异内容 → 拒收；旧阶段迟到回包 →
  阶段栅栏拒收，不会串入当前阶段；
- 所有变更持锁进行，并发推进与迟到回包不会交叉污染。
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

from . import algorithm as alg
from . import protocol as proto


class StoreError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class Store:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        (self.dir / "runs").mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.runs = {}
        self.sites = {}
        self._load()

    # ------------------------------------------------------------ 持久化与恢复
    def _load(self):
        for path in sorted((self.dir / "runs").glob("*.json")):
            with open(path, encoding="utf-8") as fh:
                run = json.load(fh)
            self.runs[run["run_id"]] = run
        sites_path = self.dir / "sites.json"
        if sites_path.exists():
            with open(sites_path, encoding="utf-8") as fh:
                self.sites = json.load(fh)

    def _persist_run(self, run):
        _atomic_write_json(self.dir / "runs" / f"{run['run_id']}.json", run)

    def _persist_sites(self):
        _atomic_write_json(self.dir / "sites.json", self.sites)

    def _event(self, run, kind, detail=None):
        run["events"].append({"at": proto.now_iso(), "kind": kind, "detail": detail or {}})
        if len(run["events"]) > 500:
            del run["events"][:-500]

    # ------------------------------------------------------------ 运行创建（开算冻结）
    def create_run(self, params, boundary, participants, created_by=""):
        with self.lock:
            try:
                clean_params = alg.parse_params(params)
            except ValueError as exc:
                raise StoreError(400, "bad_params", str(exc)) from None
            if not isinstance(boundary, dict) or "start" not in boundary or "end" not in boundary:
                raise StoreError(400, "bad_boundary", "必须给出日期边界start/end")
            if not isinstance(participants, list) or len(participants) < 2:
                raise StoreError(400, "bad_participants", "联合归因至少需要2个参与厂")
            seen = set()
            clean_parts = []
            for p in participants:
                site_id = str(p.get("site_id", "")).strip()
                version = str(p.get("sample_version", "")).strip()
                if not site_id or not version:
                    raise StoreError(400, "bad_participants", "每个参与厂必须给出site_id与样本版本sample_version")
                if site_id in seen:
                    raise StoreError(400, "bad_participants", f"参与厂{site_id}重复")
                seen.add(site_id)
                clean_parts.append({"site_id": site_id, "sample_version": version})
            run_id = "FR-" + datetime.now().strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:8]
            try:
                manifest = proto.build_manifest(run_id, clean_params, boundary, clean_parts, created_by)
            except (ValueError, KeyError) as exc:
                raise StoreError(400, "bad_boundary", str(exc)) from None
            run = {
                "run_id": run_id,
                "status": proto.STATUS_COLLECTING,
                "open_stage": proto.STAGE_TRAIN,
                "manifest": manifest,
                "contributions": {proto.STAGE_TRAIN: {}, proto.STAGE_HOLDOUT: {}},
                "snapshots": {},
                "result": None,
                "failure": None,
                "approval": None,
                "events": [],
            }
            self._event(run, "run_created", {
                "manifest_hash": manifest["manifest_hash"],
                "participants": [p["site_id"] for p in clean_parts],
                "boundary": manifest["boundary"],
                "params": clean_params,
            })
            self.runs[run_id] = run
            self._persist_run(run)
            return run

    # ------------------------------------------------------------ 贡献提交（幂等+栅栏）
    def submit(self, env):
        with self.lock:
            errors = proto.envelope_errors(env)
            if errors:
                raise StoreError(400, "bad_envelope", "；".join(errors))
            run = self.runs.get(env["run_id"])
            if run is None:
                raise StoreError(404, "unknown_run", f"运行{env['run_id']}不存在")
            if run["status"] == proto.STATUS_FAILED:
                raise StoreError(409, "run_failed", "运行已失败，不再接收贡献")
            if run["status"] in (proto.STATUS_AWAITING, proto.STATUS_APPROVED):
                raise StoreError(409, "stage_closed", "运行汇总已完成，旧阶段回包拒收")
            if env["stage"] != run["open_stage"]:
                raise StoreError(409, "stage_closed",
                               f"阶段{env['stage']}已关闭，当前开放阶段{run['open_stage']}，迟到回包不串阶段")
            manifest = run["manifest"]
            site_id = env["site_id"]
            if site_id not in proto.participant_ids(manifest):
                raise StoreError(403, "unknown_site", f"{site_id}不在冻结参与厂名单内")
            frozen = proto.participant_version(manifest, site_id)
            if env["payload"].get("sample_version") != frozen:
                raise StoreError(409, "version_mismatch",
                                 f"{site_id}样本版本与冻结清单不符，数据不得在开算后变更")
            if proto.content_hash(env["payload"]) != env["content_hash"]:
                raise StoreError(409, "hash_mismatch", "内容指纹与payload不符，疑似篡改或截断")
            if not proto.payload_finite(env["payload"]):
                raise StoreError(400, "non_finite", "payload包含非有限数值")
            box = run["contributions"][env["stage"]]
            existing = box.get(site_id)
            if existing is not None:
                if env["seq"] == existing["seq"]:
                    if env["content_hash"] == existing["content_hash"]:
                        return {"accepted": True, "duplicate": True, "stage": env["stage"],
                                "status": run["status"], "open_stage": run["open_stage"]}
                    raise StoreError(409, "seq_conflict",
                                     f"{site_id}在阶段{env['stage']}的序号{env['seq']}已占用且内容不同，同号异内容拒收")
                if env["seq"] < existing["seq"]:
                    raise StoreError(409, "stale_seq", f"{site_id}的序号{env['seq']}早于已接收序号，按过期拒收")
                self._event(run, "contribution_replaced", {"stage": env["stage"], "site_id": site_id,
                                                           "seq": env["seq"]})
            box[site_id] = {"seq": env["seq"], "content_hash": env["content_hash"],
                            "payload": env["payload"], "received_at": proto.now_iso()}
            if existing is None:
                self._event(run, "contribution_accepted", {
                    "stage": env["stage"], "site_id": site_id, "seq": env["seq"],
                    "hash": env["content_hash"][:12]})
            self._persist_run(run)
            advanced = self._maybe_advance(run)
            return {"accepted": True, "duplicate": False, "stage_complete": advanced,
                    "status": run["status"], "open_stage": run["open_stage"]}

    # ------------------------------------------------------------ 阶段屏障与聚合
    def _maybe_advance(self, run):
        stage = run["open_stage"]
        if stage is None or run["status"] != proto.STATUS_COLLECTING:
            return False
        participants = proto.participant_ids(run["manifest"])
        box = run["contributions"][stage]
        if set(box.keys()) != set(participants):
            return False  # 缺席不当零：不齐备不推进、不以零充数
        if stage in run["snapshots"]:
            return False  # 已确认阶段快照不可变
        try:
            if stage == proto.STAGE_TRAIN:
                snapshot = self._aggregate_train(run, box)
                result = None
            else:
                snapshot = self._aggregate_holdout(run, box)
                result = self._assemble_result(run, snapshot)
        except alg.DegenerateRun as exc:
            run["status"] = proto.STATUS_FAILED
            run["failure"] = {"stage": stage, "reason": str(exc), "at": proto.now_iso()}
            self._event(run, "run_failed", {"stage": stage, "reason": str(exc)})
            self._persist_run(run)
            return False
        run["snapshots"][stage] = snapshot
        self._event(run, "stage_confirmed", {"stage": stage, "sites": sorted(box.keys())})
        if stage == proto.STAGE_TRAIN:
            run["open_stage"] = proto.STAGE_HOLDOUT
            self._event(run, "stage_advanced", {"open_stage": proto.STAGE_HOLDOUT})
        else:
            run["open_stage"] = None
            run["status"] = proto.STATUS_AWAITING
            run["result"] = result
            self._event(run, "awaiting_approval", {"holdout_rmse": run["result"]["holdout"]["rmse"]})
        self._persist_run(run)
        return True

    def _aggregate_train(self, run, box):
        manifest = run["manifest"]
        order = proto.participant_ids(manifest)
        payloads = [box[s]["payload"] for s in order]
        cum = []
        for i in range(len(manifest["windows"]["cum_bounds"])):
            seg = alg.zero_stats()
            for p in payloads:
                seg = alg.stats_add(seg, p["cum"][i])
            cum.append(seg)
        scen = alg.zero_stats()
        for p in payloads:
            scen = alg.stats_add(scen, p["scenario"])
        train_min, train_max = [None] * 4, [None] * 4
        for p in payloads:
            for j in range(4):
                lo, hi = p["train_min"][j], p["train_max"][j]
                if lo is not None:
                    train_min[j] = lo if train_min[j] is None else min(train_min[j], lo)
                    train_max[j] = hi if train_max[j] is None else max(train_max[j], hi)
        total_records = sum(p["boundary_records"] for p in payloads)
        if total_records < alg.MIN_DAYS:
            raise alg.DegenerateRun(f"至少需要{alg.MIN_DAYS}天能耗记录，以保留时间验证窗（全局仅{total_records}天）")
        if cum[-1]["n"] <= 0:
            raise alg.DegenerateRun("训练窗在冻结边界内无样本")
        _, _, _, _, _, m2yy = alg.moments(cum[-1])
        if (m2yy / cum[-1]["n"]) ** 0.5 < alg.MIN_TARGET_STD:
            raise alg.DegenerateRun("耗电量没有变化，无法辨识能耗驱动")
        best, candidates, model = alg.select_and_fit(cum, manifest["params"]["max_features"])
        return {"cum": cum, "scenario_stats": scen, "train_min": train_min, "train_max": train_max,
                "boundary_records": total_records, "train_n": cum[-1]["n"],
                "selection": {"features": best["features"], "alpha": best["alpha"],
                              "cv_rmse": best["cv_rmse"], "candidates": candidates},
                "model": model}

    def _aggregate_holdout(self, run, box):
        order = proto.participant_ids(run["manifest"])
        evals = []
        for site_id in order:
            p = box[site_id]["payload"]
            evals.append({"site_id": site_id, "n": p["n"], "sse": p["sse"],
                          "sum_y": p["sum_y"], "sum_y2": p["sum_y2"]})
        return alg.holdout_metrics(evals)

    def _assemble_result(self, run, holdout):
        manifest = run["manifest"]
        train = run["snapshots"][proto.STAGE_TRAIN]
        model = train["model"]
        delta = manifest["params"]["scenario_pct"]
        scenario = alg.scenario_report(model, train["scenario_stats"],
                                       train["train_min"], train["train_max"], delta)
        warnings = []
        if scenario["extrapolated"]:
            warnings.append("情景超出训练期变量范围，外推风险较高。")
        if not holdout["r2_defined"]:
            warnings.append("留出期耗电量恒定，R²不可定义，请以RMSE判读验证误差。")
        elif holdout["r2"] is not None and holdout["r2"] < 0.5:
            warnings.append("留出期解释度偏低，需补充设备状态、产品组合等变量后再做业务决策。")
        if scenario["scenario_kwh"] < 0:
            warnings.append("线性模型给出负的情景日电耗，当前情景超出模型的物理适用范围，需重新建模。")
        labels = manifest["feature_labels"]
        rows = []
        for name in model["features"]:
            j = alg.FEATURES.index(name)
            rows.append({"feature": labels[name], "coefficient": round(model["coef"][name], 4),
                         "baseline": round(scenario["baseline"][j], 3),
                         "scenario": round(scenario["scenario"][j], 3)})
        sites_n = len(manifest["participants"])
        diff = scenario["scenario_kwh"] - scenario["base_kwh"]
        r2_text = f"{holdout['r2']:.4f}" if holdout["r2_defined"] and holdout["r2"] is not None else "不可定义"
        return {
            "summary": (f"联合{sites_n}厂共{train['boundary_records']}天记录，"
                        f"选出{len(model['features'])}个变量，留出{holdout['n']}天"
                        f"RMSE为{holdout['rmse']:.2f}kWh；负荷变化{delta:+g}%情景日电耗"
                        f"{scenario['scenario_kwh']:.2f}kWh。"),
            "metrics": [
                {"label": "验证RMSE", "value": round(holdout["rmse"], 3), "unit": "kWh"},
                {"label": "验证R²", "value": round(holdout["r2"], 4) if holdout["r2_defined"] and holdout["r2"] is not None else "不可定义"},
                {"label": "情景日电耗", "value": round(scenario["scenario_kwh"], 2), "unit": "kWh"},
                {"label": "情景差额", "value": round(diff, 2), "unit": "kWh"},
            ],
            "selection": {"features": model["features"], "alpha": train["selection"]["alpha"],
                          "cv_rmse": train["selection"]["cv_rmse"],
                          "candidates": train["selection"]["candidates"]},
            "model": model,
            "holdout": {"n": holdout["n"], "rmse": holdout["rmse"], "r2": holdout["r2"],
                        "r2_defined": holdout["r2_defined"], "sites": holdout["sites"]},
            "table": {"columns": [{"key": "feature", "label": "入选特征"},
                                  {"key": "coefficient", "label": "原始单位系数"},
                                  {"key": "baseline", "label": "近期基准"},
                                  {"key": "scenario", "label": "情景值"}], "rows": rows},
            "scenario": {"delta_pct": delta, "base_kwh": scenario["base_kwh"],
                         "scenario_kwh": scenario["scenario_kwh"]},
            "findings": [
                f"选定岭惩罚α={train['selection']['alpha']:g}；训练内交叉验证RMSE={train['selection']['cv_rmse']:.3f}kWh。",
                f"最近{alg.SCENARIO_DAYS}天平均特征对应基准日电耗{scenario['base_kwh']:.2f}kWh，情景差额{diff:+.2f}kWh。",
                "每折的标准化器仅在该折训练部分拟合；最后20%记录未参与变量和α选择。",
                "系数已还原到原始单位，解释为其他入选变量不变时的条件关联。",
                f"各厂原始日样本未出站；交换仅为充分统计量与留出误差汇总（留出R²={r2_text}）。",
            ],
            "method": alg.METHOD_TEXT,
            "warnings": warnings,
            "manifest_hash": manifest["manifest_hash"],
        }

    # ------------------------------------------------------------ 批准与分发
    def approve(self, run_id, approver, note=""):
        with self.lock:
            run = self._require_run(run_id)
            if run["status"] != proto.STATUS_AWAITING:
                raise StoreError(409, "approval_not_ready",
                                 "各厂留出误差与全局系数齐备后才可批准，当前状态" + proto.STATUS_LABELS[run["status"]])
            if proto.STAGE_TRAIN not in run["snapshots"] or proto.STAGE_HOLDOUT not in run["snapshots"]:
                raise StoreError(409, "approval_not_ready", "阶段快照不齐备，禁止批准")
            run["status"] = proto.STATUS_APPROVED
            run["approval"] = {"approver": approver or "未署名", "note": note, "at": proto.now_iso()}
            self._event(run, "approved", {"approver": run["approval"]["approver"]})
            self._persist_run(run)
            return run

    def result(self, run_id):
        with self.lock:
            run = self._require_run(run_id)
            if run["status"] != proto.STATUS_APPROVED:
                raise StoreError(409, "result_not_public", "半成品不公开：结果需各厂误差与全局系数齐备并经批准后分发")
            return {"run_id": run_id, "manifest": run["manifest"], "result": run["result"],
                    "approval": run["approval"]}

    # ------------------------------------------------------------ 恢复
    def resume(self, run_id):
        """重放未完成屏障：贡献已齐但快照缺失（崩溃于屏障计算中）时补推进。"""
        with self.lock:
            run = self._require_run(run_id)
            before = (run["status"], run["open_stage"])
            advanced = self._maybe_advance(run)
            return {"run_id": run_id, "advanced": advanced,
                    "before": before, "after": (run["status"], run["open_stage"])}

    def recover_all(self):
        outcomes = []
        with self.lock:
            for run_id in sorted(self.runs):
                if self.runs[run_id]["status"] == proto.STATUS_COLLECTING:
                    outcomes.append(self.resume(run_id))
        return outcomes

    # ------------------------------------------------------------ 失败与查询
    def fail(self, run_id, site_id, reason):
        with self.lock:
            run = self._require_run(run_id)
            if site_id not in proto.participant_ids(run["manifest"]):
                raise StoreError(403, "unknown_site", f"{site_id}不在冻结参与厂名单内")
            if run["status"] in (proto.STATUS_AWAITING, proto.STATUS_APPROVED):
                raise StoreError(409, "run_closed", "运行已完成汇总，不能再标记失败")
            if run["status"] == proto.STATUS_FAILED:
                return run
            run["status"] = proto.STATUS_FAILED
            run["failure"] = {"site_id": site_id, "reason": reason, "at": proto.now_iso()}
            self._event(run, "run_failed", {"site_id": site_id, "reason": reason})
            self._persist_run(run)
            return run

    def _require_run(self, run_id):
        run = self.runs.get(run_id)
        if run is None:
            raise StoreError(404, "unknown_run", f"运行{run_id}不存在")
        return run

    def list_runs(self):
        with self.lock:
            out = []
            for run in sorted(self.runs.values(), key=lambda r: r["manifest"]["created_at"]):
                participants = proto.participant_ids(run["manifest"])
                open_stage = run["open_stage"]
                got = set(run["contributions"][open_stage].keys()) if open_stage else set()
                out.append({
                    "run_id": run["run_id"],
                    "status": run["status"],
                    "open_stage": open_stage,
                    "created_at": run["manifest"]["created_at"],
                    "manifest_hash": run["manifest"]["manifest_hash"],
                    "participants": participants,
                    "missing": [s for s in participants if s not in got] if open_stage else [],
                    "failure": run["failure"],
                    "approval": run["approval"],
                })
            return out

    def run_detail(self, run_id):
        with self.lock:
            run = self._require_run(run_id)
            participants = proto.participant_ids(run["manifest"])
            contrib_view = {}
            for stage, box in run["contributions"].items():
                contrib_view[stage] = {
                    site: {"seq": c["seq"], "hash": c["content_hash"][:12],
                           "received_at": c["received_at"]}
                    for site, c in box.items()
                }
            open_stage = run["open_stage"]
            missing = []
            if open_stage:
                got = set(run["contributions"][open_stage].keys())
                missing = [s for s in participants if s not in got]
            selection = None
            model = None
            holdout = None
            if proto.STAGE_TRAIN in run["snapshots"]:
                selection = run["snapshots"][proto.STAGE_TRAIN]["selection"]
                model = run["snapshots"][proto.STAGE_TRAIN]["model"]
            if proto.STAGE_HOLDOUT in run["snapshots"]:
                holdout = {k: run["snapshots"][proto.STAGE_HOLDOUT][k]
                           for k in ("n", "rmse", "r2", "r2_defined", "sites")}
            return {
                "run_id": run_id, "status": run["status"], "open_stage": open_stage,
                "manifest": run["manifest"], "contributions": contrib_view,
                "missing": missing, "selection": selection, "model": model,
                "holdout": holdout, "failure": run["failure"], "approval": run["approval"],
                "result_available": run["status"] == proto.STATUS_APPROVED,
                "result": run["result"] if run["status"] == proto.STATUS_APPROVED else None,
                "events": list(run["events"]),
            }

    def brief(self, run_id, site_id):
        """厂站轮询视图：清单、开放阶段、本厂已确认贡献与阶段简报。"""
        with self.lock:
            run = self._require_run(run_id)
            manifest = run["manifest"]
            if site_id not in proto.participant_ids(manifest):
                raise StoreError(403, "unknown_site", f"{site_id}不在冻结参与厂名单内")
            my = {}
            for stage, box in run["contributions"].items():
                if site_id in box:
                    my[stage] = {"seq": box[site_id]["seq"], "hash": box[site_id]["content_hash"][:12]}
            stage_brief = None
            if run["open_stage"] == proto.STAGE_HOLDOUT and proto.STAGE_TRAIN in run["snapshots"]:
                snap = run["snapshots"][proto.STAGE_TRAIN]
                stage_brief = {"model": snap["model"], "selection": {
                    "features": snap["selection"]["features"], "alpha": snap["selection"]["alpha"]}}
            return {"run_id": run_id, "status": run["status"], "open_stage": run["open_stage"],
                    "manifest": manifest, "my": my, "stage_brief": stage_brief}

    # ------------------------------------------------------------ 站点注册与演练
    def register_site(self, site_id, control_url, sample_version, records, coverage):
        with self.lock:
            entry = self.sites.get(site_id, {})
            entry.update({"site_id": site_id, "control_url": control_url,
                          "sample_version": sample_version, "records": records,
                          "coverage": coverage, "last_seen": proto.now_iso()})
            self.sites[site_id] = entry
            self._persist_sites()
            return entry

    def list_sites(self):
        with self.lock:
            return sorted(self.sites.values(), key=lambda s: s["site_id"])

    def drill_replay(self, run_id, stage, site_id, mode):
        """演练入口：迟到回包（closed阶段重投）与同号异内容（同seq改内容重投）。"""
        with self.lock:
            run = self._require_run(run_id)
            contrib = run["contributions"].get(stage, {}).get(site_id)
            if contrib is None:
                raise StoreError(400, "no_contribution", f"{site_id}在阶段{stage}没有已接收贡献可重放")
            if mode == "late":
                if stage == run["open_stage"]:
                    raise StoreError(400, "drill_not_applicable", "阶段尚未关闭，无法演练迟到回包")
                env = proto.build_envelope(run_id, stage, site_id, contrib["seq"], contrib["payload"])
            elif mode == "conflict":
                mutated = json.loads(json.dumps(contrib["payload"]))
                mutated["drill_tampered"] = True
                env = proto.build_envelope(run_id, stage, site_id, contrib["seq"], mutated)
            else:
                raise StoreError(400, "bad_drill", f"未知演练模式{mode}")
            try:
                self.submit(env)
                outcome = {"http": 200, "code": "accepted", "note": "未被拒收（同内容幂等）"}
            except StoreError as exc:
                outcome = {"http": exc.status, "code": exc.code, "note": exc.message}
            self._event(run, "drill", {"mode": mode, "stage": stage, "site_id": site_id, "outcome": outcome})
            self._persist_run(run)
            return outcome
