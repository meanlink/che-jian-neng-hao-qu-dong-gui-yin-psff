"""联合归因测试：算法对拍、阶段屏障、幂等、防串、恢复、批准门与端到端演练。

运行：python -m unittest discover -s federated/tests -v
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from federated import algorithm as alg
from federated import protocol as proto
from federated.algorithm import merged_analyze, parse_records
from federated.coordinator import build_server
from federated.sample_data import FACTORIES, factory_records
from federated.site_agent import SiteAgent, holdout_payload, load_records, sample_version, train_payload
from federated.store import Store, StoreError

START = "2026-05-01"
END = "2026-10-07"  # 共160天
PARAMS = {"max_features": 4, "scenario_pct": 10}
SITES = [s for s, _, _ in FACTORIES]
SEEDS = {s: seed for s, _, seed in FACTORIES}
LINES = {s: line for s, line, _ in FACTORIES}
FIELDS = ["day", "line", "units", "run_hours", "temp_c", "idle_hours", "kwh"]


# ---------------------------------------------------------------- 公共工具
def site_records(site_id, days=160):
    return factory_records(SEEDS[site_id], LINES[site_id], days=days)


def site_version(site_id):
    return sample_version(site_records(site_id))


def participants(sites=SITES):
    return [{"site_id": s, "sample_version": site_version(s)} for s in sites]


def make_run(store, sites=SITES, params=PARAMS, start=START, end=END):
    return store.create_run(params, {"start": start, "end": end}, participants(sites), "test")


def stage1_envelope(run, site_id, seq=1):
    parsed = parse_records(site_records(site_id))
    payload = train_payload(run["manifest"], parsed, site_id, site_version(site_id))
    return proto.build_envelope(run["run_id"], proto.STAGE_TRAIN, site_id, seq, payload)


def stage2_envelope(store, run, site_id, seq=1):
    model = store.runs[run["run_id"]]["snapshots"][proto.STAGE_TRAIN]["model"]
    parsed = parse_records(site_records(site_id))
    payload = holdout_payload(run["manifest"], parsed, site_id, site_version(site_id), model)
    return proto.build_envelope(run["run_id"], proto.STAGE_HOLDOUT, site_id, seq, payload)


def complete_stage1(store, run, sites=SITES):
    for s in sites:
        store.submit(stage1_envelope(run, s))


def complete_stage2(store, run, sites=SITES):
    for s in sites:
        store.submit(stage2_envelope(store, run, s))


def http(method, url, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_until(fn, timeout=20.0, interval=0.05):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(interval)
    return None


class HttpServer:
    def __init__(self, data_dir, port=0):
        self.store = Store(data_dir)
        self.httpd = build_server(self.store, "127.0.0.1", port)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=3)


class AgentRunner:
    def __init__(self, site_id, csv_path, coord, state_dir, control_port=None):
        self.agent = SiteAgent(site_id, csv_path, coord, state_dir,
                               control_port=control_port, interval=0.05)
        self.thread = threading.Thread(target=self.agent.run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.agent.stop()
        self.thread.join(timeout=3)


def write_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


class FederatedTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fed-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def subdir(self, name):
        path = Path(self.tmp) / name
        path.mkdir(parents=True, exist_ok=True)
        return path


# ---------------------------------------------------------------- 算法
class TestWindows(FederatedTestCase):
    def test_windows_mirror_original_split(self):
        w = alg.compute_windows(date(2026, 5, 1), date(2026, 10, 7))
        self.assertEqual(w["boundary"]["total_days"], 160)
        self.assertEqual(w["train"]["days"], 128)          # int(160*0.8)，与单机版cut一致
        self.assertEqual(w["holdout"]["days"], 32)         # 最后20%一次验证
        self.assertEqual(w["cum_bounds"], [32, 64, 96, 128])  # 3折前向验证边界
        self.assertEqual(w["folds"][0]["test"]["start"],
                         (date(2026, 5, 1) + timedelta(days=32)).isoformat())
        self.assertEqual(w["scenario"]["days"], 14)
        self.assertEqual(w["scenario"]["end"], "2026-10-07")

    def test_boundary_too_short_rejected(self):
        with self.assertRaises(ValueError):
            alg.compute_windows(date(2026, 5, 1), date(2026, 6, 20))


class TestStatsRidge(FederatedTestCase):
    def test_fit_from_stats_matches_direct(self):
        rng_rows = []
        import random
        rng = random.Random(7)
        for i in range(200):
            x = [rng.uniform(100, 900), rng.uniform(4, 12), rng.uniform(10, 35), rng.uniform(0, 4)]
            y = 120 + 0.5 * x[0] + 22 * x[1] + 3.8 * x[2] + 30 * x[3] + rng.gauss(0, 8)
            rng_rows.append((date(2026, 1, 1) + timedelta(days=i), x, y))
        st = alg.stats_from_rows(rng_rows)
        model = alg.fit_ridge(st, (0, 1, 2, 3), 1.0)
        # 直接法：原始行上标准化后解 (ZᵀZ+αI)w=Zᵀ(y−ȳ)
        n = len(rng_rows)
        mean = [sum(r[1][j] for r in rng_rows) / n for j in range(4)]
        scale = []
        for j in range(4):
            var = sum((r[1][j] - mean[j]) ** 2 for r in rng_rows) / n
            scale.append(math.sqrt(var))
        ybar = sum(r[2] for r in rng_rows) / n
        A = [[0.0] * 4 for _ in range(4)]
        b = [0.0] * 4
        for _, x, y in rng_rows:
            z = [(x[j] - mean[j]) / scale[j] for j in range(4)]
            for a in range(4):
                b[a] += z[a] * (y - ybar)
                for c in range(4):
                    A[a][c] += z[a] * z[c]
        for j in range(4):
            A[j][j] += 1.0
        w = alg._solve(A, b)
        for j, name in enumerate(alg.FEATURES):
            self.assertAlmostEqual(model["coef"][name], w[j] / scale[j], places=9)
        intercept = ybar - sum(mean[j] * w[j] / scale[j] for j in range(4))
        self.assertAlmostEqual(model["intercept"], intercept, places=7)
        # SSE 与直接残差一致
        sse_direct = sum((y - (intercept + sum((w[j] / scale[j]) * x[j] for j in range(4)))) ** 2
                         for _, x, y in rng_rows)
        self.assertAlmostEqual(alg.sse_stats(st, model), sse_direct, places=4)


class TestParity(FederatedTestCase):
    """统计量聚合 == 原始样本合并：联合结果与单机对照实现一致。"""

    def test_federated_equals_merged_reference(self):
        store = Store(self.subdir("store"))
        run = make_run(store)
        complete_stage1(store, run)
        self.assertEqual(run["open_stage"], proto.STAGE_HOLDOUT)
        complete_stage2(store, run)
        self.assertEqual(run["status"], proto.STATUS_AWAITING)
        merged = merged_analyze([r for s in SITES for r in site_records(s)], PARAMS)
        result = run["result"]
        self.assertEqual(result["selection"]["features"], merged["selection"]["features"])
        self.assertEqual(result["selection"]["alpha"], merged["selection"]["alpha"])
        self.assertAlmostEqual(result["selection"]["cv_rmse"], merged["selection"]["cv_rmse"], places=8)
        for name, coef in merged["model"]["coef"].items():
            self.assertAlmostEqual(result["model"]["coef"][name], coef, places=6)
        self.assertAlmostEqual(result["model"]["intercept"], merged["model"]["intercept"], places=4)
        self.assertAlmostEqual(result["holdout"]["rmse"], merged["holdout"]["rmse"], places=6)
        self.assertAlmostEqual(result["holdout"]["r2"], merged["holdout"]["r2"], places=8)
        self.assertAlmostEqual(result["scenario"]["scenario_kwh"], merged["scenario"]["scenario_kwh"], places=4)
        self.assertEqual([s["n"] for s in result["holdout"]["sites"]], [32, 32, 32])


# ---------------------------------------------------------------- 协议：屏障/幂等/防串
class TestProtocol(FederatedTestCase):
    def test_barrier_waits_absence_not_zero(self):
        store = Store(self.subdir("store"))
        run = make_run(store)
        store.submit(stage1_envelope(run, "S1"))
        store.submit(stage1_envelope(run, "S2"))
        self.assertEqual(run["status"], proto.STATUS_COLLECTING)
        self.assertEqual(run["open_stage"], proto.STAGE_TRAIN)
        self.assertEqual(run["snapshots"], {})  # 缺席不当零：未齐备不聚合
        detail = store.run_detail(run["run_id"])
        self.assertEqual(detail["missing"], ["S3"])
        store.submit(stage1_envelope(run, "S3"))
        self.assertEqual(run["open_stage"], proto.STAGE_HOLDOUT)
        self.assertIn(proto.STAGE_TRAIN, run["snapshots"])

    def test_idempotent_retry_not_double_counted(self):
        store = Store(self.subdir("store"))
        run = make_run(store)
        env = stage1_envelope(run, "S1")
        first = store.submit(env)
        second = store.submit(env)
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(len(run["contributions"][proto.STAGE_TRAIN]), 1)

    def test_same_seq_different_content_rejected(self):
        store = Store(self.subdir("store"))
        run = make_run(store)
        store.submit(stage1_envelope(run, "S1"))
        tampered = stage1_envelope(run, "S1", seq=1)
        tampered["payload"]["boundary_records"] += 1
        tampered["content_hash"] = proto.content_hash(tampered["payload"])
        with self.assertRaises(StoreError) as ctx:
            store.submit(tampered)
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(ctx.exception.code, "seq_conflict")

    def test_hash_mismatch_rejected(self):
        store = Store(self.subdir("store"))
        run = make_run(store)
        env = stage1_envelope(run, "S1")
        env["content_hash"] = "0" * 64
        with self.assertRaises(StoreError) as ctx:
            store.submit(env)
        self.assertEqual(ctx.exception.code, "hash_mismatch")

    def test_late_envelope_rejected_no_cross_stage(self):
        store = Store(self.subdir("store"))
        run = make_run(store)
        late = stage1_envelope(run, "S1")
        complete_stage1(store, run)
        snapshot_hash = proto.content_hash(run["snapshots"][proto.STAGE_TRAIN])
        with self.assertRaises(StoreError) as ctx:
            store.submit(late)  # 旧阶段迟到回包
        self.assertEqual(ctx.exception.code, "stage_closed")
        self.assertEqual(proto.content_hash(run["snapshots"][proto.STAGE_TRAIN]), snapshot_hash)
        self.assertEqual(len(run["contributions"][proto.STAGE_TRAIN]), 3)
        # 当前阶段不受影响，可正常推进
        store.submit(stage2_envelope(store, run, "S1"))
        self.assertEqual(len(run["contributions"][proto.STAGE_HOLDOUT]), 1)

    def test_version_mismatch_rejected(self):
        store = Store(self.subdir("store"))
        run = make_run(store)
        env = stage1_envelope(run, "S1")
        env["payload"]["sample_version"] = "f" * 64
        env["content_hash"] = proto.content_hash(env["payload"])
        with self.assertRaises(StoreError) as ctx:
            store.submit(env)
        self.assertEqual(ctx.exception.code, "version_mismatch")

    def test_unknown_site_rejected(self):
        store = Store(self.subdir("store"))
        run = make_run(store)
        env = stage1_envelope(run, "S1")
        env["site_id"] = "S9"
        with self.assertRaises(StoreError) as ctx:
            store.submit(env)
        self.assertEqual(ctx.exception.status, 403)

    def test_min_records_enforced_globally(self):
        store = Store(self.subdir("store"))
        short_sites = SITES[:2]
        parts = [{"site_id": s, "sample_version": sample_version(site_records(s, days=10))}
                 for s in short_sites]
        run = store.create_run(PARAMS, {"start": START, "end": "2026-07-09"}, parts, "test")
        for s in short_sites:
            parsed = parse_records(site_records(s, days=10))
            payload = train_payload(run["manifest"], parsed, s,
                                    sample_version(site_records(s, days=10)))
            store.submit(proto.build_envelope(run["run_id"], proto.STAGE_TRAIN, s, 1, payload))
        self.assertEqual(run["status"], proto.STATUS_FAILED)
        self.assertIn("至少需要70天", run["failure"]["reason"])

    def test_concurrent_advance_and_late_replay(self):
        store = Store(self.subdir("store"))
        run = make_run(store)
        late_envs = [stage1_envelope(run, s) for s in SITES]
        complete_stage1(store, run)
        snapshot_hash = proto.content_hash(run["snapshots"][proto.STAGE_TRAIN])
        outcomes = {"accepted": 0, "stage_closed": 0, "other": 0}
        barrier = threading.Barrier(len(SITES) * 2)

        def genuine(site):
            barrier.wait()
            try:
                store.submit(stage2_envelope(store, run, site))
                outcomes["accepted"] += 1
            except StoreError:
                outcomes["other"] += 1

        def late(env):
            barrier.wait()
            try:
                store.submit(env)
                outcomes["other"] += 1
            except StoreError as exc:
                if exc.code == "stage_closed":
                    outcomes["stage_closed"] += 1
                else:
                    outcomes["other"] += 1

        threads = [threading.Thread(target=genuine, args=(s,)) for s in SITES]
        threads += [threading.Thread(target=late, args=(e,)) for e in late_envs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(outcomes["accepted"], 3)
        self.assertEqual(outcomes["stage_closed"], 3)
        self.assertEqual(outcomes["other"], 0)
        self.assertEqual(run["status"], proto.STATUS_AWAITING)
        self.assertEqual(proto.content_hash(run["snapshots"][proto.STAGE_TRAIN]), snapshot_hash)
        self.assertEqual(len(run["contributions"][proto.STAGE_HOLDOUT]), 3)


# ---------------------------------------------------------------- 批准门与失败
class TestApproval(FederatedTestCase):
    def test_approval_gate_and_distribution(self):
        store = Store(self.subdir("store"))
        run = make_run(store)
        with self.assertRaises(StoreError) as ctx:
            store.approve(run["run_id"], "集团能源组")
        self.assertEqual(ctx.exception.code, "approval_not_ready")
        with self.assertRaises(StoreError) as ctx:
            store.result(run["run_id"])
        self.assertEqual(ctx.exception.code, "result_not_public")
        complete_stage1(store, run)
        with self.assertRaises(StoreError):
            store.approve(run["run_id"], "集团能源组")  # 留出误差未齐备不可批准
        complete_stage2(store, run)
        with self.assertRaises(StoreError) as ctx:
            store.result(run["run_id"])  # 半成品不公开
        self.assertEqual(ctx.exception.code, "result_not_public")
        store.approve(run["run_id"], "集团能源组", "复核通过")
        package = store.result(run["run_id"])
        self.assertEqual(package["approval"]["approver"], "集团能源组")
        self.assertEqual(package["result"]["manifest_hash"], run["manifest"]["manifest_hash"])
        with self.assertRaises(StoreError):
            store.approve(run["run_id"], "集团能源组")  # 重复批准拒收

    def test_r2_undefined_when_holdout_constant(self):
        def const_holdout_records(site_tag):
            rows = []
            for i in range(160):
                day = date(2026, 5, 1) + timedelta(days=i)
                kwh = (800 + (i % 13) * 7) if i < 128 else 500.0
                rows.append({"day": day.isoformat(), "line": site_tag, "units": 500 + (i % 40) * 3,
                             "run_hours": 10.0, "temp_c": 20.0, "idle_hours": 1.0, "kwh": float(kwh)})
            return rows
        store = Store(self.subdir("store"))
        sites = SITES[:2]
        recs = {s: const_holdout_records(s) for s in sites}
        parts = [{"site_id": s, "sample_version": sample_version(recs[s])} for s in sites]
        run = store.create_run(PARAMS, {"start": START, "end": END}, parts, "test")
        for s in sites:
            parsed = parse_records(recs[s])
            payload = train_payload(run["manifest"], parsed, s, sample_version(recs[s]))
            store.submit(proto.build_envelope(run["run_id"], proto.STAGE_TRAIN, s, 1, payload))
        for s in sites:
            model = run["snapshots"][proto.STAGE_TRAIN]["model"]
            parsed = parse_records(recs[s])
            payload = holdout_payload(run["manifest"], parsed, s, sample_version(recs[s]), model)
            store.submit(proto.build_envelope(run["run_id"], proto.STAGE_HOLDOUT, s, 1, payload))
        self.assertEqual(run["status"], proto.STATUS_AWAITING)
        self.assertFalse(run["result"]["holdout"]["r2_defined"])
        self.assertIsNone(run["result"]["holdout"]["r2"])
        self.assertTrue(any("R²不可定义" in w for w in run["result"]["warnings"]))

    def test_scenario_window_empty_fails_gracefully(self):
        # 记录覆盖训练窗与部分留出窗，但不含边界最后14天（情景窗为空）
        def sparse_records(site_tag):
            rows = []
            for i in list(range(100)) + list(range(128, 141)):
                day = date(2026, 5, 1) + timedelta(days=i)
                rows.append({"day": day.isoformat(), "line": site_tag,
                             "units": 500 + (i % 40) * 3, "run_hours": 10.0,
                             "temp_c": 20.0, "idle_hours": 1.0, "kwh": float(800 + (i % 13) * 7)})
            return rows
        store = Store(self.subdir("store"))
        sites = SITES[:2]
        recs = {s: sparse_records(s) for s in sites}
        parts = [{"site_id": s, "sample_version": sample_version(recs[s])} for s in sites]
        run = store.create_run(PARAMS, {"start": START, "end": END}, parts, "test")
        for s in sites:
            parsed = parse_records(recs[s])
            payload = train_payload(run["manifest"], parsed, s, sample_version(recs[s]))
            store.submit(proto.build_envelope(run["run_id"], proto.STAGE_TRAIN, s, 1, payload))
        self.assertEqual(run["open_stage"], proto.STAGE_HOLDOUT)
        for s in sites:
            model = run["snapshots"][proto.STAGE_TRAIN]["model"]
            parsed = parse_records(recs[s])
            payload = holdout_payload(run["manifest"], parsed, s, sample_version(recs[s]), model)
            store.submit(proto.build_envelope(run["run_id"], proto.STAGE_HOLDOUT, s, 1, payload))
        self.assertEqual(run["status"], proto.STATUS_FAILED)
        self.assertIn("情景窗", run["failure"]["reason"])

    def test_site_can_fail_run_with_provenance(self):
        store = Store(self.subdir("store"))
        run = make_run(store)
        with self.assertRaises(StoreError):
            store.fail(run["run_id"], "S9", "非参与厂")
        store.fail(run["run_id"], "S2", "本地数据校验失败：日期重复")
        self.assertEqual(run["status"], proto.STATUS_FAILED)
        self.assertEqual(run["failure"]["site_id"], "S2")
        with self.assertRaises(StoreError) as ctx:
            store.submit(stage1_envelope(run, "S1"))
        self.assertEqual(ctx.exception.code, "run_failed")


# ---------------------------------------------------------------- 恢复
class TestRecovery(FederatedTestCase):
    def test_resume_replays_barrier_after_crash(self):
        data_dir = self.subdir("store")
        store = Store(data_dir)
        run = make_run(store)
        complete_stage1(store, run)
        expected = proto.content_hash(run["snapshots"][proto.STAGE_TRAIN])
        # 模拟崩溃于屏障计算中：贡献已落盘，快照缺失
        path = data_dir / "runs" / f"{run['run_id']}.json"
        disk = json.loads(path.read_text(encoding="utf-8"))
        disk["snapshots"] = {}
        disk["open_stage"] = proto.STAGE_TRAIN
        disk["status"] = proto.STATUS_COLLECTING
        path.write_text(json.dumps(disk, ensure_ascii=False), encoding="utf-8")
        store2 = Store(data_dir)  # 协调服务重启：从磁盘恢复
        outcome = store2.resume(run["run_id"])
        self.assertTrue(outcome["advanced"])
        restored = store2.runs[run["run_id"]]
        self.assertEqual(restored["open_stage"], proto.STAGE_HOLDOUT)
        self.assertEqual(proto.content_hash(restored["snapshots"][proto.STAGE_TRAIN]), expected)

    def test_coordinator_restart_keeps_confirmed_stage(self):
        data_dir = self.subdir("store")
        server_a = HttpServer(data_dir)
        try:
            st, body = http("POST", server_a.url + "/api/runs",
                            {"params": PARAMS, "boundary": {"start": START, "end": END},
                             "participants": participants()})
            self.assertEqual(st, 201)
            run_id = body["run_id"]
            _, detail = http("GET", server_a.url + f"/api/runs/{run_id}")
            manifest = detail["manifest"]
            for s in SITES:
                parsed = parse_records(site_records(s))
                payload = train_payload(manifest, parsed, s, site_version(s))
                env = proto.build_envelope(run_id, proto.STAGE_TRAIN, s, 1, payload)
                st, resp = http("POST", server_a.url + f"/api/runs/{run_id}/contributions", env)
                self.assertEqual(st, 200, resp)
            _, detail = http("GET", server_a.url + f"/api/runs/{run_id}")
            self.assertEqual(detail["open_stage"], proto.STAGE_HOLDOUT)
        finally:
            server_a.stop()
        # 协调服务中断后重启：已确认阶段保留，继续汇总
        server_b = HttpServer(data_dir)
        try:
            st, detail = http("GET", server_b.url + f"/api/runs/{run_id}")
            self.assertEqual(st, 200)
            self.assertEqual(detail["open_stage"], proto.STAGE_HOLDOUT)
            self.assertIsNotNone(detail["selection"])
            self.assertEqual(len(detail["contributions"][proto.STAGE_TRAIN]), 3)
            # 旧阶段迟到回包在重启后仍被拒
            parsed = parse_records(site_records("S1"))
            payload = train_payload(detail["manifest"], parsed, "S1", site_version("S1"))
            env = proto.build_envelope(run_id, proto.STAGE_TRAIN, "S1", 1, payload)
            st, resp = http("POST", server_b.url + f"/api/runs/{run_id}/contributions", env)
            self.assertEqual(st, 409)
            self.assertEqual(resp["error"]["code"], "stage_closed")
            # 阶段二齐备后进入待批准
            model = server_b.store.runs[run_id]["snapshots"][proto.STAGE_TRAIN]["model"]
            for s in SITES:
                parsed = parse_records(site_records(s))
                payload = holdout_payload(detail["manifest"], parsed, s, site_version(s), model)
                env = proto.build_envelope(run_id, proto.STAGE_HOLDOUT, s, 1, payload)
                st, resp = http("POST", server_b.url + f"/api/runs/{run_id}/contributions", env)
                self.assertEqual(st, 200, resp)
            _, detail = http("GET", server_b.url + f"/api/runs/{run_id}")
            self.assertEqual(detail["status"], proto.STATUS_AWAITING)
        finally:
            server_b.stop()


# ---------------------------------------------------------------- 演练端点
class TestDrills(FederatedTestCase):
    def test_late_and_conflict_drill_endpoints(self):
        server = HttpServer(self.subdir("store"))
        self.addCleanup(server.stop)
        st, body = http("POST", server.url + "/api/runs",
                        {"params": PARAMS, "boundary": {"start": START, "end": END},
                         "participants": participants()})
        run_id = body["run_id"]
        _, detail = http("GET", server.url + f"/api/runs/{run_id}")
        manifest = detail["manifest"]
        # 阶段未关闭时演练迟到 → 提示不可演练
        st, resp = http("POST", server.url + f"/api/runs/{run_id}/drill/late",
                        {"stage": proto.STAGE_TRAIN, "site_id": "S1"})
        self.assertEqual(st, 400)
        for s in SITES:
            parsed = parse_records(site_records(s))
            payload = train_payload(manifest, parsed, s, site_version(s))
            env = proto.build_envelope(run_id, proto.STAGE_TRAIN, s, 1, payload)
            http("POST", server.url + f"/api/runs/{run_id}/contributions", env)
        # 迟到回包演练：重投已关闭阶段 → 阶段栅栏拒收
        st, resp = http("POST", server.url + f"/api/runs/{run_id}/drill/late",
                        {"stage": proto.STAGE_TRAIN, "site_id": "S1"})
        self.assertEqual(st, 200)
        self.assertEqual(resp["outcome"]["code"], "stage_closed")
        # 同号异内容演练：同序号改内容重投 → 拒收
        parsed = parse_records(site_records("S1"))
        model = server.store.runs[run_id]["snapshots"][proto.STAGE_TRAIN]["model"]
        payload = holdout_payload(manifest, parsed, "S1", site_version("S1"), model)
        env = proto.build_envelope(run_id, proto.STAGE_HOLDOUT, "S1", 1, payload)
        http("POST", server.url + f"/api/runs/{run_id}/contributions", env)
        st, resp = http("POST", server.url + f"/api/runs/{run_id}/drill/conflict",
                        {"stage": proto.STAGE_HOLDOUT, "site_id": "S1"})
        self.assertEqual(st, 200)
        self.assertEqual(resp["outcome"]["code"], "seq_conflict")
        _, detail = http("GET", server.url + f"/api/runs/{run_id}")
        drill_events = [e for e in detail["events"] if e["kind"] == "drill"]
        self.assertEqual(len(drill_events), 2)
        # 正常贡献未被演练污染
        self.assertEqual(len(detail["contributions"][proto.STAGE_HOLDOUT]), 1)


# ---------------------------------------------------------------- 端到端（代理+HTTP）
class TestEndToEnd(FederatedTestCase):
    def _prepare(self, sites=SITES):
        data_dir = self.subdir("coord")
        csvs = {}
        for s in sites:
            path = self.subdir(f"csv-{s}") / f"{s}.csv"
            write_csv(path, site_records(s))
            csvs[s] = path
        server = HttpServer(data_dir)
        self.addCleanup(server.stop)
        return server, csvs

    def _start_agents(self, server, csvs, sites, with_control=False):
        runners = {}
        for s in sites:
            port = free_port() if with_control else None
            runner = AgentRunner(s, csvs[s], server.url, self.subdir(f"state-{s}"),
                                 control_port=port)
            runner.start()
            runners[s] = runner
            self.addCleanup(runner.stop)
        return runners

    def _create_run(self, server, sites=SITES):
        st, body = http("POST", server.url + "/api/runs",
                        {"params": PARAMS, "boundary": {"start": START, "end": END},
                         "participants": participants(sites)})
        self.assertEqual(st, 201, body)
        return body["run_id"]

    def _await_status(self, server, run_id, status, timeout=20):
        return wait_until(lambda: http("GET", server.url + f"/api/runs/{run_id}")[1].get("status") == status,
                          timeout=timeout)

    def test_full_run_with_agents(self):
        server, csvs = self._prepare()
        self._start_agents(server, csvs, SITES)
        run_id = self._create_run(server)
        self.assertTrue(self._await_status(server, run_id, proto.STATUS_AWAITING),
                        "运行未在时限内进入待批准")
        _, detail = http("GET", server.url + f"/api/runs/{run_id}")
        for stage in (proto.STAGE_TRAIN, proto.STAGE_HOLDOUT):
            self.assertEqual(len(detail["contributions"][stage]), 3)
            for meta in detail["contributions"][stage].values():
                self.assertEqual(meta["seq"], 1)  # 重试未重复累计
        # 批准前结果包不公开
        st, _ = http("GET", server.url + f"/api/runs/{run_id}/result")
        self.assertEqual(st, 409)
        st, _ = http("POST", server.url + f"/api/runs/{run_id}/approve",
                     {"approver": "集团能源组"})
        self.assertEqual(st, 200)
        st, package = http("GET", server.url + f"/api/runs/{run_id}/result")
        self.assertEqual(st, 200)
        merged = merged_analyze([r for s in SITES for r in site_records(s)], PARAMS)
        self.assertAlmostEqual(package["result"]["holdout"]["rmse"],
                               merged["holdout"]["rmse"], places=6)
        self.assertEqual(package["result"]["selection"]["features"],
                         merged["selection"]["features"])

    def test_disconnect_drill_pauses_barrier_then_recovers(self):
        server, csvs = self._prepare()
        self._start_agents(server, csvs, SITES, with_control=True)
        # 等S2注册控制地址
        self.assertTrue(wait_until(
            lambda: any(s["site_id"] == "S2" for s in http("GET", server.url + "/api/sites")[1]["sites"]),
            timeout=10))
        # 先令S2断线，再发起运行：S1/S3齐备而S2缺席，屏障必须等待
        st, resp = http("POST", server.url + "/api/sites/S2/drill/disconnect", {"seconds": 4})
        self.assertEqual(st, 200, resp)
        run_id = self._create_run(server)
        def s2_missing():
            _, detail = http("GET", server.url + f"/api/runs/{run_id}")
            return (detail["open_stage"] == proto.STAGE_TRAIN
                    and detail["missing"] == ["S2"]
                    and len(detail["contributions"][proto.STAGE_TRAIN]) == 2)
        self.assertTrue(wait_until(s2_missing, timeout=10), "S2断线期间屏障未按预期等待")
        _, detail = http("GET", server.url + f"/api/runs/{run_id}")
        self.assertIsNone(detail["selection"])  # 未齐备不聚合
        # 断线恢复后自动补齐并推进至待批准
        self.assertTrue(self._await_status(server, run_id, proto.STATUS_AWAITING, timeout=25))

    def test_coordinator_restart_mid_run_recovers(self):
        server, csvs = self._prepare()
        self._start_agents(server, csvs, SITES)
        run_id = self._create_run(server)
        # 等阶段一确认
        self.assertTrue(wait_until(
            lambda: http("GET", server.url + f"/api/runs/{run_id}")[1].get("open_stage") == proto.STAGE_HOLDOUT,
            timeout=20))
        port = server.port
        server.stop()  # 协调服务中断
        time.sleep(0.3)  # 厂站进入重试
        server_b = HttpServer(Path(self.tmp) / "coord", port=port)  # 同端口恢复
        self.addCleanup(server_b.stop)
        _, detail = http("GET", server_b.url + f"/api/runs/{run_id}")
        self.assertEqual(detail["open_stage"], proto.STAGE_HOLDOUT)
        self.assertEqual(len(detail["contributions"][proto.STAGE_TRAIN]), 3)
        self.assertTrue(self._await_status(server_b, run_id, proto.STATUS_AWAITING, timeout=25))
        _, detail = http("GET", server_b.url + f"/api/runs/{run_id}")
        self.assertEqual(len(detail["contributions"][proto.STAGE_TRAIN]), 3)
        self.assertEqual(len(detail["contributions"][proto.STAGE_HOLDOUT]), 3)
        for meta in detail["contributions"][proto.STAGE_TRAIN].values():
            self.assertEqual(meta["seq"], 1)

    def test_agent_restart_no_duplicate(self):
        server, csvs = self._prepare()
        runners = self._start_agents(server, csvs, SITES)
        run_id = self._create_run(server)
        # 等S1阶段一确认后重启S1代理
        self.assertTrue(wait_until(
            lambda: "S1" in http("GET", server.url + f"/api/runs/{run_id}")[1]
            ["contributions"][proto.STAGE_TRAIN], timeout=20))
        runners["S1"].stop()
        time.sleep(0.2)
        runners["S1"] = AgentRunner("S1", csvs["S1"], server.url,
                                    Path(self.tmp) / "state-S1").start()
        self.addCleanup(runners["S1"].stop)
        self.assertTrue(self._await_status(server, run_id, proto.STATUS_AWAITING))
        _, detail = http("GET", server.url + f"/api/runs/{run_id}")
        self.assertEqual(len(detail["contributions"][proto.STAGE_TRAIN]), 3)
        self.assertEqual(detail["contributions"][proto.STAGE_TRAIN]["S1"]["seq"], 1)


class TestSampleVersion(FederatedTestCase):
    def test_version_stable_across_csv_roundtrip(self):
        for s in SITES:
            records = site_records(s)
            path = self.subdir(f"csv-{s}") / f"{s}.csv"
            write_csv(path, records)
            self.assertEqual(sample_version(load_records(path)), sample_version(records))


if __name__ == "__main__":
    unittest.main()
