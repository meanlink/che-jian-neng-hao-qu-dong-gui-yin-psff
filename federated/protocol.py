"""联合归因协议：冻结清单、阶段定义、贡献信封与内容指纹。

- 开算冻结：参与厂、各厂样本版本、日期边界与派生时间窗全部写入清单并给出指纹；
- 阶段屏障：train_stats → holdout_eval，同阶段各厂齐备才推进，缺席不当零；
- 幂等与防串：贡献按 (run, stage, site) 归键，重试同内容幂等接受，
  同号异内容拒收，旧阶段迟到回包按阶段栅栏拒收。
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime

from .algorithm import ALPHAS, FEATURES, FEATURE_LABELS, compute_windows

STAGE_TRAIN = "train_stats"
STAGE_HOLDOUT = "holdout_eval"
STAGES = (STAGE_TRAIN, STAGE_HOLDOUT)
STAGE_LABELS = {STAGE_TRAIN: "训练统计汇总", STAGE_HOLDOUT: "留出误差汇总"}

STATUS_COLLECTING = "collecting"        # 待汇总
STATUS_AWAITING = "awaiting_approval"   # 待批准
STATUS_APPROVED = "approved"            # 已完成（可分发）
STATUS_FAILED = "failed"                # 失败
STATUS_LABELS = {
    STATUS_COLLECTING: "待汇总",
    STATUS_AWAITING: "待批准",
    STATUS_APPROVED: "已完成",
    STATUS_FAILED: "失败",
}

PROTOCOL_VERSION = "fed-ridge/1"


def canonical(obj) -> str:
    """确定性JSON编码：排序键、紧凑分隔、禁止NaN/Inf，用于指纹与内容哈希。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(payload) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_manifest(run_id, params, boundary, participants, created_by=""):
    """生成冻结清单：参与厂+样本版本+日期边界+派生时间窗+选参网格。"""
    start = date.fromisoformat(str(boundary["start"]))
    end = date.fromisoformat(str(boundary["end"]))
    windows = compute_windows(start, end)
    manifest = {
        "run_id": run_id,
        "created_at": now_iso(),
        "created_by": created_by,
        "protocol": PROTOCOL_VERSION,
        "params": params,
        "boundary": {"start": start.isoformat(), "end": end.isoformat()},
        "participants": [{"site_id": p["site_id"], "sample_version": p["sample_version"]} for p in participants],
        "features": list(FEATURES),
        "feature_labels": dict(FEATURE_LABELS),
        "alphas": list(ALPHAS),
        "windows": windows,
    }
    manifest["manifest_hash"] = content_hash(manifest)
    return manifest


def participant_ids(manifest):
    return [p["site_id"] for p in manifest["participants"]]


def participant_version(manifest, site_id):
    for p in manifest["participants"]:
        if p["site_id"] == site_id:
            return p["sample_version"]
    return None


def build_envelope(run_id, stage, site_id, seq, payload):
    """厂站贡献信封：序号seq在同(run,stage,site)下标识内容版本。"""
    return {
        "run_id": run_id,
        "stage": stage,
        "site_id": site_id,
        "seq": int(seq),
        "content_hash": content_hash(payload),
        "payload": payload,
    }


def envelope_errors(env):
    """信封结构浅校验，返回错误消息列表（空为通过）。"""
    errors = []
    if not isinstance(env, dict):
        return ["信封必须为对象"]
    for key in ("run_id", "stage", "site_id", "seq", "content_hash", "payload"):
        if key not in env:
            errors.append(f"信封缺少字段{key}")
    if errors:
        return errors
    if env["stage"] not in STAGES:
        errors.append(f"未知阶段{env['stage']}")
    if not isinstance(env["seq"], int) or env["seq"] < 1:
        errors.append("seq必须为≥1的整数")
    if not isinstance(env["content_hash"], str) or len(env["content_hash"]) != 64:
        errors.append("content_hash必须为64位十六进制")
    if not isinstance(env["payload"], dict):
        errors.append("payload必须为对象")
    return errors


def payload_finite(payload) -> bool:
    """递归检查payload内所有数值有限（防止NaN/Inf污染聚合）。"""
    if isinstance(payload, bool):
        return True
    if isinstance(payload, (int, float)):
        return math.isfinite(payload)
    if isinstance(payload, dict):
        return all(payload_finite(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(payload_finite(v) for v in payload)
    return payload is None or isinstance(payload, str)
