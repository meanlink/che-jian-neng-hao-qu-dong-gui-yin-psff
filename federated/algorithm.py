"""联合归因算法内核：充分统计量上的标准化岭回归。

单机原版见 backend/src/services/business.py 的 analyze()：
前 80% 时间训练、最后 20% 一次留出验证；训练内 3 折前向时间序列交叉验证，
搜索 2..k 个特征子集与 alpha∈{0.1,1,10}，逐折独立标准化，系数还原原始单位。

本模块把同一语义改写为可联邦执行的形式：
- 各厂本地只计算"充分统计量"（样本数、特征与目标的一阶/二阶和及交叉和），
  原始日样本不出站；
- 协调方把各厂统计量相加后，在统计量上完成与单机版一致的
  StandardScaler+Ridge 求解、时间折验证、留出评估与负荷情景；
- merged_analyze() 是同一语义的单机对照实现（按记录序号切分），
  用于校验"统计量聚合 == 原始样本合并"的正确性。

时间留出语义对齐方式：单机版按排序后的记录序号切分（cut=int(n*0.8)，
TimeSeriesSplit 在 cut 内取 3 段递增验证窗）。联合版在开算时冻结日期边界，
把同一比例换算为日期窗（训练窗天数=int(总天数*0.8)，折边界按天数取整），
各厂按日期窗各自切分；当各厂在边界内逐日齐备时，两者切分结果完全一致。
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from itertools import combinations

FEATURES = ["units", "run_hours", "temp_c", "idle_hours"]
FEATURE_LABELS = {"units": "产量", "run_hours": "运行小时", "temp_c": "环境温度", "idle_hours": "待机小时"}
TARGET = "kwh"
ALPHAS = (0.1, 1.0, 10.0)
MIN_DAYS = 70
TRAIN_FRACTION = 0.8
N_SPLITS = 3
SCENARIO_DAYS = 14
MIN_TARGET_STD = 1e-9
# 与单机版一致的业务范围校验
_BOUNDS = ((0.0, 10_000_000.0), (0.0, 24.0), (-100.0, 100.0), (0.0, 24.0))
_TARGET_BOUNDS = (0.0, 1_000_000_000.0)

METHOD_TEXT = (
    "前80%日期训练；训练内3折前向时间序列交叉验证搜索2至k个变量组合和α∈{0.1,1,10}。"
    "每折独立标准化，最后20%仅验证一次。"
    "集团联合版开算冻结参与厂、样本版本与日期边界，各厂原始日样本不出站，"
    "仅交换充分统计量与留出误差汇总，由协调方聚合求解。"
)


class DegenerateRun(Exception):
    """冻结边界与数据不符或目标无变化等无法继续联合计算的情形。"""


def _bounded(value, name, low, high):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}必须为数值") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name}必须介于{low}和{high}")
    return value


def parse_params(params):
    """与单机版一致的参数边界：max_features∈[2,4] 整数，scenario_pct∈[-30,30]。"""
    params = params or {}
    k = _bounded(params.get("max_features", 4), "解释变量数", 2, 4)
    if not k.is_integer():
        raise ValueError("解释变量数必须为整数")
    delta = _bounded(params.get("scenario_pct", 10), "情景负荷变化", -30, 30)
    return {"max_features": int(k), "scenario_pct": delta}


def parse_records(records, require_unique_days=True):
    """与单机版一致的记录校验：类型、有限性、业务范围，按日期排序。

    require_unique_days=True 时与单机版相同，拒绝重复日期（单厂日样本口径）；
    联合对照路径合并多厂数据时置 False——不同厂的同一日历日是不同样本，
    不属于单机版要防的"同厂重复记录导致权重失真"。
    """
    if not isinstance(records, list):
        raise ValueError("记录必须为列表")
    parsed = []
    for item in records:
        try:
            day = date.fromisoformat(str(item["day"]))
            vector = [float(item[key]) for key in FEATURES]
            target = float(item[TARGET])
        except (KeyError, TypeError, ValueError):
            raise ValueError("日期须为ISO日期，4个特征及kwh必须是数值") from None
        if not all(math.isfinite(v) for v in vector + [target]):
            raise ValueError("数值不能包含缺失、NaN或无穷")
        if not (
            _BOUNDS[0][0] <= vector[0] <= _BOUNDS[0][1]
            and _BOUNDS[1][0] <= vector[1] <= _BOUNDS[1][1]
            and _BOUNDS[2][0] <= vector[2] <= _BOUNDS[2][1]
            and _BOUNDS[3][0] <= vector[3] <= _BOUNDS[3][1]
            and _TARGET_BOUNDS[0] <= target <= _TARGET_BOUNDS[1]
        ):
            raise ValueError("日产量须在0至1000万，运行/待机小时0至24，温度−100至100℃，日电耗0至10亿kWh")
        parsed.append((day, vector, target))
    parsed.sort(key=lambda row: row[0])
    if require_unique_days and len({row[0] for row in parsed}) != len(parsed):
        raise ValueError("同一日期必须先聚合为一条记录，避免样本权重失真")
    return parsed


def compute_windows(start: date, end: date) -> dict:
    """由冻结日期边界推出训练/留出/三折/情景日期窗（全部确定，不再依赖数据）。"""
    if end <= start:
        raise ValueError("日期边界结束日必须晚于开始日")
    total_days = (end - start).days + 1
    if total_days < MIN_DAYS:
        raise ValueError(f"日期边界至少需要{MIN_DAYS}天，以保留时间验证窗")
    train_days = int(total_days * TRAIN_FRACTION)
    test_size = train_days // (N_SPLITS + 1)
    if test_size < 1:
        raise ValueError("训练窗过短，无法划分3折前向验证")
    test_starts = [train_days - (N_SPLITS - i) * test_size for i in range(N_SPLITS)]
    cum_bounds = test_starts + [train_days]  # 累计统计量边界（训练窗内天数偏移）

    def d(offset):
        return (start + timedelta(days=offset)).isoformat()

    folds = []
    for begin in test_starts:
        folds.append({"train": {"start": d(0), "end": d(begin - 1)},
                      "test": {"start": d(begin), "end": d(begin + test_size - 1)}})
    return {
        "boundary": {"start": start.isoformat(), "end": end.isoformat(), "total_days": total_days},
        "train": {"start": d(0), "end": d(train_days - 1), "days": train_days},
        "holdout": {"start": d(train_days), "end": d(total_days - 1), "days": total_days - train_days},
        "folds": folds,
        "cum_bounds": cum_bounds,
        "test_size": test_size,
        "scenario": {"start": d(total_days - SCENARIO_DAYS), "end": d(total_days - 1), "days": SCENARIO_DAYS},
    }


# ---------------------------------------------------------------- 充分统计量
# 统计块：n、特征一阶/二阶和、特征交叉和(上三角6项)、特征×目标和、目标一阶/二阶和。
# 由它可精确还原标准化设计矩阵的格拉姆矩阵与目标交叉项，无需任何原始样本。
PAIR_INDEX = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
STAT_KEYS = ("n", "sum_x", "sum_xx", "sum_xx_off", "sum_xy", "sum_y", "sum_y2")


def zero_stats():
    return {"n": 0, "sum_x": [0.0] * 4, "sum_xx": [0.0] * 4, "sum_xx_off": [0.0] * 6,
            "sum_xy": [0.0] * 4, "sum_y": 0.0, "sum_y2": 0.0}


def stats_add_row(st, vector, target):
    st["n"] += 1
    for j in range(4):
        st["sum_x"][j] += vector[j]
        st["sum_xx"][j] += vector[j] * vector[j]
        st["sum_xy"][j] += vector[j] * target
    for (a, b), ref in zip(PAIR_INDEX, range(6)):
        st["sum_xx_off"][ref] += vector[a] * vector[b]
    st["sum_y"] += target
    st["sum_y2"] += target * target
    return st


def stats_from_rows(rows):
    st = zero_stats()
    for _, vector, target in rows:
        stats_add_row(st, vector, target)
    return st


def _stats_combine(a, b, sign):
    out = zero_stats()
    out["n"] = a["n"] + sign * b["n"]
    for key in ("sum_x", "sum_xx", "sum_xx_off", "sum_xy"):
        out[key] = [x + sign * y for x, y in zip(a[key], b[key])]
    out["sum_y"] = a["sum_y"] + sign * b["sum_y"]
    out["sum_y2"] = a["sum_y2"] + sign * b["sum_y2"]
    return out


def stats_add(a, b):
    return _stats_combine(a, b, 1)


def stats_sub(a, b):
    return _stats_combine(a, b, -1)


def stats_finite(st):
    values = list(st["sum_x"]) + list(st["sum_xx"]) + list(st["sum_xx_off"]) + list(st["sum_xy"])
    values += [st["sum_y"], st["sum_y2"]]
    return st["n"] >= 0 and all(math.isfinite(v) for v in values)


def moments(st):
    """由统计块求均值与中心化二阶矩（数值稳定形式）。"""
    n = st["n"]
    if n <= 0:
        raise DegenerateRun("统计段无样本")
    mean = [s / n for s in st["sum_x"]]
    ybar = st["sum_y"] / n
    m2 = [[0.0] * 4 for _ in range(4)]
    for j in range(4):
        m2[j][j] = st["sum_xx"][j] - n * mean[j] * mean[j]
    for (a, b), ref in zip(PAIR_INDEX, range(6)):
        m2[a][b] = m2[b][a] = st["sum_xx_off"][ref] - n * mean[a] * mean[b]
    m2xy = [st["sum_xy"][j] - n * mean[j] * ybar for j in range(4)]
    m2yy = st["sum_y2"] - n * ybar * ybar
    return n, mean, ybar, m2, m2xy, m2yy


def _solve(matrix, rhs):
    """部分主元高斯消元（特征数≤4，无需第三方库）。"""
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            raise DegenerateRun("设计矩阵奇异，无法求解岭回归")
        aug[col], aug[piv] = aug[piv], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[row][c] -= factor * aug[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (aug[i][n] - sum(aug[i][j] * out[j] for j in range(i + 1, n))) / aug[i][i]
    return out


def fit_ridge(st, feature_idx, alpha):
    """在统计量上复现 StandardScaler+Ridge(fit_intercept=True)。

    标准化器按该段统计量拟合（有偏方差，零方差特征尺度取1，与sklearn一致）；
    岭回归对标准化特征求解 (ZᵀZ+αI)w=Zᵀ(y−ȳ)，再还原原始单位系数与截距。
    """
    n, mean, ybar, m2, m2xy, _ = moments(st)
    scale = {}
    for j in feature_idx:
        s = math.sqrt(max(m2[j][j], 0.0) / n)
        scale[j] = s if s > 0.0 else 1.0
    p = len(feature_idx)
    matrix = [[m2[a][b] / (scale[a] * scale[b]) + (alpha if i == k else 0.0)
               for k, b in enumerate(feature_idx)] for i, a in enumerate(feature_idx)]
    rhs = [m2xy[a] / scale[a] for a in feature_idx]
    w = _solve(matrix, rhs)
    coef = {FEATURES[a]: w[i] / scale[a] for i, a in enumerate(feature_idx)}
    intercept = ybar - sum(mean[a] * coef[FEATURES[a]] for a in feature_idx)
    return {"features": [FEATURES[a] for a in feature_idx], "intercept": intercept, "coef": coef}


def predict(model, vector):
    """vector 为与 FEATURES 对齐的4元特征列表。"""
    total = model["intercept"]
    for name, beta in model["coef"].items():
        total += beta * vector[FEATURES.index(name)]
    return total


def sse_stats(st, model):
    """留出/验证段平方误差和：中心化形式，避免大数相消。"""
    n, mean, ybar, m2, m2xy, m2yy = moments(st)
    idx = [FEATURES.index(name) for name in model["features"]]
    beta = [model["coef"][name] for name in model["features"]]
    quad = m2yy
    for b, j in zip(beta, idx):
        quad -= 2.0 * b * m2xy[j]
    for bi, a in zip(beta, idx):
        for bj, b in zip(beta, idx):
            quad += bi * bj * m2[a][b]
    resid = ybar - model["intercept"] - sum(b * mean[j] for b, j in zip(beta, idx))
    return max(quad + n * resid * resid, 0.0)


def select_and_fit(cum_stats, k):
    """训练窗累计统计量(4个边界)上的3折前向CV选参 + 全训练窗重拟合。

    折 i 训练段=累计到第 i 个边界；验证段=相邻边界差分。候选顺序与单机版一致：
    子集大小 2..k → 组合字典序 → α∈(0.1,1,10)，严格小于才替换，平局取先见。
    """
    segments = [cum_stats[0]] + [stats_sub(cum_stats[i + 1], cum_stats[i]) for i in range(N_SPLITS)]
    for seg in segments:
        if seg["n"] <= 0:
            raise DegenerateRun("时间折段在冻结边界内无样本，无法执行前向验证")
    best = None
    candidates = []
    for count in range(2, int(k) + 1):
        for feature_idx in combinations(range(4), count):
            for alpha in ALPHAS:
                errors = []
                for i in range(N_SPLITS):
                    train_st = cum_stats[i]
                    valid_st = stats_sub(cum_stats[i + 1], cum_stats[i])
                    model = fit_ridge(train_st, feature_idx, alpha)
                    errors.append(math.sqrt(sse_stats(valid_st, model) / valid_st["n"]))
                cv_rmse = sum(errors) / len(errors)
                candidate = {"features": [FEATURES[j] for j in feature_idx], "alpha": alpha, "cv_rmse": cv_rmse}
                candidates.append(candidate)
                if best is None or cv_rmse < best["cv_rmse"]:
                    best = candidate
    final_idx = tuple(FEATURES.index(name) for name in best["features"])
    model = fit_ridge(cum_stats[-1], final_idx, best["alpha"])
    return best, candidates, model


def holdout_metrics(site_evals):
    """各厂留出误差汇总：全局RMSE与R²（留出期目标恒定时R²不可定义）。"""
    n = sum(e["n"] for e in site_evals)
    if n <= 0:
        raise DegenerateRun("留出窗在冻结边界内无样本")
    sse = sum(e["sse"] for e in site_evals)
    sum_y = sum(e["sum_y"] for e in site_evals)
    sum_y2 = sum(e["sum_y2"] for e in site_evals)
    sst = sum_y2 - sum_y * sum_y / n
    std = math.sqrt(max(sst, 0.0) / n)
    r2_defined = std > MIN_TARGET_STD
    r2 = (1.0 - sse / sst) if (r2_defined and sst > 0) else None
    sites = []
    for e in site_evals:
        sites.append({"site_id": e["site_id"], "n": e["n"],
                      "rmse": (math.sqrt(e["sse"] / e["n"]) if e["n"] > 0 else None)})
    return {"n": n, "sse": sse, "rmse": math.sqrt(sse / n), "r2": r2, "r2_defined": r2_defined,
            "sum_y": sum_y, "sum_y2": sum_y2, "sites": sites}


def scenario_report(model, scen_stats, train_min, train_max, delta):
    """负荷情景：近SCENARIO_DAYS天均值基准，产量与运行小时同步缩放delta%。"""
    if scen_stats["n"] <= 0:
        raise DegenerateRun("情景窗在冻结边界内无样本")
    n = scen_stats["n"]
    baseline = [scen_stats["sum_x"][j] / n for j in range(4)]
    scenario = list(baseline)
    scenario[0] *= 1.0 + delta / 100.0
    scenario[1] *= 1.0 + delta / 100.0
    base_kwh = predict(model, baseline)
    scenario_kwh = predict(model, scenario)
    extrapolated = False
    if train_min and train_max:
        for name in model["features"]:
            j = FEATURES.index(name)
            if scenario[j] < train_min[j] or scenario[j] > train_max[j]:
                extrapolated = True
                break
    return {"baseline": baseline, "scenario": scenario, "base_kwh": base_kwh,
            "scenario_kwh": scenario_kwh, "extrapolated": extrapolated}


# ---------------------------------------------------------------- 单机对照实现
def _index_folds(n):
    """与 sklearn TimeSeriesSplit(n_splits=3) 默认切分一致的序号边界。"""
    test_size = n // (N_SPLITS + 1)
    starts = list(range(n - N_SPLITS * test_size, n, test_size))
    return [(s, s + test_size) for s in starts]


def merged_analyze(records, params):
    """单机版 analyze 的纯标准库对照实现（按记录序号切分，供对拍与本地复核）。

    与单机原版的唯一口径差异：多厂合并后同一日历日有多条记录（不同厂），
    故跳过同厂重复日期检查；情景基准取"最近14天"（单厂逐日数据下与原版
    "最后14条记录"完全一致），以保留"近14天平均负荷"的业务含义。
    """
    p = parse_params(params)
    parsed = parse_records(records, require_unique_days=False)
    if len(parsed) < MIN_DAYS:
        raise ValueError(f"至少需要{MIN_DAYS}天能耗记录，以保留时间验证窗")
    targets = [row[2] for row in parsed]
    mean_y = sum(targets) / len(targets)
    if math.sqrt(sum((v - mean_y) ** 2 for v in targets) / len(targets)) < MIN_TARGET_STD:
        raise ValueError("耗电量没有变化，无法辨识能耗驱动")
    n = len(parsed)
    cut = int(n * TRAIN_FRACTION)
    folds = _index_folds(cut)
    best = None
    candidates = []
    for count in range(2, p["max_features"] + 1):
        for feature_idx in combinations(range(4), count):
            for alpha in ALPHAS:
                errors = []
                for begin, end in folds:
                    train_st = stats_from_rows(parsed[:begin])
                    valid_st = stats_from_rows(parsed[begin:end])
                    model = fit_ridge(train_st, feature_idx, alpha)
                    errors.append(math.sqrt(sse_stats(valid_st, model) / valid_st["n"]))
                cv_rmse = sum(errors) / len(errors)
                candidate = {"features": [FEATURES[j] for j in feature_idx], "alpha": alpha, "cv_rmse": cv_rmse}
                candidates.append(candidate)
                if best is None or cv_rmse < best["cv_rmse"]:
                    best = candidate
    final_idx = tuple(FEATURES.index(name) for name in best["features"])
    model = fit_ridge(stats_from_rows(parsed[:cut]), final_idx, best["alpha"])
    holdout = parsed[cut:]
    sse = sum((row[2] - predict(model, row[1])) ** 2 for row in holdout)
    hold_targets = [row[2] for row in holdout]
    hmean = sum(hold_targets) / len(hold_targets)
    sst = sum((v - hmean) ** 2 for v in hold_targets)
    r2_defined = math.sqrt(sst / len(hold_targets)) > MIN_TARGET_STD
    r2 = (1.0 - sse / sst) if (r2_defined and sst > 0) else None
    last_day = parsed[-1][0]
    scen_start = last_day - timedelta(days=SCENARIO_DAYS - 1)
    scen_stats = stats_from_rows([row for row in parsed if row[0] >= scen_start])
    train_min = [min(row[1][j] for row in parsed[:cut]) for j in range(4)]
    train_max = [max(row[1][j] for row in parsed[:cut]) for j in range(4)]
    scenario = scenario_report(model, scen_stats, train_min, train_max, p["scenario_pct"])
    return {"selection": best, "candidates": candidates, "model": model,
            "holdout": {"n": len(holdout), "rmse": math.sqrt(sse / len(holdout)),
                        "r2": r2, "r2_defined": r2_defined},
            "scenario": scenario, "train_n": cut, "records": n}
