"""演示厂站日样本生成：固定随机种子，口径与单机版 seed_records 一致。

能耗合成含产量、运行、温度和待机的真实驱动项与噪声；各厂基线负荷略有差异。
所有数据为模拟业务数据，不能作为实际生产决策证据。
"""
from __future__ import annotations

import math
import random
from datetime import date, timedelta

FACTORIES = (
    ("S1", "澄川一厂冲压P01", 1016),
    ("S2", "澄川二厂冲压P02", 2024),
    ("S3", "澄川三厂冲压P03", 3307),
)


def factory_records(seed, line, days=160, start=date(2026, 5, 1)):
    rng = random.Random(seed)
    base = 95 + (seed % 37)  # 各厂基线负荷差异
    rows = []
    for i in range(days):
        units = round(760 + 180 * math.sin(i / 11) + rng.gauss(0, 85), 1)
        run_hours = round(6.4 + units / 140 + rng.gauss(0, 0.65), 2)
        temp = round(24 + 7 * math.sin(i / 33) + rng.gauss(0, 1.4), 2)
        idle = round(rng.uniform(0.3, 3.6), 2)
        kwh = round(base + units * 0.46 + run_hours * 24 + temp * 4.1 + idle * 38 + rng.gauss(0, 13), 2)
        rows.append({"day": str(start + timedelta(days=i)), "line": line,
                     "units": units, "run_hours": run_hours, "temp_c": temp,
                     "idle_hours": idle, "kwh": kwh})
    return rows
