"""决策层：把"增量"翻译成"钱"。

全量触达是很多公司的默认做法。这个模块回答两个问题：
1. 完全不筛选地群发，值不值？
2. 按增量分排序后只发前 k%，利润能提高多少？最优 k 在哪？
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def breakeven_uplift(cost_per_email: float, aov: float, margin: float) -> float:
    """盈亏平衡的增量转化率：低于这个数，发邮件就是在亏钱。"""
    return cost_per_email / (aov * margin)


def _uplift_with_ci(y: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    yt, yc = y[t == 1], y[t == 0]
    est = yt.mean() - yc.mean()
    se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    return float(est), float(est - 1.96 * se), float(est + 1.96 * se)


def policy_curve(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    fractions: np.ndarray | None = None,
    cost_per_email: float = 0.10,
    aov: float = 120.0,
    margin: float = 0.40,
) -> pd.DataFrame:
    """按 score 从高到低切 k%，计算"只发这 k%"的观测增量与利润。

    利润 = 增量转化数 × 客单价 × 毛利率 − 触达人数 × 单封成本
    这里的增量是**观测**到的（同一批人内部处理组 vs 对照组），不是模型预测值。
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, t = y[order], t[order]

    if fractions is None:
        fractions = np.arange(0.05, 1.0001, 0.05)

    rows = []
    n = len(y)
    for f in fractions:
        k = max(2, int(round(n * f)))
        yk, tk = y[:k], t[:k]
        if tk.sum() == 0 or (1 - tk).sum() == 0:
            continue
        uplift, lo, hi = _uplift_with_ci(yk, tk)
        inc_conv = uplift * k
        revenue = inc_conv * aov * margin
        cost = k * cost_per_email
        rows.append(
            {
                "fraction": round(float(f), 3),
                "n_sent": int(k),
                "obs_uplift": uplift,
                "uplift_ci_low": lo,
                "uplift_ci_high": hi,
                "incremental_conversions": inc_conv,
                "incremental_revenue": revenue,
                "cost": cost,
                "profit": revenue - cost,
                "profit_per_1000": (revenue - cost) / k * 1000,
            }
        )
    return pd.DataFrame(rows)


def treat_all_baseline(
    y: np.ndarray, t: np.ndarray, cost_per_email: float = 0.10, aov: float = 120.0, margin: float = 0.40
) -> dict:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    uplift, lo, hi = _uplift_with_ci(y, t)
    n = len(y)
    revenue = uplift * n * aov * margin
    cost = n * cost_per_email
    return {
        "n_sent": int(n),
        "obs_uplift": uplift,
        "uplift_ci_low": lo,
        "uplift_ci_high": hi,
        "incremental_conversions": uplift * n,
        "incremental_revenue": revenue,
        "cost": cost,
        "profit": revenue - cost,
        "profit_per_1000": (revenue - cost) / n * 1000,
    }


def segment_uplift(
    df: pd.DataFrame,
    by: str,
    y_col: str = "conversion",
    t_col: str = "is_treated",
    min_n: int = 500,
) -> pd.DataFrame:
    """分组看实测增量，用于找"负增量"人群（sleeping dogs）。"""
    rows = []
    for value, part in df.groupby(by, observed=True):
        if len(part) < min_n:
            continue
        y = part[y_col].to_numpy(dtype=float)
        t = part[t_col].to_numpy(dtype=int)
        if t.sum() == 0 or (1 - t).sum() == 0:
            continue
        est, lo, hi = _uplift_with_ci(y, t)
        rows.append(
            {
                by: str(value),
                "n": len(part),
                "n_treated": int(t.sum()),
                "uplift": est,
                "ci_low": lo,
                "ci_high": hi,
                "significant": bool(lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(rows).sort_values("uplift", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- 策略对比
def profit_per_email(uplift: float, cost_per_email: float, aov: float, margin: float) -> float:
    """发一封邮件带来的净利：增量转化 × 客单价 × 毛利率 − 成本。"""
    return uplift * aov * margin - cost_per_email


def strategy_table(
    y: np.ndarray,
    t: np.ndarray,
    strategies: dict[str, np.ndarray],
    cost_per_email: float = 0.10,
    aov: float = 120.0,
    margin: float = 0.40,
) -> pd.DataFrame:
    """对比多个投放策略（每个策略是一个布尔 mask：这些人发，其余不发）。

    关键口径：
      * profit_per_1000_sent = 每发 1000 封的净利（绝对值，不随名单大小变）
      * coverage            = 该策略覆盖名单的比例（名单大小的另一面）
      * total_profit_per_1000_population = 摊到"全体 1000 人"的净利 = 覆盖 × 每封利润
    最后一个才是"同样规模的名单，总利润差多少"，前两个回答"单封效率"。
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    n = len(y)
    rows = []
    for name, mask in strategies.items():
        mask = np.asarray(mask, dtype=bool)
        k = int(mask.sum())
        if k == 0 or t[mask].sum() == 0 or (1 - t[mask]).sum() == 0:
            continue
        uplift = float(y[mask & (t == 1)].mean() - y[mask & (t == 0)].mean())
        ppe = profit_per_email(uplift, cost_per_email, aov, margin)
        rows.append(
            {
                "strategy": name,
                "n_sent": k,
                "coverage": k / n,
                "obs_uplift": uplift,
                "profit_per_1000_sent": ppe * 1000,
                "total_profit_per_1000_population": ppe * 1000 * (k / n),
            }
        )
    return pd.DataFrame(rows).sort_values("profit_per_1000_sent", ascending=False, ignore_index=True)


def bootstrap_strategy(
    y: np.ndarray,
    t: np.ndarray,
    strategies: dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: int = 0,
    cost_per_email: float = 0.10,
    aov: float = 120.0,
    margin: float = 0.40,
) -> pd.DataFrame:
    """对策略利润做 bootstrap，给出 95% 区间。

    没有区间的策略对比是耍流氓：top 10% 的利润本来就该比全量高，
    关键看"高出多少"和"这个差距稳不稳"。
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    n = len(y)
    names = list(strategies)
    masks = [np.asarray(strategies[k], dtype=bool) for k in names]
    draws = {k: [] for k in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, tb = y[idx], t[idx]
        for name, mask in zip(names, masks):
            mb = mask[idx]
            if tb[mb].sum() == 0 or (1 - tb[mb]).sum() == 0:
                continue
            u = yb[mb & (tb == 1)].mean() - yb[mb & (tb == 0)].mean()
            draws[name].append(profit_per_email(u, cost_per_email, aov, margin) * 1000)
    rows = []
    for name in names:
        arr = np.asarray(draws[name])
        if len(arr) < 10:
            continue
        rows.append(
            {
                "strategy": name,
                "profit_per_1000_sent": float(np.mean(arr)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
                "n_boot_ok": int(len(arr)),
            }
        )
    return pd.DataFrame(rows).sort_values("profit_per_1000_sent", ascending=False, ignore_index=True)


def sensitivity_grid(
    uplift: float,
    costs: list[float],
    margins: list[float],
    aov: float,
) -> pd.DataFrame:
    """成本 × 毛利率的敏感性网格：在什么假设下"发"变成"不发"。"""
    rows = []
    for c in costs:
        for m in margins:
            rows.append({"cost_per_email": c, "margin": m,
                         "profit_per_1000": profit_per_email(uplift, c, aov, m) * 1000})
    return pd.DataFrame(rows)