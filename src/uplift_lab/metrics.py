"""Uplift 评估指标：Qini / AUUC / GATE / 策略价值。

关键区别（面试高频）：
  * 响应模型回答"谁会买"，uplift 回答"谁是因为被打扰才买的"。
  * 用 AUC 评估 uplift 模型是错的，因为标签里根本没有反事实。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """按 score 从高到低累积，计算累计增量转化数。

    qini(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)
    含义：把前 k 个人按当前排序全部触达，相比"不触达"，预期多出多少转化。
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, t = y[order], t[order]

    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1 - t))
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)

    with np.errstate(divide="ignore", invalid="ignore"):
        qini = y_t - np.where(n_c > 0, y_c * n_t / np.where(n_c > 0, n_c, 1), 0.0)
    frac = np.arange(1, len(y) + 1) / len(y)
    return pd.DataFrame({"frac": frac, "qini": qini, "n_t": n_t, "n_c": n_c})


def auuc(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Qini 曲线下面积 / 总人数。

    量纲：**每人**的平均累计增量转化率（0.001 = 0.1 个百分点），
    这样可以直接和"总体增量 / 2"的随机基线比较。
    """
    curve = qini_curve(y, t, score)
    n = len(curve)
    return float(np.trapezoid(curve["qini"].to_numpy(), curve["frac"].to_numpy()) / n)


def total_uplift_per_person(y: np.ndarray, t: np.ndarray) -> float:
    """把前 100% 全部触达时的增量，摊到每个人头上。"""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    n_t, n_c = t.sum(), (1 - t).sum()
    total_gain = y[t == 1].sum() - y[t == 0].sum() * (n_t / n_c)
    return float(total_gain / len(y))


def random_auuc(y: np.ndarray, t: np.ndarray) -> float:
    """随机排序的期望 AUUC。

    随机排序的 Qini 曲线是一条从 0 线性上升到"全部触达增量"的直线，
    因此面积 = 全部触达增量 / 2（再摊到每人）。
    """
    return float(total_uplift_per_person(y, t) / 2)


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """相对随机排序的提升倍数：(AUUC_model / AUUC_random) - 1。

    >0 才算"排序有用"；0 表示模型等于随机；负值说明排序是反的。
    """
    base = random_auuc(y, t)
    if base == 0:
        return float("nan")
    return float(auuc(y, t, score) / base - 1)


def bootstrap_auuc(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """对样本做有放回重采样，给 AUUC 一个置信区间。

    注意：这里置信区间反映的是"这群人上的估计不确定性"，不是模型选择带来的不确定性。
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        values[b] = auuc(y[idx], t[idx], score[idx])
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {
        "auuc": round(float(auuc(y, t, score)), 8),
        "ci_low": round(float(lo), 8),
        "ci_high": round(float(hi), 8),
        "se": round(float(values.std(ddof=1)), 8),
        "n_boot": n_boot,
    }


def gate_table(y: np.ndarray, t: np.ndarray, score: np.ndarray, q: int = 10) -> pd.DataFrame:
    """GATE：按预测增量分位分组，看每组的**实测**增量是否单调。

    这是检验 uplift 排序有没有意义的最直接方法（比 Qini 曲线更不容易被误读）。
    """
    df = pd.DataFrame({"y": y, "t": t, "score": score})
    # 分位组：score 高的组 = 第 1 组
    df["group"] = pd.qcut(df["score"].rank(method="first", ascending=False), q, labels=False) + 1
    rows = []
    for g, part in df.groupby("group", observed=True):
        yt = part.loc[part.t == 1, "y"]
        yc = part.loc[part.t == 0, "y"]
        est = yt.mean() - yc.mean()
        se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
        rows.append(
            {
                "group": int(g),
                "n": len(part),
                "pred_uplift_mean": part["score"].mean(),
                "obs_uplift": est,
                "ci_low": est - 1.96 * se,
                "ci_high": est + 1.96 * se,
                "n_t": len(yt),
                "n_c": len(yc),
            }
        )
    return pd.DataFrame(rows)


def uplift_at_top(y: np.ndarray, t: np.ndarray, score: np.ndarray, frac: float = 0.1) -> dict:
    """Top-frac 人群里的实测增量（与总体 ATE 对比，看排序是否有效）。"""
    k = max(1, int(len(y) * frac))
    order = np.argsort(-np.asarray(score), kind="mergesort")[:k]
    yk, tk = np.asarray(y)[order], np.asarray(t)[order]
    if tk.sum() == 0 or (1 - tk).sum() == 0:
        return {"frac": frac, "uplift": float("nan"), "n": int(k)}
    est = yk[tk == 1].mean() - yk[tk == 0].mean()
    se = np.sqrt(
        yk[tk == 1].var(ddof=1) / tk.sum() + yk[tk == 0].var(ddof=1) / (1 - tk).sum()
    )
    return {
        "frac": frac,
        "n": int(k),
        "uplift": float(est),
        "ci_low": float(est - 1.96 * se),
        "ci_high": float(est + 1.96 * se),
    }


def topk_table(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, fracs: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
) -> pd.DataFrame:
    """按预测增量排序后，取前 k% 人群的**实测**增量。

    这是最诚实的检查：如果 D1 的实测增量不比全体高，模型就没有排序能力。
    """
    rows = []
    for frac in fracs:
        info = uplift_at_top(y, t, score, frac)
        rows.append({"frac": frac, **info})
    return pd.DataFrame(rows)