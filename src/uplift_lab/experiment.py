"""实验体检与因果估计。

包含四件事：
1. SRM（样本比例失配）检验 —— 先确认分组没坏
2. 协变量平衡性（标准化均值差 SMD）—— 随机化是否真的成功了
3. 组间差异的因果估计（差分 + OLS 协变量调整，HC1 稳健标准误）
4. CUPED 方差缩减 —— 用实验前协变量把置信区间压窄
5. 多重比较校正（Holm）
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Estimate:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    n_treatment: int
    n_control: int

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


# ---------------------------------------------------------------- SRM / 平衡性
def srm_test(counts: dict[str, int], expected_share: dict[str, float] | None = None) -> dict:
    """卡方拟合优度检验：观测分组人数是否符合设计比例。"""
    arms = list(counts)
    observed = np.array([counts[a] for a in arms], dtype=float)
    if expected_share is None:
        expected = np.full(len(arms), observed.sum() / len(arms))
    else:
        expected = np.array([expected_share[a] for a in arms]) * observed.sum()
    chi2, p = stats.chisquare(observed, expected)
    return {
        "observed": {a: int(v) for a, v in zip(arms, observed)},
        "expected": {a: round(float(v), 1) for a, v in zip(arms, expected)},
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p), 4),
        "pass": bool(p > 0.01),
    }


def _dummies(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[cols].astype(str), drop_first=True, dtype=float)


def standardized_mean_diff(
    df: pd.DataFrame, numeric: list[str], categorical: list[str], treat_col: str = "is_treated"
) -> pd.DataFrame:
    """SMD = (均值差) / 合并标准差。经验阈值：|SMD| < 0.1 视为平衡。"""
    rows = []
    for col in numeric:
        t = df.loc[df[treat_col] == 1, col].to_numpy(dtype=float)
        c = df.loc[df[treat_col] == 0, col].to_numpy(dtype=float)
        pooled = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
        smd = (t.mean() - c.mean()) / pooled if pooled else 0.0
        rows.append({"covariate": col, "mean_treatment": t.mean(), "mean_control": c.mean(), "smd": smd})

    dummies = _dummies(df, categorical)
    dummies[treat_col] = df[treat_col].to_numpy()
    for col in [c for c in dummies.columns if c != treat_col]:
        t = dummies.loc[dummies[treat_col] == 1, col].to_numpy(dtype=float)
        c = dummies.loc[dummies[treat_col] == 0, col].to_numpy(dtype=float)
        pooled = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
        smd = (t.mean() - c.mean()) / pooled if pooled else 0.0
        rows.append({"covariate": col, "mean_treatment": t.mean(), "mean_control": c.mean(), "smd": smd})

    out = pd.DataFrame(rows)
    out["abs_smd"] = out["smd"].abs()
    return out.sort_values("abs_smd", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- 效应估计
def diff_in_means(y: np.ndarray, treat: np.ndarray, alpha: float = 0.05) -> Estimate:
    """两独立样本比例/均值差（Welch 标准误）。"""
    y = np.asarray(y, dtype=float)
    treat = np.asarray(treat, dtype=int)
    yt, yc = y[treat == 1], y[treat == 0]
    est = yt.mean() - yc.mean()
    se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * (1 - stats.norm.cdf(abs(est / se))) if se > 0 else 1.0
    return Estimate(float(est), float(se), float(est - z * se), float(est + z * se), float(p), len(yt), len(yc))


def _design_matrix(df: pd.DataFrame, treat_col: str, numeric: list[str], categorical: list[str]) -> np.ndarray:
    X = [np.ones(len(df)), df[treat_col].to_numpy(dtype=float)]
    for col in numeric:
        v = df[col].to_numpy(dtype=float)
        sd = v.std(ddof=0) or 1.0
        X.append((v - v.mean()) / sd)
    d = _dummies(df, categorical)
    for col in d.columns:
        X.append(d[col].to_numpy(dtype=float))
    return np.column_stack(X)


def ols_ate(
    df: pd.DataFrame,
    outcome: str,
    treat_col: str = "is_treated",
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
    alpha: float = 0.05,
) -> Estimate:
    """协变量调整的 ATE（Lin 估计量思想），HC1 稳健标准误。

    为什么不用裸差分：即使随机化成功，协变量调整也能降低方差（CUPED 的表亲），
    并且在样本比例轻微失衡时更稳健。
    """
    from .config import NUMERIC_COVARIATES, CATEGORICAL_COVARIATES

    numeric = NUMERIC_COVARIATES if numeric is None else numeric
    categorical = CATEGORICAL_COVARIATES if categorical is None else categorical
    sub = df.dropna(subset=[outcome]).copy()
    X = _design_matrix(sub, treat_col, numeric, categorical)
    y = sub[outcome].to_numpy(dtype=float)

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    meat = (X * resid[:, None] ** 2).T @ X
    cov = XtX_inv @ meat @ XtX_inv * (n / (n - k))
    se = float(np.sqrt(cov[1, 1]))
    est = float(beta[1])
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * (1 - stats.norm.cdf(abs(est / se))) if se > 0 else 1.0
    return Estimate(
        est, se, est - z * se, est + z * se, float(p),
        int(sub[treat_col].sum()), int((1 - sub[treat_col]).sum()),
    )


# ---------------------------------------------------------------- CUPED
@dataclass
class CupedResult:
    theta: float
    var_before: float
    var_after: float
    variance_reduction: float
    ci_width_before: float
    ci_width_after: float
    ci_shrink: float
    raw: Estimate
    adjusted: Estimate

    def to_dict(self) -> dict:
        return {
            "theta": round(self.theta, 6),
            "variance_before": round(self.var_before, 6),
            "variance_after": round(self.var_after, 6),
            "variance_reduction_pct": round(self.variance_reduction * 100, 2),
            "ci_width_before": round(self.ci_width_before, 6),
            "ci_width_after": round(self.ci_width_after, 6),
            "ci_shrink_pct": round(self.ci_shrink * 100, 2),
            "raw": self.raw.to_dict(),
            "adjusted": self.adjusted.to_dict(),
        }


def cuped(
    df: pd.DataFrame, outcome: str, covariate: str = "history", treat_col: str = "is_treated"
) -> CupedResult:
    """CUPED：Y_adj = Y - theta * (X - mean(X))，theta = Cov(Y,X)/Var(X)。

    注意 X 必须是**实验前**变量（这里是过去 12 个月消费额），否则会把处理效应
    一起减掉，得到一个"看起来很精确但错得离谱"的估计。
    """
    sub = df.dropna(subset=[outcome, covariate]).copy()
    y = sub[outcome].to_numpy(dtype=float)
    x = sub[covariate].to_numpy(dtype=float)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    y_adj = y - theta * (x - x.mean())

    raw = diff_in_means(y, sub[treat_col].to_numpy())
    adj = diff_in_means(y_adj, sub[treat_col].to_numpy())
    width_before = raw.ci_high - raw.ci_low
    width_after = adj.ci_high - adj.ci_low
    return CupedResult(
        theta=theta,
        var_before=float(np.var(y, ddof=1)),
        var_after=float(np.var(y_adj, ddof=1)),
        variance_reduction=float(1 - np.var(y_adj, ddof=1) / np.var(y, ddof=1)),
        ci_width_before=float(width_before),
        ci_width_after=float(width_after),
        ci_shrink=float(1 - width_after / width_before),
        raw=raw,
        adjusted=adj,
    )


# ---------------------------------------------------------------- 多重比较
def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni 逐步校正，控制族错误率。"""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adjusted[name] = running
    return adjusted


# ---------------------------------------------------------------- 异质性检验
def interaction_test(
    df: pd.DataFrame,
    outcome: str,
    subgroup_col: str,
    focus: str,
    treat_col: str = "is_treated",
    alpha: float = 0.05,
) -> dict:
    """2×2 差分中的差分（difference-in-differences）：子群增量 - 其余人增量。

    只在"子群增量不同"这件事上做推断，而不是比较子群的绝对转化率——
    后者会把基准率差异误当成增量差异（这是简历上写分群结论最常见的错误）。
    """
    y = df[outcome].to_numpy(dtype=float)
    t = df[treat_col].to_numpy(dtype=int)
    g = (df[subgroup_col].astype(str) == str(focus)).to_numpy()

    def cell(mask):
        v = y[mask]
        if len(v) == 0:
            return np.nan, np.nan, 0
        return float(v.mean()), float(v.var(ddof=1) / len(v)), int(len(v))

    mt, vt, nt = cell(g & (t == 1))
    mc, vc, nc = cell(g & (t == 0))
    ot, vo, no_ = cell(~g & (t == 1))
    oc, vc2, nc2 = cell(~g & (t == 0))

    est = (mt - mc) - (ot - oc)
    se = float(np.sqrt(vt + vc + vo + vc2))
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = float(2 * (1 - stats.norm.cdf(abs(est / se)))) if se > 0 else 1.0
    return {
        "outcome": outcome,
        "subgroup": f"{subgroup_col}={focus}",
        "uplift_focus": round(float(mt - mc), 6),
        "uplift_rest": round(float(ot - oc), 6),
        "diff": round(float(est), 6),
        "se": round(se, 6),
        "ci_low": round(float(est - z * se), 6),
        "ci_high": round(float(est + z * se), 6),
        "p_value": round(p, 8),
        "n_focus": nt + nc,
        "n_rest": no_ + nc2,
    }


def mde_proportion_diff(p_treat: float, p_control: float, n_per_cell: int,
                        alpha: float = 0.05, power: float = 0.8) -> float:
    """4 个格子各 n 人时，能被 80% 概率检出的最小增量差异（MDE）。

    用途：回答"模型排出来的 top 10% 增量比全体高 2 个点，这是真的还是噪声"。
    """
    var = 2 * (p_treat * (1 - p_treat) + p_control * (1 - p_control)) / n_per_cell
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(var))


def n_for_mde(delta: float, p_treat: float, p_control: float,
              alpha: float = 0.05, power: float = 0.8) -> int:
    """要检出 delta 这么大的增量差异，每个格子需要多少人。"""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    var_unit = 2 * (p_treat * (1 - p_treat) + p_control * (1 - p_control))
    return int(np.ceil(var_unit * (z / delta) ** 2))