"""出图：统一风格、英文标注（跨平台不会掉字）。

每张图都在报告里对应一个结论，不做"好看但没信息"的图。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR

PALETTE = {
    "mens": "#1f77b4",
    "womens": "#ff7f0e",
    "control": "#7f7f7f",
    "model": "#2ca02c",
    "baseline": "#d62728",
    "neutral": "#4c4c4c",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.autolayout": True,
        }
    )


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def arm_rates(arm_metrics: pd.DataFrame, name: str = "01_arm_rates.png") -> Path:
    df = arm_metrics.set_index("arm").loc[["No E-Mail", "Womens E-Mail", "Mens E-Mail"]]
    colors = [PALETTE["control"], PALETTE["womens"], PALETTE["mens"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    specs = [
        ("visit_rate", "Visit rate", 100),
        ("conv_rate", "Conversion rate", 100),
        ("conv_given_visit", "Conversion | Visit", 100),
    ]
    for ax, (col, title, scale) in zip(axes, specs):
        vals = df[col].to_numpy() * scale
        bars = ax.bar(range(3), vals, color=colors, width=0.62)
        ax.set_xticks(range(3), ["No E-Mail", "Womens", "Mens"], fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("%" if scale == 100 else col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Three-arm experiment: raw outcome rates", fontsize=13, fontweight="bold")
    return _save(fig, name)


def segment_uplift_bars(seg: pd.DataFrame, by: str, title: str, name: str) -> Path:
    s = seg.sort_values("uplift")
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.42 * len(s) + 1.2)))
    y = np.arange(len(s))
    err = np.vstack([s["uplift"] - s["ci_low"], s["ci_high"] - s["uplift"]]) * 100
    colors = [PALETTE["baseline"] if v < 0 else PALETTE["mens"] for v in s["uplift"]]
    ax.barh(y, s["uplift"] * 100, xerr=err, color=colors, alpha=0.85, height=0.62, error_kw={"lw": 1})
    ax.set_yticks(y, s[by].astype(str), fontsize=9)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("Incremental conversion rate (percentage points)")
    ax.set_title(title)
    hi = (s["ci_high"] * 100).to_numpy()
    lo = (s["ci_low"] * 100).to_numpy()
    for yi, v, n, h, l in zip(y, s["uplift"] * 100, s["n"], hi, lo):
        # 标签放在置信区间之外，避免压住误差线
        if v >= 0:
            ax.text(h, yi, f"  {v:+.2f}pp (n={n:,})", va="center", fontsize=8, ha="left")
        else:
            ax.text(l, yi, f"{v:+.2f}pp (n={n:,})  ", va="center", fontsize=8, ha="right")
    lo_all, hi_all = min(float(lo.min()), 0.0), max(float(hi.max()), 0.0)
    span = max(hi_all - lo_all, 1e-6)
    ax.set_xlim(lo_all - 0.02 * span, hi_all + 0.28 * span)
    return _save(fig, name)


def qini_curves(curves: dict[str, np.ndarray], frac: np.ndarray, name: str = "03_qini_curves.png") -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, y in curves.items():
        is_random = "random" in label.lower()
        ax.plot(frac, y, "--" if is_random else "-", label=label,
                lw=1.2 if is_random else 2)
    ax.set_xlabel("Fraction of population targeted (sorted by predicted uplift)")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title("Qini curves on hold-out set")
    ax.legend(fontsize=8.5, frameon=False)
    return _save(fig, name)


def gate_bars(gate: pd.DataFrame, name: str = "04_gate_deciles.png") -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(gate))
    err = np.vstack([gate["obs_uplift"] - gate["ci_low"], gate["ci_high"] - gate["obs_uplift"]]) * 100
    ax.bar(x, gate["obs_uplift"] * 100, yerr=err, color=PALETTE["model"], alpha=0.85,
           error_kw={"lw": 1}, width=0.65)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x, [f"D{i+1}" for i in range(len(gate))])
    ax.set_xlabel("Decile of predicted uplift (D1 = highest)")
    ax.set_ylabel("Observed uplift (percentage points)")
    ax.set_title("GATE: does the ranking actually separate responders?")
    return _save(fig, name)


def profit_curve(policy: pd.DataFrame, baseline_per_1000: float, name: str = "05_profit_curve.png") -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(policy["fraction"] * 100, policy["profit_per_1000"], "o-", color=PALETTE["model"],
            lw=2, ms=4, label="Uplift targeting")
    ax.axhline(baseline_per_1000, color=PALETTE["baseline"], ls="--", lw=1.4,
               label=f"Send to everyone ({baseline_per_1000:,.1f}/1000)")
    ax.axhline(0, color="black", lw=0.9)
    best = policy.loc[policy["profit_per_1000"].idxmax()]
    ax.scatter([best["fraction"] * 100], [best["profit_per_1000"]], s=90, zorder=5,
               facecolors="none", edgecolors="black")
    ax.annotate(f"best {best['fraction']*100:.0f}%\n{best['profit_per_1000']:,.1f}/1000",
                (best["fraction"] * 100, best["profit_per_1000"]),
                textcoords="offset points", xytext=(8, 12), fontsize=8.5)
    ax.set_xlabel("Share of audience targeted (%)")
    ax.set_ylabel("Incremental profit per 1,000 sends")
    ax.set_title("Budget decision: where does marginal profit turn negative?")
    ax.legend(fontsize=8.5, frameon=False)
    return _save(fig, name)


def ate_forest(rows: pd.DataFrame, name: str = "06_ate_forest.png") -> Path:
    """ATE 与 CUPED 调整后的 ATE 对比：看区间收窄了多少。"""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(rows))
    ax.errorbar(rows["estimate"] * 100, y + 0.12,
                xerr=[(rows["estimate"] - rows["ci_low"]) * 100, (rows["ci_high"] - rows["estimate"]) * 100],
                fmt="o", color=PALETTE["baseline"], label="Raw difference", capsize=3, ms=5)
    ax.errorbar(rows["adj_estimate"] * 100, y - 0.12,
                xerr=[(rows["adj_estimate"] - rows["adj_ci_low"]) * 100,
                      (rows["adj_ci_high"] - rows["adj_estimate"]) * 100],
                fmt="s", color=PALETTE["model"], label="CUPED adjusted", capsize=3, ms=5)
    ax.set_yticks(y, rows["label"], fontsize=9)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("Treatment effect on Mens e-mail arm (percentage points)")
    ax.set_title("Point estimates and 95% CIs: raw vs CUPED")
    ax.legend(fontsize=8.5, frameon=False)
    return _save(fig, name)


def strategy_bars(boot: pd.DataFrame, name: str = "07_strategy_compare.png") -> Path:
    """各策略"每发一千封的净利"与 bootstrap 区间。"""
    s = boot.sort_values("profit_per_1000_sent")
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    y = np.arange(len(s))
    err = np.vstack([s["profit_per_1000_sent"] - s["ci_low"],
                     s["ci_high"] - s["profit_per_1000_sent"]])
    ax.barh(y, s["profit_per_1000_sent"], xerr=err, height=0.6,
            color=[PALETTE["model"] if "model" in n else PALETTE["mens"] for n in s["strategy"]],
            alpha=0.85, error_kw={"lw": 1})
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y, s["strategy"].str.replace("_", " "), fontsize=9)
    ax.set_xlabel("Incremental profit per 1,000 e-mails (currency units)")
    ax.set_title("Strategy comparison with bootstrap 95% intervals")
    for yi, v, lo, hi in zip(y, s["profit_per_1000_sent"], s["ci_low"], s["ci_high"]):
        ax.text(hi, yi, f"  {v:,.0f} [{lo:,.0f}, {hi:,.0f}]", va="center", fontsize=8)
    return _save(fig, name)


def sensitivity_heatmap(sens: pd.DataFrame, name: str = "08_sensitivity.png") -> Path:
    """成本 × 毛利率网格：哪些假设下每千封仍然赚钱。"""
    pivot = sens.pivot(index="margin", columns="cost_per_email", values="profit_per_1000")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]), [f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]), [f"{m:.0%}" for m in pivot.index])
    ax.set_xlabel("Cost per e-mail (assumption)")
    ax.set_ylabel("Gross margin (assumption)")
    ax.set_title("Profit per 1,000 sends under different assumptions")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy()[i, j]
            ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=8,
                    color="black" if v > 0 else "white")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="profit / 1,000 sends")
    return _save(fig, name)