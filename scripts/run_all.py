"""一键跑完整条链路：体检 -> 因果估计 -> 异质性 -> 增量建模 -> 决策 -> 功效。

用法：
    python scripts/run_all.py

产出：
    outputs/tables/*.csv    每个结论背后的数字
    outputs/figures/*.png   报告用图
    reports/results.json    汇总指标（README 里的每个数字都来自这里）
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uplift_lab import data as D  # noqa: E402
from uplift_lab import experiment as E  # noqa: E402
from uplift_lab import metrics as M  # noqa: E402
from uplift_lab import plots as P  # noqa: E402
from uplift_lab import policy as POL  # noqa: E402
from uplift_lab import sql as SQL  # noqa: E402
from uplift_lab.config import (  # noqa: E402
    CATEGORICAL_COVARIATES, N_BOOTSTRAP, NUMERIC_COVARIATES, RANDOM_STATE,
    TAB_DIR, TEST_SIZE, BusinessAssumptions,
)
from uplift_lab.uplift import ResponseModel, UpliftModel, build_design  # noqa: E402

ASSUMPTIONS = BusinessAssumptions()
MENS, WOMENS, CONTROL = "Mens E-Mail", "Womens E-Mail", "No E-Mail"
OUTCOMES = ["visit", "conversion", "spend"]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def main() -> dict:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # ---------------------------------------------------------------- 1. 体检
    log("加载原始数据并体检 ...")
    raw = D.load_raw()
    sha = D.file_sha256()
    audit = D.audit(raw, sha=sha)
    df = D.add_features(raw)
    results["audit"] = audit.to_dict()
    log(f"  rows={audit.n_rows:,} dup={audit.duplicate_rows} missing={audit.missing_cells}")

    # ---------------------------------------------------------------- 2. SQL 口径层
    log("构建 DuckDB 口径层 ...")
    con = SQL.build_tables()
    arm_metrics = SQL.fetch(con, "SELECT * FROM arm_metrics")
    segment_metrics = SQL.fetch(con, "SELECT * FROM segment_metrics")
    arm_metrics.to_csv(TAB_DIR / "arm_metrics.csv", index=False, encoding="utf-8-sig")
    segment_metrics.to_csv(TAB_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")
    results["arm_metrics"] = arm_metrics.round(6).to_dict(orient="records")

    # ---------------------------------------------------------------- 3. 实验体检
    log("SRM 检验 + 协变量平衡性 ...")
    srm = E.srm_test({str(k): int(v) for k, v in raw["segment"].value_counts().items()})
    balanced = E.standardized_mean_diff(
        df[df["arm"].isin([MENS, CONTROL])], NUMERIC_COVARIATES, CATEGORICAL_COVARIATES
    )
    balanced.to_csv(TAB_DIR / "covariate_balance.csv", index=False, encoding="utf-8-sig")
    results["srm"] = srm
    results["max_abs_smd"] = round(float(balanced["abs_smd"].max()), 4)
    log(f"  SRM p={srm['p_value']} pass={srm['pass']}; max|SMD|={results['max_abs_smd']}")

    # ---------------------------------------------------------------- 4. ATE + Holm
    log("三臂对比的 ATE（协变量调整 + Holm 校正）...")
    ate_rows, pvals = [], {}
    for outcome in OUTCOMES:
        for arm in [MENS, WOMENS]:
            sub = df[df["arm"].isin([arm, CONTROL])].copy()
            sub["t"] = (sub["arm"] == arm).astype(int)
            est = E.ols_ate(sub, outcome, treat_col="t")
            label = f"{arm}|{outcome}"
            pvals[label] = est.p_value
            ate_rows.append({
                "arm": arm, "outcome": outcome,
                "n_t": est.n_treatment, "n_c": est.n_control,
                "estimate": est.estimate, "se": est.se,
                "ci_low": est.ci_low, "ci_high": est.ci_high, "p_value": est.p_value,
            })
    ate = pd.DataFrame(ate_rows)
    holm = E.holm_adjust(pvals)
    ate["p_holm"] = [holm[f"{r.arm}|{r.outcome}"] for r in ate.itertuples()]
    ate["significant_holm"] = ate["p_holm"] < 0.05
    ate.to_csv(TAB_DIR / "ate.csv", index=False, encoding="utf-8-sig")
    results["ate"] = ate.round(8).to_dict(orient="records")

    # ---------------------------------------------------------------- 5. CUPED
    log("CUPED 方差缩减 ...")
    cuped_rows, forest_rows = [], []
    for outcome, label in [("visit", "Visit"), ("conversion", "Conversion"), ("spend", "Spend")]:
        sub = df[df["arm"].isin([MENS, CONTROL])].copy()
        sub["t"] = (sub["arm"] == MENS).astype(int)
        res = E.cuped(sub, outcome, covariate="history", treat_col="t")
        cuped_rows.append({"outcome": outcome, **res.to_dict()})
        forest_rows.append({
            "label": label, "estimate": res.raw.estimate,
            "ci_low": res.raw.ci_low, "ci_high": res.raw.ci_high,
            "adj_estimate": res.adjusted.estimate,
            "adj_ci_low": res.adjusted.ci_low, "adj_ci_high": res.adjusted.ci_high,
            "var_reduction_pct": res.variance_reduction * 100,
        })
    cuped_df = pd.DataFrame(forest_rows)
    cuped_df.to_csv(TAB_DIR / "cuped.csv", index=False, encoding="utf-8-sig")
    results["cuped"] = cuped_rows

    # ---------------------------------------------------------------- 6. 异质性
    log("分群增量 + 交互检验 ...")
    mc = df[df["arm"].isin([MENS, CONTROL])].copy()
    mc["t"] = (mc["arm"] == MENS).astype(int)
    het_tables = {}
    for outcome in ["visit", "conversion"]:
        for by in ["buyer_type", "recency_band", "history_band", "channel", "zip_code", "newbie"]:
            seg = POL.segment_uplift(mc, by=by, y_col=outcome, t_col="t")
            het_tables[(outcome, by)] = seg
            seg.to_csv(TAB_DIR / f"heterogeneity_{outcome}_{by}.csv", index=False, encoding="utf-8-sig")

    interactions = []
    for outcome in ["visit", "conversion", "spend"]:
        interactions.append(E.interaction_test(mc, outcome, "buyer_type", "both", treat_col="t"))
    inter = pd.DataFrame(interactions)
    inter.to_csv(TAB_DIR / "interaction_tests.csv", index=False, encoding="utf-8-sig")
    results["interaction_both"] = inter.round(8).to_dict(orient="records")
    log(f"  buyer_type=both 与其余人的增量差（visit）: {inter.loc[0, 'diff']:+.4f} "
        f"p={inter.loc[0, 'p_value']:.2e}")

    # Womens 邮件在"只买男装"的客群上几乎无效：说明选品/创意匹配是增量的来源
    wm = df[df["arm"].isin([WOMENS, CONTROL])].copy()
    wm["t"] = (wm["arm"] == WOMENS).astype(int)
    womens_on_mens_only = E.interaction_test(wm, "visit", "buyer_type", "mens_only", treat_col="t")
    results["womens_email_on_mens_only"] = womens_on_mens_only

    # ---------------------------------------------------------------- 7. 增量建模
    log("训练增量模型（T-learner / S-learner / 响应模型）...")
    train, test = D.train_test_split_stratified(mc, t_col="t", y_col="conversion",
                                                test_size=TEST_SIZE, seed=RANDOM_STATE)
    Xtr = build_design(train, NUMERIC_COVARIATES, CATEGORICAL_COVARIATES)
    Xte = build_design(test, NUMERIC_COVARIATES, CATEGORICAL_COVARIATES).reindex(
        columns=Xtr.columns, fill_value=0)
    ttr, tte = train["t"].to_numpy(), test["t"].to_numpy()

    scores: dict[str, np.ndarray] = {}
    for kind in ["t", "s"]:
        model = UpliftModel(kind=kind, seed=RANDOM_STATE).fit(Xtr, ttr, train["conversion"].to_numpy())
        scores[f"{kind.upper()}-learner"] = model.predict_uplift(Xte)
    resp = ResponseModel(seed=RANDOM_STATE).fit(Xtr, train["conversion"].to_numpy())
    scores["Response model"] = resp.predict_score(Xte)
    scores["Random"] = np.random.default_rng(RANDOM_STATE).random(len(test))

    yte = test["conversion"].to_numpy(dtype=float)
    model_rows, qini_payload, topk_payload = [], {}, {}
    for name, score in scores.items():
        qini_payload[name] = M.qini_curve(yte, tte, score)
        topk = M.topk_table(yte, tte, score)
        topk_payload[name] = topk
        row = {
            "model": name,
            "auuc": float(M.auuc(yte, tte, score)),
            "qini_vs_random_pct": None,
            "top10_uplift": float(topk.loc[topk["frac"] == 0.1, "uplift"].iloc[0]),
            "top10_ci_low": float(topk.loc[topk["frac"] == 0.1, "ci_low"].iloc[0]),
            "top10_ci_high": float(topk.loc[topk["frac"] == 0.1, "ci_high"].iloc[0]),
        }
        if name != "Random":
            row["qini_vs_random_pct"] = float(M.qini_coefficient(yte, tte, score)) * 100
            ci = M.bootstrap_auuc(yte, tte, score, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE)
            row["auuc_ci_low"], row["auuc_ci_high"] = ci["ci_low"], ci["ci_high"]
        model_rows.append(row)
    uplift_models = pd.DataFrame(model_rows)
    uplift_models.to_csv(TAB_DIR / "uplift_models.csv", index=False, encoding="utf-8-sig")
    results["uplift_models"] = uplift_models.round(8).to_dict(orient="records")
    results["random_auuc"] = round(float(M.random_auuc(yte, tte)), 8)

    # 分群参照：用训练集里 buyer_type 的真实组增量给留出集排序。
    # 这是"一条人工规则能拿到的最好结果"，用来判断个体模型有没有真的超过分群。
    grp_uplift = {
        k: float(
            sub.loc[sub["is_treated"] == 1, "conversion"].mean()
            - sub.loc[sub["is_treated"] == 0, "conversion"].mean()
        )
        for k, sub in train.groupby("buyer_type")
    }
    score_grp = test["buyer_type"].map(grp_uplift).to_numpy(dtype=float)
    topk_grp = M.topk_table(yte, tte, score_grp)
    results["group_oracle"] = {
        "group_uplift": {k: round(v, 6) for k, v in grp_uplift.items()},
        "auuc": round(float(M.auuc(yte, tte, score_grp)), 8),
        "qini_vs_random_pct": round(float(M.qini_coefficient(yte, tte, score_grp)) * 100, 2),
        "top10_uplift": round(float(topk_grp.loc[topk_grp["frac"] == 0.1, "uplift"].iloc[0]), 6),
    }
    log(
        f"  分群参照 AUUC={results['group_oracle']['auuc']:.6f} "
        f"({results['group_oracle']['qini_vs_random_pct']:+.1f}%)"
    )
    results["overall_uplift_holdout"] = round(float(yte[tte == 1].mean() - yte[tte == 0].mean()), 6)
    results["holdout_n"] = int(len(test))
    pd.concat([t.assign(model=n) for n, t in topk_payload.items()], ignore_index=True).to_csv(
        TAB_DIR / "topk_uplift.csv", index=False, encoding="utf-8-sig")

    gate = M.gate_table(yte, tte, scores["T-learner"])
    gate.to_csv(TAB_DIR / "gate.csv", index=False, encoding="utf-8-sig")
    results["gate"] = gate.round(8).to_dict(orient="records")
    log(f"  留出集实测增量={results['overall_uplift_holdout']:.4%} "
        f"随机基线 AUUC={results['random_auuc']:.6f}")

    # ---------------------------------------------------------------- 8. 决策
    log("策略对比 + 预算-利润曲线 ...")
    aov = float(df.loc[df["conversion"] == 1, "spend"].mean())
    if not np.isfinite(aov) or aov <= 0:
        aov = ASSUMPTIONS.fallback_aov
    kw = dict(cost_per_email=ASSUMPTIONS.cost_per_email, aov=aov, margin=ASSUMPTIONS.gross_margin)

    order_t = np.argsort(-scores["T-learner"], kind="mergesort")
    order_r = np.argsort(-scores["Response model"], kind="mergesort")
    both_mask = (test["buyer_type"] == "both").to_numpy()
    strategies = {
        "send_all": np.ones(len(test), dtype=bool),
        "rule:buyer_type=both": both_mask,
        "model_top10%": np.isin(np.arange(len(test)), order_t[: int(len(test) * 0.10)]),
        "model_top30%": np.isin(np.arange(len(test)), order_t[: int(len(test) * 0.30)]),
        "response_top10%": np.isin(np.arange(len(test)), order_r[: int(len(test) * 0.10)]),
    }
    strat = POL.strategy_table(yte, tte, strategies, **kw)
    strat.to_csv(TAB_DIR / "strategy_compare.csv", index=False, encoding="utf-8-sig")
    boot = POL.bootstrap_strategy(yte, tte, strategies, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE, **kw)
    boot.to_csv(TAB_DIR / "strategy_bootstrap.csv", index=False, encoding="utf-8-sig")
    results["strategies"] = strat.round(6).to_dict(orient="records")
    results["strategy_bootstrap"] = boot.round(6).to_dict(orient="records")

    policy = POL.policy_curve(yte, tte, scores["T-learner"], **kw)
    policy_resp = POL.policy_curve(yte, tte, scores["Response model"], **kw)
    policy.to_csv(TAB_DIR / "policy_curve.csv", index=False, encoding="utf-8-sig")
    policy_resp.to_csv(TAB_DIR / "policy_curve_response.csv", index=False, encoding="utf-8-sig")
    baseline = POL.treat_all_baseline(yte, tte, **kw)
    radius = POL.breakeven_uplift(ASSUMPTIONS.cost_per_email, aov, ASSUMPTIONS.gross_margin)
    results["decision"] = {
        "aov": round(aov, 2),
        "breakeven_uplift": round(float(radius), 8),
        "treat_all": {k: round(float(v), 6) for k, v in baseline.items()},
        "assumptions": {
            "cost_per_email": ASSUMPTIONS.cost_per_email,
            "gross_margin": ASSUMPTIONS.gross_margin,
        },
    }
    log(f"  AOV={aov:.2f} 盈亏平衡增量={radius:.4%}")

    sens = POL.sensitivity_grid(baseline["obs_uplift"], [0.02, 0.05, 0.10, 0.20, 0.50],
                                [0.20, 0.30, 0.40, 0.50, 0.60], aov)
    sens.to_csv(TAB_DIR / "sensitivity.csv", index=False, encoding="utf-8-sig")
    results["sensitivity"] = sens.round(2).to_dict(orient="records")

    # ---------------------------------------------------------------- 9. 功效
    log("功效分析：这个样本能分辨多大的增量差异 ...")
    p_t = float(yte[tte == 1].mean())
    p_c = float(yte[tte == 0].mean())
    n_decile = int(len(test) * 0.10 / 2)  # 十分位内两组各多少人
    power_rows = []
    for d in [0.002, 0.005, 0.010, 0.020]:
        power_rows.append({
            "target_diff": d,
            "n_per_cell_needed": E.n_for_mde(d, p_t, p_c),
            "n_per_cell_available": n_decile,
        })
    power = pd.DataFrame(power_rows)
    power["mde_at_current_n"] = E.mde_proportion_diff(p_t, p_c, n_decile)
    power.to_csv(TAB_DIR / "power.csv", index=False, encoding="utf-8-sig")
    results["power"] = power.round(8).to_dict(orient="records")

    # ---------------------------------------------------------------- 10. 出图
    log("出图 ...")
    P.arm_rates(arm_metrics)
    P.segment_uplift_bars(het_tables[("conversion", "buyer_type")], "buyer_type",
                          "Incremental conversion by buyer type", "02_uplift_by_buyer_type.png")
    P.qini_curves({k: v["qini"].to_numpy() for k, v in qini_payload.items()},
                  next(iter(qini_payload.values()))["frac"].to_numpy())
    P.gate_bars(gate)
    P.profit_curve(policy, baseline["profit_per_1000"])
    P.ate_forest(cuped_df)
    P.strategy_bars(boot, "07_strategy_compare.png")
    P.sensitivity_heatmap(sens, "08_sensitivity.png")

    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "reports" / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    log(f"完成，结果写入 {ROOT / 'reports' / 'results.json'}")
    return results


if __name__ == "__main__":
    res = main()
    print("\n=== 关键数字 ===")
    print(f"SRM p: {res['srm']['p_value']} | max|SMD|: {res['max_abs_smd']}")
    for row in res["ate"]:
        print(f"  {row['arm']:<13} {row['outcome']:<10} ATE={row['estimate']:+.4f} "
              f"[{row['ci_low']:+.4f},{row['ci_high']:+.4f}]")
    for row in res["uplift_models"]:
        q = row["qini_vs_random_pct"]
        tag = " base" if (q is None or (isinstance(q, float) and math.isnan(q))) else f"{q:+5.1f}%"
        print(f"  {row['model']:<15} AUUC={row['auuc']:.6f}  ({tag} vs random)")
    go = res["group_oracle"]
    print(f"  {'分群参照':<13} AUUC={go['auuc']:.6f}  ({go['qini_vs_random_pct']:+5.1f}% vs random)")
    print(f"AOV={res['decision']['aov']} 盈亏平衡增量={res['decision']['breakeven_uplift']:.4%}")
    print(f"treat-all 利润/千封={res['decision']['treat_all']['profit_per_1000']:.1f}")
    for row in res["strategy_bootstrap"]:
        print(f"  {row['strategy']:<24} {row['profit_per_1000_sent']:8.1f} "
              f"[{row['ci_low']:7.1f},{row['ci_high']:7.1f}] 每千封")