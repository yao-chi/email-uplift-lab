"""决策层（单位经济、盈亏平衡、敏感性）的单元测试。"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uplift_lab import policy as P  # noqa: E402


class TestUnitEconomics(unittest.TestCase):
    def test_breakeven_formula(self):
        tau = P.breakeven_uplift(cost_per_email=0.10, aov=116.36, margin=0.40)
        self.assertAlmostEqual(tau, 0.10 / (116.36 * 0.40), places=12)
        self.assertAlmostEqual(tau, 0.002148, delta=1e-6)

    def test_profit_zero_at_breakeven(self):
        c, aov, m = 0.10, 116.36, 0.40
        tau = P.breakeven_uplift(c, aov, m)
        self.assertAlmostEqual(P.profit_per_email(tau, c, aov, m), 0.0, places=10)

    def test_profit_increases_with_uplift(self):
        low = P.profit_per_email(0.005, 0.10, 116.36, 0.40)
        high = P.profit_per_email(0.010, 0.10, 116.36, 0.40)
        self.assertGreater(high, low)

    def test_profit_linear_in_margin(self):
        a = P.profit_per_email(0.01, 0.10, 100.0, 0.40)
        b = P.profit_per_email(0.01, 0.10, 100.0, 0.20)
        # 毛利部分翻倍，成本不变
        self.assertAlmostEqual(a - b, 0.01 * 100.0 * 0.20, places=10)


class TestSensitivity(unittest.TestCase):
    def test_grid_shape_and_monotonicity(self):
        grid = P.sensitivity_grid(
            uplift=0.00685, costs=[0.02, 0.10, 0.50], margins=[0.2, 0.4, 0.6], aov=116.36
        )
        self.assertEqual(len(grid), 9)
        for _, row in grid.groupby("cost_per_email"):
            vals = row.sort_values("margin")["profit_per_1000"].to_numpy()
            self.assertTrue((np.diff(vals) > 0).all())
        for _, row in grid.groupby("margin"):
            vals = row.sort_values("cost_per_email")["profit_per_1000"].to_numpy()
            self.assertTrue((np.diff(vals) < 0).all())


class TestStrategies(unittest.TestCase):
    def _data(self, n=40_000, seed=3):
        rng = np.random.default_rng(seed)
        t = rng.integers(0, 2, n)
        high = rng.random(n) < 0.3
        effect = np.where(high, 0.20, 0.03)
        y = (rng.random(n) < 0.05 + effect * t).astype(float)
        score = np.where(high, 1.0, 0.0) + rng.normal(0, 0.01, n)
        return y, t, score

    def test_strategy_table_columns_and_coverage(self):
        y, t, score = self._data()
        table = P.strategy_table(
            y, t, {"all": np.ones_like(t), "top10": score >= np.quantile(score, 0.9)}
        )
        self.assertIn("coverage", table.columns)
        self.assertIn("profit_per_1000_sent", table.columns)
        row = table.set_index("strategy")
        self.assertAlmostEqual(float(row.loc["all", "coverage"]), 1.0, places=6)
        self.assertAlmostEqual(float(row.loc["top10", "coverage"]), 0.1, delta=0.01)

    def test_targeting_beats_blanket_when_heterogeneity_strong(self):
        y, t, score = self._data(seed=4)
        table = P.strategy_table(
            y, t, {"all": np.ones_like(t), "top30": score >= np.quantile(score, 0.7)}
        ).set_index("strategy")
        self.assertGreater(
            float(table.loc["top30", "profit_per_1000_sent"]),
            float(table.loc["all", "profit_per_1000_sent"]),
        )

    def test_treat_all_baseline_matches_manual(self):
        y, t, _ = self._data()
        res = P.treat_all_baseline(y, t, cost_per_email=0.10, aov=116.36, margin=0.40)
        n_t, n_c = t.sum(), (1 - t).sum()
        uplift = y[t == 1].sum() / n_t - y[t == 0].sum() / n_c
        expect_profit_per_1000 = (uplift * 116.36 * 0.40 - 0.10) * 1000
        self.assertAlmostEqual(res["profit_per_1000"], expect_profit_per_1000, delta=1e-6)


if __name__ == "__main__":
    unittest.main()