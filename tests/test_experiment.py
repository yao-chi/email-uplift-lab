"""统计与因果推断函数的单元测试。

运行：python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uplift_lab import experiment as E  # noqa: E402


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestSRM(unittest.TestCase):
    def test_balanced_passes(self):
        res = E.srm_test({"a": 10000, "b": 10100, "c": 9950})
        self.assertGreater(res["p_value"], 0.05)

    def test_imbalanced_flagged(self):
        res = E.srm_test({"a": 11000, "b": 10000, "c": 9000})
        self.assertLess(res["p_value"], 1e-3)

    def test_expected_share_respected(self):
        # 2:1:1 的设计，人数按比例时不应报警
        res = E.srm_test(
            {"a": 10000, "b": 5000, "c": 5000},
            expected_share={"a": 0.5, "b": 0.25, "c": 0.25},
        )
        self.assertGreater(res["p_value"], 0.05)


class TestSMD(unittest.TestCase):
    def test_identical_groups_near_zero(self):
        rng = _rng()
        n = 500
        df = pd.DataFrame(
            {
                "is_treated": [1] * n + [0] * n,
                "x": np.concatenate([rng.normal(0, 1, n), rng.normal(0, 1, n)]),
                "g": ["a", "b"] * n,
            }
        )
        smd = E.standardized_mean_diff(df, numeric=["x"], categorical=["g"])
        self.assertLess(smd["smd"].abs().max(), 0.25)

    def test_shifted_group_detected(self):
        rng = _rng()
        n = 500
        df = pd.DataFrame(
            {
                "is_treated": [1] * n + [0] * n,
                "x": np.concatenate([rng.normal(1.0, 1, n), rng.normal(0, 1, n)]),
                "g": ["a", "b"] * n,
            }
        )
        smd = E.standardized_mean_diff(df, numeric=["x"], categorical=["g"])
        self.assertGreater(smd["smd"].abs().max(), 0.5)


class TestATE(unittest.TestCase):
    def test_diff_in_means_recovers_effect(self):
        rng = _rng(11)
        n = 200_000
        t = rng.integers(0, 2, n)
        y = rng.random(n) < np.where(t == 1, 0.30, 0.20)
        est = E.diff_in_means(y.astype(float), t)
        self.assertAlmostEqual(est.estimate, 0.10, delta=0.01)

    def test_ols_ate_recovers_effect_with_covariate(self):
        rng = _rng(12)
        n = 60_000
        t = rng.integers(0, 2, n)
        x = rng.normal(0, 1, n)
        y = 0.10 * t + 0.40 * x + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "is_treated": t, "x": x})
        est = E.ols_ate(df, outcome="y", numeric=["x"], categorical=[])
        self.assertAlmostEqual(est.estimate, 0.10, delta=0.02)
        self.assertLess(est.ci_low, est.estimate)
        self.assertGreater(est.ci_high, est.estimate)


class TestHolm(unittest.TestCase):
    def test_holm_monotone_and_bounded(self):
        adj = E.holm_adjust({"a": 0.001, "b": 0.04, "c": 0.20})
        self.assertAlmostEqual(adj["a"], 0.003, places=6)
        self.assertAlmostEqual(adj["b"], 0.08, places=6)
        self.assertAlmostEqual(adj["c"], 0.20, places=6)
        for k, v in adj.items():
            self.assertGreaterEqual(v, {"a": 0.001, "b": 0.04, "c": 0.20}[k] - 1e-12)
            self.assertLessEqual(v, 1.0)


class TestInteraction(unittest.TestCase):
    def test_detects_heterogeneous_effect(self):
        rng = _rng(21)
        n = 80_000
        t = rng.integers(0, 2, n)
        both = rng.random(n) < 0.20
        base = 0.05
        effect = np.where(both, 0.20, 0.05)
        y = (rng.random(n) < base + effect * t).astype(int)
        df = pd.DataFrame(
            {
                "is_treated": t,
                "both": np.where(both, "yes", "no"),
                "y": y,
            }
        )
        res = E.interaction_test(df, outcome="y", subgroup_col="both", focus="yes")
        self.assertAlmostEqual(res["diff"], 0.15, delta=0.02)
        self.assertLess(res["p_value"], 1e-6)

    def test_no_heterogeneity_not_flagged(self):
        rng = _rng(22)
        n = 40_000
        t = rng.integers(0, 2, n)
        g = np.where(rng.random(n) < 0.5, "yes", "no")
        y = (rng.random(n) < 0.05 + 0.10 * t).astype(int)
        df = pd.DataFrame({"is_treated": t, "g": g, "y": y})
        res = E.interaction_test(df, outcome="y", subgroup_col="g", focus="yes")
        self.assertGreater(res["p_value"], 0.01)


class TestPower(unittest.TestCase):
    def test_mde_shrinks_with_n(self):
        small = E.mde_proportion_diff(0.01253, 0.00573, 1_065)
        large = E.mde_proportion_diff(0.01253, 0.00573, 71_000)
        self.assertGreater(small, large)
        self.assertAlmostEqual(small, 0.0163, delta=0.002)
        self.assertAlmostEqual(large, 0.0020, delta=0.0008)

    def test_n_for_mde_inverts_mde(self):
        n = E.n_for_mde(0.002, 0.01253, 0.00573)
        self.assertAlmostEqual(n, 71_000, delta=2_500)


class TestCUPED(unittest.TestCase):
    def test_reduces_variance_when_covariate_predictive(self):
        rng = _rng(31)
        n = 20_000
        t = rng.integers(0, 2, n)
        x = rng.normal(0, 1, n)
        y = 0.30 * x + 0.10 * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "is_treated": t, "x": x})
        res = E.cuped(df, outcome="y", covariate="x")
        self.assertGreater(res.variance_reduction * 100, 5.0)

    def test_no_gain_when_covariate_noise(self):
        rng = _rng(32)
        n = 20_000
        t = rng.integers(0, 2, n)
        x = rng.normal(0, 1, n)
        y = 0.10 * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "is_treated": t, "x": x})
        res = E.cuped(df, outcome="y", covariate="x")
        self.assertLess(abs(res.variance_reduction * 100), 2.0)


if __name__ == "__main__":
    unittest.main()