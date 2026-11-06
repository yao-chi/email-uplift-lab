"""uplift 评价指标的单元测试。

核心不变量：
- 完美排序的 AUUC 必须高于随机；
- 反向排序的 AUUC 必须低于随机；
- 随机排序的 AUUC 约等于解析随机基线；
- Qini 曲线在 100% 触达处等于总体增量。
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uplift_lab import metrics as M  # noqa: E402


def _sim(seed: int = 5, n: int = 60_000):
    """构造有异质性的数据：一半人增量为 0.20，一半人为 0.02。"""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    high = rng.random(n) < 0.5
    base = 0.05
    effect = np.where(high, 0.20, 0.02)
    y = (rng.random(n) < base + effect * t).astype(float)
    oracle = np.where(high, 1.0, 0.0)  # 完美排序
    return y, t, oracle, high


class TestQiniCurve(unittest.TestCase):
    def test_curve_shape_and_bounds(self):
        y, t, oracle, _ = _sim()
        curve = M.qini_curve(y, t, oracle)
        self.assertEqual(len(curve), len(y))
        self.assertTrue((curve["frac"] > 0).all())
        self.assertLessEqual(curve["frac"].max(), 1.0 + 1e-9)
        # 100% 触达时的累计增量 = 总体增量（按人数摊）
        self.assertAlmostEqual(
            curve["qini"].iloc[-1] / len(y), M.total_uplift_per_person(y, t), delta=1e-6
        )

    def test_scores_are_sorted_desc(self):
        y, t, oracle, _ = _sim()
        curve = M.qini_curve(y, t, oracle)
        self.assertTrue((np.diff(curve["frac"].to_numpy()) > 0).all())


class TestAUUC(unittest.TestCase):
    def test_oracle_beats_random(self):
        y, t, oracle, _ = _sim()
        # 50/50 分配下随机排序已吃掉一半面积，完美排序的比值有上限（约 1.4~1.5）
        ratio = M.auuc(y, t, oracle) / M.random_auuc(y, t)
        self.assertGreater(ratio, 1.2)
        self.assertLess(ratio, 2.0)

    def test_random_score_close_to_baseline(self):
        y, t, _, _ = _sim(seed=9)
        rng = np.random.default_rng(99)
        noisy = rng.normal(size=len(y))
        ratio = M.auuc(y, t, noisy) / M.random_auuc(y, t)
        self.assertGreater(ratio, 0.7)
        self.assertLess(ratio, 1.3)

    def test_inverted_score_hurts(self):
        y, t, oracle, _ = _sim()
        self.assertLess(M.auuc(y, t, -oracle), M.random_auuc(y, t))

    def test_qini_coefficient_sign(self):
        y, t, oracle, _ = _sim()
        self.assertGreater(M.qini_coefficient(y, t, oracle), 0.2)
        self.assertLess(M.qini_coefficient(y, t, -oracle), -0.2)

    def test_total_uplift_matches_group_means(self):
        y, t, _, _ = _sim()
        n_t, n_c = t.sum(), (1 - t).sum()
        expect = (y[t == 1].sum() - y[t == 0].sum() * n_t / n_c) / len(y)
        self.assertAlmostEqual(M.total_uplift_per_person(y, t), expect, places=10)


class TestGATE(unittest.TestCase):
    def test_deciles_partition_data(self):
        y, t, oracle, _ = _sim()
        gate = M.gate_table(y, t, oracle, q=10)
        self.assertEqual(len(gate), 10)
        self.assertEqual(int(gate["n"].sum()), len(y))

    def test_oracle_deciles_monotone(self):
        y, t, oracle, _ = _sim(seed=13)
        gate = M.gate_table(y, t, oracle, q=10).sort_values("group")
        ups = gate["obs_uplift"].to_numpy()
        self.assertGreater(ups[0], ups[-1] + 0.10)


class TestTopK(unittest.TestCase):
    def test_uplift_at_top_decreases_with_frac(self):
        y, t, oracle, _ = _sim()
        small = M.uplift_at_top(y, t, oracle, 0.1)["uplift"]
        large = M.uplift_at_top(y, t, oracle, 1.0)["uplift"]
        self.assertGreater(small, large)

    def test_topk_table_rows(self):
        y, t, oracle, _ = _sim()
        tab = M.topk_table(y, t, oracle, fracs=(0.05, 0.1, 1.0))
        self.assertEqual(len(tab), 3)
        for col in ("uplift", "ci_low", "ci_high", "n"):
            self.assertIn(col, tab.columns)
        self.assertLessEqual(tab["ci_low"].iloc[0], tab["uplift"].iloc[0])
        self.assertGreaterEqual(tab["ci_high"].iloc[0], tab["uplift"].iloc[0])


class TestBootstrap(unittest.TestCase):
    def test_ci_brackets_point_estimate(self):
        y, t, oracle, _ = _sim(seed=17, n=20_000)
        res = M.bootstrap_auuc(y, t, oracle, n_boot=200, seed=1)
        self.assertLess(res["ci_low"], res["ci_high"])
        self.assertLessEqual(res["ci_low"], res["auuc"] + 1e-9)
        self.assertGreaterEqual(res["ci_high"], res["auuc"] - 1e-9)


if __name__ == "__main__":
    unittest.main()