"""全局配置：路径、随机种子、业务假设参数。

业务假设集中在 BusinessAssumptions 里，方便读者一眼看到"结论依赖哪些假设"。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SQL_DIR = ROOT / "sql"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
REPORT_DIR = ROOT / "reports"

RAW_CSV = DATA_DIR / "hillstrom_email.csv"

RANDOM_STATE = 20261106
N_BOOTSTRAP = 2000
TEST_SIZE = 0.5  # 训练/评估各一半，评估集足够大才能稳定估计 Qini

ARMS = ("Mens E-Mail", "Womens E-Mail", "No E-Mail")
CONTROL_ARM = "No E-Mail"
MAIN_TREATMENT_ARM = "Mens E-Mail"  # 增量建模主分析用效应最强的臂
PRIMARY_OUTCOME = "conversion"  # 业务主指标
SECONDARY_OUTCOME = "visit"  # 样本量充足的辅助指标

NUMERIC_COVARIATES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_COVARIATES = ["zip_code", "channel", "history_segment", "recency_band"]


@dataclass(frozen=True)
class BusinessAssumptions:
    """决策层假设。改这里，预算结论会变，报告里必须同步说明。"""

    cost_per_email: float = 0.10  # 单封邮件成本（与 spend 同币种）
    gross_margin: float = 0.40  # 成交额的毛利率
    fallback_aov: float = 120.0  # 数据里没有成交均值时的兜底客单价


@dataclass(frozen=True)
class ModelParams:
    learning_rate: float = 0.05
    max_iter: int = 300
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 200
    l2_regularization: float = 1.0