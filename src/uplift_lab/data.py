"""数据加载与体检。

设计原则：先体检再分析。一个实验数据集最常出的问题是
(1) 分组不均衡（SRM）、(2) 指标口径错、(3) 重复行/脏值。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import RAW_CSV

REQUIRED_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]


@dataclass
class AuditReport:
    n_rows: int
    n_columns: int
    duplicate_rows: int
    missing_cells: int
    arm_counts: dict[str, int]
    outcome_ranges: dict[str, tuple[float, float]]
    constant_columns: list[str]
    sha256: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "duplicate_rows": self.duplicate_rows,
            "missing_cells": self.missing_cells,
            "arm_counts": self.arm_counts,
            "outcome_ranges": {k: list(v) for k, v in self.outcome_ranges.items()},
            "constant_columns": self.constant_columns,
            "sha256": self.sha256,
            "notes": self.notes,
        }


def file_sha256(path=RAW_CSV) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    return df


def audit(df: pd.DataFrame, sha: str = "") -> AuditReport:
    """做一份可写进报告的体检结论。"""
    notes: list[str] = []

    # 交易约束：conversion=1 必须伴随 visit=1；spend>0 必须 conversion=1
    bad_funnel = int(((df["conversion"] == 1) & (df["visit"] == 0)).sum())
    bad_spend = int(((df["spend"] > 0) & (df["conversion"] == 0)).sum())
    if bad_funnel:
        notes.append(f"conversion=1 但 visit=0 的行: {bad_funnel}")
    if bad_spend:
        notes.append(f"spend>0 但 conversion=0 的行: {bad_spend}")

    zero_spend = int(((df["conversion"] == 1) & (df["spend"] == 0)).sum())
    if zero_spend:
        notes.append(f"成交但 spend=0 的行: {zero_spend}（口径需说明）")

    dup = int(df.duplicated().sum())
    if dup:
        notes.append(
            f"完全重复行 {dup} 行：数据集没有用户 ID，重复行可能是不同的真实用户，"
            "去重会人为缩小样本并改变基准率，因此保留（这是与'脏数据去重'相反的处理）"
        )

    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant:
        notes.append(f"常量列: {constant}")

    return AuditReport(
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        duplicate_rows=int(df.duplicated().sum()),
        missing_cells=int(df.isna().sum().sum()),
        arm_counts={str(k): int(v) for k, v in df["segment"].value_counts().items()},
        outcome_ranges={
            "visit": (float(df.visit.min()), float(df.visit.max())),
            "conversion": (float(df.conversion.min()), float(df.conversion.max())),
            "spend": (float(df.spend.min()), float(df.spend.max())),
        },
        constant_columns=constant,
        sha256=sha,
        notes=notes,
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """派生特征。所有派生都只用到实验前信息或分组标签，不碰结果变量。"""
    out = df.copy()
    out["arm"] = out["segment"].astype(str)
    out["is_treated"] = (out["segment"] != "No E-Mail").astype(int)
    out["is_mens"] = (out["segment"] == "Mens E-Mail").astype(int)
    out["is_womens"] = (out["segment"] == "Womens E-Mail").astype(int)

    out["recency_band"] = pd.cut(
        out["recency"],
        bins=[0, 3, 6, 9, 12],
        labels=["R1_1_3", "R2_4_6", "R3_7_9", "R4_10_12"],
    ).astype(str)

    out["history_band"] = pd.cut(
        out["history"],
        bins=[0, 100, 200, 350, 500, 750, 1000, np.inf],
        labels=["H1_<100", "H2_100_200", "H3_200_350", "H4_350_500", "H5_500_750", "H6_750_1000", "H7_1000+"],
    ).astype(str)

    out["multi_category"] = (out["mens"] + out["womens"] > 1).astype(int)
    out["buyer_type"] = np.select(
        [(out["mens"] > 0) & (out["womens"] > 0), out["mens"] > 0, out["womens"] > 0],
        ["both", "mens_only", "womens_only"],
        default="none",
    )
    out["buyer_type"] = pd.Series(out["buyer_type"], index=out.index).astype(str)
    return out


def train_test_split_stratified(
    df: pd.DataFrame, t_col: str = "is_treated", y_col: str = "conversion",
    test_size: float = 0.5, seed: int = 0,
):
    """按 (处理组, 结果) 四格分层切分，保证两半的转化率一致。"""
    rng = np.random.default_rng(seed)
    stratum = df[t_col].astype(str) + "_" + df[y_col].astype(str)
    test_idx: list[int] = []
    for _, idx in df.groupby(stratum, observed=True).groups.items():
        idx = np.asarray(idx)
        n_test = int(round(len(idx) * test_size))
        if n_test:
            test_idx.extend(rng.choice(idx, size=n_test, replace=False).tolist())
    mask = df.index.isin(test_idx)
    return df.loc[~mask].copy(), df.loc[mask].copy()