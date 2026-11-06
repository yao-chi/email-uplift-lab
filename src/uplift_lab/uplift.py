"""Uplift 建模：S-learner / T-learner / 响应模型对照组。

三种建模思路的差别（面试必问）：
  * 响应模型 P(Y=1|X)：不分组建模，直接预测"谁最可能买"。营销上最常见的做法，
    但它分不清"本来就会买"和"因为被推了一把才买"，容易把钱发给最忠诚的老客。
  * S-learner：把处理标记当特征塞进一个模型，用 f(X,1)-f(X,0) 得到增量。
    优点简单，缺点是被正则化拉平（增量估计偏向 0）。
  * T-learner：处理组、对照组各建一个模型，再相减。缺点是两条曲线各自有噪声。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import ModelParams


def _make_model(params: ModelParams, seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=params.learning_rate,
        max_iter=params.max_iter,
        max_leaf_nodes=params.max_leaf_nodes,
        min_samples_leaf=params.min_samples_leaf,
        l2_regularization=params.l2_regularization,
        early_stopping=False,
        random_state=seed,
    )


@dataclass
class DesignMatrix:
    columns: list[str]

    def transform(self, df: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
        out = pd.get_dummies(df[cats].astype(str), drop_first=True, dtype=float)
        return out.reindex(columns=self.columns, fill_value=0.0)


def build_design(df: pd.DataFrame, numeric: list[str], cats: list[str]) -> pd.DataFrame:
    dummies = pd.get_dummies(df[cats].astype(str), drop_first=True, dtype=float)
    return pd.concat([df[numeric].astype(float).reset_index(drop=True),
                      dummies.reset_index(drop=True)], axis=1)


class UpliftModel:
    """统一的增量模型接口：fit(X, t, y) -> predict_uplift(X)。"""

    def __init__(self, kind: str = "t", params: ModelParams | None = None, seed: int = 0):
        assert kind in {"t", "s"}, kind
        self.kind = kind
        self.params = params or ModelParams()
        self.seed = seed
        self.columns: list[str] | None = None
        self.model_t = None
        self.model_c = None
        self.model_s = None

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        # 统一列顺序：训练时记住 columns，预测时对齐
        if self.columns is None:
            self.columns = list(df.columns)
        return df.reindex(columns=self.columns, fill_value=0.0)

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "UpliftModel":
        t = np.asarray(t, dtype=int)
        y = np.asarray(y, dtype=float)
        self.columns = list(X.columns)
        if self.kind == "t":
            self.model_t = _make_model(self.params, self.seed).fit(X[t == 1], y[t == 1])
            self.model_c = _make_model(self.params, self.seed + 1).fit(X[t == 0], y[t == 0])
        else:
            Xs = X.copy()
            Xs["__treat__"] = t
            self.model_s = _make_model(self.params, self.seed).fit(Xs, y)
            self.columns_s = list(Xs.columns)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X = self._prep(X)
        if self.kind == "t":
            return self.model_t.predict_proba(X)[:, 1] - self.model_c.predict_proba(X)[:, 1]
        Xs = X.copy()
        Xs["__treat__"] = 0
        p0 = self.model_s.predict_proba(Xs)[:, 1]
        Xs["__treat__"] = 1
        p1 = self.model_s.predict_proba(Xs)[:, 1]
        return p1 - p0


class ResponseModel:
    """对照组：不分处理，只预测成交概率（“谁会买”）。"""

    def __init__(self, params: ModelParams | None = None, seed: int = 0):
        self.params = params or ModelParams()
        self.seed = seed
        self.model = None
        self.columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ResponseModel":
        self.columns = list(X.columns)
        self.model = _make_model(self.params, self.seed).fit(X, np.asarray(y, dtype=float))
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.columns, fill_value=0.0)
        return self.model.predict_proba(X)[:, 1]