"""
model_core/strategy_factory/models/ — M3 预测模型池

首版基线：LightGBM（机构标配，Qlib 基准模型；实测 A 股多因子选股稳定）。
备选：sklearn GradientBoosting（无 lightgbm 时兜底）。

接口统一：fit(X, y) → predict(X)，供 walk_forward 调用。
"""
from __future__ import annotations

import numpy as np


def make_lgbm_regressor(n_estimators: int = 300, learning_rate: float = 0.05,
                        num_leaves: int = 31, min_child_samples: int = 30,
                        subsample: float = 0.8, colsample: float = 0.8,
                        seed: int = 42):
    """构造 LightGBM 回归器（走 sklearn API，兼容 walk_forward）。"""
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=min_child_samples,
            subsample=subsample, subsample_freq=1,
            colsample_bytree=colsample, random_state=seed,
            n_jobs=-1, verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=min(n_estimators, 200), learning_rate=learning_rate,
            max_depth=6, min_samples_leaf=min_child_samples,
            random_state=seed,
        )


def make_gbdt_regressor(n_estimators: int = 200, learning_rate: float = 0.05,
                        max_depth: int = 6, seed: int = 42):
    """sklearn GBDT 兜底（无 lightgbm 时）。"""
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, random_state=seed,
    )


def feature_importance(model) -> np.ndarray:
    """特征重要性（兼容 lightgbm / sklearn）。"""
    try:
        return np.asarray(model.feature_importances_, dtype=np.float64)
    except AttributeError:
        return np.zeros(0)
