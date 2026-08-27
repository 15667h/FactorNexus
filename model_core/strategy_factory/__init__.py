"""
model_core/strategy_factory/ — 策略工厂（P24 中层策略层）

把因子库加工成预测信号（对标 Qlib ML 选股管线 + 华泰 ML 合成）：

  dataset       M1 数据层：因子库 → 特征矩阵 + 未来 H 日收益标签（全因果）
  walk_forward  M2 walk-forward 训练框架（滚动训练/OOS 预测/gap 防泄漏）
  models        M3/M4 模型池：LightGBM 基线 + MLP + S4/SSM 对照
  ensemble      M5 集成：排名平均 / bagging / 异质集成 / stacking
  evaluate      信号评估：RankIC/ICIR/分层/换手/分段方向一致

用法：
    from model_core.strategy_factory import build_dataset, walk_forward_fit_predict
    ds = build_dataset("store", horizon=5)
    res = walk_forward_fit_predict(ds, model_factory=make_lgbm_regressor)
"""
from model_core.strategy_factory.dataset import FactorDataset, build_dataset
from model_core.strategy_factory.walk_forward import (
    WalkForwardResult, walk_forward_fit_predict,
)
from model_core.strategy_factory.models.lgbm_model import (
    feature_importance, make_gbdt_regressor, make_lgbm_regressor,
)
from model_core.strategy_factory.models.mlp_model import (
    MLPRegressor, make_mlp_regressor,
)
from model_core.strategy_factory.models.ssm_model import (
    S4Regressor, make_mamba_regressor, make_s4_regressor,
)
from model_core.strategy_factory.ensemble import (
    make_bagging_factory, make_ensemble_factory, rank_average,
    stacking_fit_predict,
)
from model_core.strategy_factory.evaluate import (
    cross_sectional_rankic, evaluate_signal, quantile_analysis,
)

__all__ = [
    "FactorDataset", "build_dataset",
    "WalkForwardResult", "walk_forward_fit_predict",
    "make_lgbm_regressor", "make_gbdt_regressor", "feature_importance",
    "MLPRegressor", "make_mlp_regressor",
    "S4Regressor", "make_s4_regressor", "make_mamba_regressor",
    "make_bagging_factory", "make_ensemble_factory", "rank_average",
    "stacking_fit_predict",
    "cross_sectional_rankic", "evaluate_signal", "quantile_analysis",
]
