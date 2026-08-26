"""
model_core/portfolio/ — 机构级组合层（P14，对标华泰因子工厂 + Qlib + Barra）

模块：
  neutralization   五因子中性化（行业/市值/20日收益/20日波动/20日换手）
  orthogonalization 因子正交化（残差收益率目标增量挖掘）
  combination      多因子合成（IC_IR 加权 / 随机森林 / 等权）
  portfolio        组合构建 + 组合回测 + 简化风险模型（Qlib Portfolio 对齐）
  attribution      Brinson 绩效归因 + 风格风险归因
"""
from model_core.portfolio.neutralization import (
    build_style_features,
    fetch_industry_map,
    neutralize_panel,
)
from model_core.portfolio.orthogonalization import (
    incremental_rankic,
    orthogonalize_panel,
    orthogonalize_series,
)
from model_core.portfolio.combination import (
    combine_equal,
    combine_icir,
    combine_ml,
    icir_weights,
)
from model_core.portfolio.portfolio import (
    backtest_portfolio,
    build_portfolio,
    performance,
    risk_model,
)
from model_core.portfolio.attribution import (
    brinson_attribution,
    style_attribution,
)

__all__ = [
    "build_style_features", "fetch_industry_map", "neutralize_panel",
    "incremental_rankic", "orthogonalize_panel", "orthogonalize_series",
    "combine_equal", "combine_icir", "combine_ml", "icir_weights",
    "backtest_portfolio", "build_portfolio", "performance", "risk_model",
    "brinson_attribution", "style_attribution",
]
