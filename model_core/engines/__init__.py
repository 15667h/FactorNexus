"""model_core/engines 包 — 因子挖掘引擎注册层。"""
from model_core.engines.gp_engine import NSGA3FactorMiner, evaluate_five_objectives

__all__ = ["NSGA3FactorMiner", "evaluate_five_objectives"]
