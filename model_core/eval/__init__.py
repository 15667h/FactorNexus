"""model_core/eval 包 — 因子评价层（AlphaEval 五维 + 过拟合统计控制）。"""
from model_core.eval.five_dim import FiveDimScore, five_dim_evaluate
from model_core.eval.significance import compute_dsr, compute_pbo_cscv, cpcv_paths
from model_core.eval.report import FactorReport, build_factor_report

__all__ = [
    "FiveDimScore", "five_dim_evaluate",
    "compute_dsr", "compute_pbo_cscv", "cpcv_paths",
    "FactorReport", "build_factor_report",
]
