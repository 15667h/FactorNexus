"""
model_core/portfolio/attribution.py — 绩效归因（P14）

对标华泰 Brinson 模型 + 风格风险归因：
  1. brinson_attribution : 配置效应 + 选股效应 + 交互效应（行业维度）
  2. style_attribution   : 风格收益回归分解（组合收益 = 风格暴露×风格收益 + 特质）

用法：
    from model_core.portfolio.attribution import brinson_attribution
    att = brinson_attribution(port_ret, bench_ret, port_w, bench_w, ind_ret)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def brinson_attribution(port_ret: np.ndarray, bench_ret: np.ndarray,
                        port_industry_weights: np.ndarray,
                        bench_industry_weights: np.ndarray,
                        industry_returns: np.ndarray) -> dict:
    """Brinson 单期归因（行业维度）。

    Args:
        port_ret: 组合总收益（标量或 [T] 均值）。
        bench_ret: 基准总收益。
        port_industry_weights: [K] 组合行业权重。
        bench_industry_weights: [K] 基准行业权重。
        industry_returns: [K] 行业收益。

    Returns:
        {"allocation": 配置效应, "selection": 选股效应,
         "interaction": 交互效应, "total": 总超额}
    """
    pw = np.asarray(port_industry_weights, dtype=np.float64)
    bw = np.asarray(bench_industry_weights, dtype=np.float64)
    ir = np.asarray(industry_returns, dtype=np.float64)
    k = min(len(pw), len(bw), len(ir))
    pw, bw, ir = pw[:k], bw[:k], ir[:k]
    bench = float(bench_ret)
    total = float(port_ret) - bench
    allocation = float(np.sum((pw - bw) * (ir - bench)))
    selection = float(np.sum(bw * ir)) - bench
    # 交互效应 = 余量，保证 allocation + selection + interaction == total（自洽）。
    # 历史 bug：旧实现 `Σ(pw-bw)·(ir-bench)*0.0 + Σ(pw-bw)·ir - allocation`
    # 在权重归一化（Σpw=Σbw=1）下恒等于 bench·Σ(pw-bw) = 0，
    # 且三效应之和 ≠ total（归因结果与超额收益对不上）。
    interaction = total - allocation - selection
    return {"allocation": allocation, "selection": selection,
            "interaction": interaction, "total": total}


def style_attribution(daily_ret: np.ndarray,
                      style_exposure: np.ndarray,
                      style_returns: np.ndarray) -> dict:
    """风格风险归因：组合收益 = Σ 暴露×风格收益 + 特质（OLS 分解）。

    Returns:
        {"style_contrib": {name: 贡献收益}, "idiosyncratic": 特质收益,
         "r2": 风格解释度}
    """
    r = np.asarray(daily_ret, dtype=np.float64)
    ex = np.asarray(style_exposure, dtype=np.float64)
    sr = np.asarray(style_returns, dtype=np.float64)
    n = min(len(r), len(ex), len(sr))
    if n < 20:
        return {"style_contrib": {}, "idiosyncratic": float(r.mean()),
                "r2": 0.0}
    r, ex, sr = r[-n:], ex[-n:], sr[-n:]
    k = ex.shape[1]
    # 每期贡献 = 暴露 × 风格收益；特质 = 组合收益 - Σ贡献
    contrib = ex * sr                       # [T, K]
    fitted = contrib.sum(axis=1)
    idio = r - fitted
    total_var = float(np.var(r))
    r2 = 1.0 - float(np.var(idio)) / total_var if total_var > 1e-12 else 0.0
    return {
        "style_contrib": {f"style_{i}": float(contrib[:, i].mean())
                          for i in range(k)},
        "idiosyncratic": float(idio.mean()),
        "r2": r2,
    }
