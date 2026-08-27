"""
model_core/portfolio/impact_cost.py — 冲击成本模型（P21）

机构级交易成本：佣金 + 滑点 + 市场冲击（Almgren-Chriss 平方根法则）。

  冲击成本 ≈ k · σ · √(Q / ADV)   （Q=交易量, ADV=日均成交量, σ=日波动率）
  平方根法则（Almgren 2005 / 华泰等机构通用近似）：
    成本率 = a·σ·√(Q/ADV) + 固定滑点 + 佣金

用法：
    from model_core.portfolio.impact_cost import impact_cost_rate
    rate = impact_cost_rate(notional=1e6, adv=5e7, sigma=0.02, price=10.0)
    # 或接入回测：backtest_portfolio(..., impact=True)
"""
from __future__ import annotations

import numpy as np


def impact_cost_rate(notional: float, adv_notional: float,
                     sigma: float, k: float = 0.15,
                     fixed_slip: float = 0.0005,
                     commission: float = 0.0003) -> float:
    """单边冲击成本率（Almgren-Chriss 平方根法则）。

    Args:
        notional: 本次交易金额（元）。
        adv_notional: 该标的日均成交额（元，ADV）。
        sigma: 日波动率（如 0.02 = 2%）。
        k: 冲击系数（经验 0.1-0.25，A股常见 0.15）。
        fixed_slip: 固定滑点（默认 0.0005）。
        commission: 佣金费率（默认 0.0003）。

    Returns:
        单边成本率（如 0.002 = 0.2%）。
    """
    if adv_notional <= 0 or notional <= 0:
        return fixed_slip + commission
    participation = min(notional / adv_notional, 1.0)
    impact = k * sigma * float(np.sqrt(participation))
    return float(impact + fixed_slip + commission)


def participation_rate(notional: float, adv_notional: float) -> float:
    """参与率（交易量 / ADV）。"""
    if adv_notional <= 0:
        return 0.0
    return min(notional / adv_notional, 1.0)


def estimate_adv(volume: np.ndarray, price: np.ndarray,
                 window: int = 20) -> np.ndarray:
    """滚动 ADV（日均成交额）序列，因果（t 只用 t 及以前）。"""
    v = np.asarray(volume, dtype=np.float64)
    p = np.asarray(price, dtype=np.float64)
    amt = v * p
    n = len(amt)
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - window + 1)
        seg = amt[lo:t + 1]
        out[t] = float(np.mean(seg)) if len(seg) > 0 else np.nan
    return out


def daily_sigma(close: np.ndarray, window: int = 20) -> np.ndarray:
    """滚动日波动率（因果）。"""
    c = np.asarray(close, dtype=np.float64)
    ret = np.zeros(len(c))
    ret[1:] = c[1:] / np.maximum(c[:-1], 1e-9) - 1.0
    n = len(ret)
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - window + 1)
        seg = ret[lo:t + 1]
        out[t] = float(np.std(seg)) if len(seg) > 1 else 0.0
    return out
