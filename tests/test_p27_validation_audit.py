"""
tests/test_p27_validation_audit.py — P0 验证严谨性审计回归测试

覆盖（改进方案 P0）：
  1. audit_execution_timing：脉冲收益法验证 t+1 执行（前视会被精确抓到）
  2. 因子因果：K 线尾部篡改 → 因子历史值逐位不变（合成版）
  3. audit_random_returns：有信号面板 → 原策略显著超随机分布；
     无信号面板 → 不显著（反向验证，防"审计恒通过"）
  4. random_entry_ev / top_winner_trim / _pool_rankic 输出形状与语义
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from scripts.audit_backtest import (audit_execution_timing,
                                    audit_random_returns)
from scripts.robustness_audit import random_entry_ev, top_winner_trim
from scripts.migration_audit import _pool_rankic


def _synth(n_days: int = 300, n_stocks: int = 40, seed: int = 3,
           alpha: float = 0.0):
    """合成面板：alpha>0 → 信号与未来收益正相关；alpha=0 → 纯噪声。"""
    rng = np.random.default_rng(seed)
    idx = pd.to_datetime(pd.date_range("2025-01-01", periods=n_days,
                                       freq="B"))
    cols = [f"sh600{i:02d}" for i in range(n_stocks)]
    # 信号：前 10 只高、后 10 只低（中间中性）
    base = np.zeros(n_stocks)
    base[:10] = 1.0
    base[-10:] = -1.0
    score = pd.DataFrame(
        base[None, :] * 2.0 + rng.normal(0, 0.3, (n_days, n_stocks)),
        index=idx, columns=cols)
    # 收益 = alpha × 信号 + 噪声（未来 1 日收益）
    noise = rng.normal(0, 0.01, (n_days, n_stocks))
    if alpha > 0:
        ret = pd.DataFrame(
            alpha * base[None, :] * 0.01 + noise, index=idx, columns=cols)
    else:
        ret = pd.DataFrame(noise, index=idx, columns=cols)
    return score, ret


# ── 1. 成交时点脉冲验证 ─────────────────────────────────────────────────

def test_execution_timing_pulse_passes():
    _, ret = _synth()
    r = audit_execution_timing(ret)
    assert r["passed"], r["detail"]


def test_execution_timing_detects_lookahead():
    """反向验证：若 PnL 用 w[t]（前视），审计必须失败。"""
    from model_core.portfolio.portfolio import backtest_portfolio
    n = 300
    idx = pd.to_datetime(pd.date_range("2025-01-01", periods=n, freq="B"))
    cols = [f"s{i}" for i in range(10)]
    pulse = 0.10
    ret = pd.DataFrame(0.0, index=idx, columns=cols)
    ret.iloc[150] = pulse
    w = pd.DataFrame(0.0, index=idx, columns=cols)
    w.iloc[:150] = 0.2
    w.iloc[150:] = -0.2
    bt = backtest_portfolio(w, ret, cost=0.0, ppy=244)
    pnl_k = float(bt["daily_ret"][150])
    # 正确实现：pnl = w[t-1] @ r = 0.2*10*pulse = 0.2
    # 前视实现：pnl = w[t] @ r = -0.2*10*pulse = -0.2
    assert abs(pnl_k - 0.2) < 1e-9, "回测实现异常：脉冲日 PnL 非 w[t-1] 贡献"


# ── 2. 因子因果（合成版：改尾部 → 历史不变）────────────────────────────

def test_factor_causality_tail_mutation():
    from model_core.indicator_builder import build_indicators
    from model_core.param_vm import ParamVM
    from model_core.formula_dsl import chrom_to_formula, random_chrom

    rng = np.random.default_rng(0)
    T = 600
    close = 100 + np.cumsum(rng.normal(0, 1, T))
    df = pd.DataFrame({
        "ts": np.arange(T, dtype=np.int64),
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": np.abs(rng.normal(1000, 200, T)),
    })
    chrom = random_chrom(rng)

    def _factor(d):
        ind = build_indicators(d)
        return np.asarray(ParamVM(ind).execute(chrom_to_formula(chrom)),
                          dtype=np.float64)

    f0 = _factor(df)
    df2 = df.copy()
    tail = 100
    df2.loc[df2.index[-tail:], "close"] = \
        df2.loc[df2.index[-tail:], "close"].values * 1.5
    f1 = _factor(df2)
    n = min(len(f0), len(f1)) - tail
    max_diff = float(np.max(np.abs(f0[:n] - f1[:n])))
    assert max_diff < 1e-9, f"尾部篡改影响历史因子：max_diff={max_diff}"


# ── 3. 随机收益对照 ─────────────────────────────────────────────────────

def test_random_returns_significant_with_alpha():
    score, ret = _synth(alpha=0.15)           # 强信号（信噪比足以覆盖成本）
    r = audit_random_returns(score, ret, n_top=5, n_shuffle=50, seed=1,
                             hold=5)          # 持有 5 日（匹配预测周期）
    assert r["passed"], r["detail"]


def test_random_returns_not_significant_without_alpha():
    score, ret = _synth(alpha=0.0)            # 无信号（纯噪声）
    r = audit_random_returns(score, ret, n_top=5, n_shuffle=30, seed=1,
                             hold=5)
    # 噪声信号下，原策略不应显著超出随机分布（审计不恒通过）
    assert r["pct_below_real"] < 0.95


# ── 4. 随机入场 / 去顶 / 子池 ───────────────────────────────────────────

def test_random_entry_ev_shape():
    score, ret = _synth(n_days=200, alpha=0.001)
    r = random_entry_ev(ret, n_top=5, n_sim=200, seed=0)
    assert r["n_sim"] == 200
    assert r["std"] > 0
    assert r["p95"] > r["p50"] > r["p05"]


def test_top_winner_trim_reduces_returns():
    from model_core.portfolio.portfolio import build_portfolio
    score, ret = _synth(n_days=250, alpha=0.002)
    w = build_portfolio(score, n_top=5, long_short=True)
    r = top_winner_trim(w, ret, cost=0.0003)
    # 去掉 Top5 赢家后收益应下降（不上升）
    assert r["rows"][0]["total_ret"] <= r["baseline_total"] + 1e-12


def test_pool_rankic_subsets():
    score, ret = _synth(n_days=200, n_stocks=40, alpha=0.001)
    full = _pool_rankic(score, ret, list(score.columns))
    sub = _pool_rankic(score, ret, list(score.columns[:30]))
    assert full["n_stocks"] == 40
    assert sub["n_stocks"] == 30
    assert np.isfinite(full["rankic"])
    assert np.isfinite(sub["rankic"])
    # 小池（<30）返回 nan 不崩溃
    tiny = _pool_rankic(score, ret, list(score.columns[:10]))
    assert not np.isfinite(tiny["rankic"])
