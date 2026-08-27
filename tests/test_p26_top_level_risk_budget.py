"""
tests/test_p26_top_level_risk_budget.py — P19 顶层风险预算面板优化回归测试

覆盖 optimize_portfolio_panel：三种优化器（markowitz/risk_parity/
black_litterman）、多空/纯多头归一化、持有期语义、防前视窗口、NaN 安全。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_core.portfolio.optimizer import (
    black_litterman, markowitz, optimize_portfolio_panel, risk_parity,
)


def _synth_panels(n_days: int = 200, n_stocks: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = pd.to_datetime(
        pd.date_range("2025-01-01", periods=n_days, freq="B"))
    cols = [f"sh6000{i:02d}" for i in range(n_stocks)]
    # 信号：部分股票有正 alpha 结构（前 4 只高分，后 4 只低分）
    base = np.zeros(n_stocks)
    base[:4] = 1.0
    base[-4:] = -1.0
    score = pd.DataFrame(
        base[None, :] * 2.0 + rng.normal(0, 0.5, (n_days, n_stocks)),
        index=idx, columns=cols)
    # 收益：与信号正相关的 alpha + 噪声（1 日收益）
    alpha = rng.normal(0.0005, 0.0002, (n_days, n_stocks)) * base[None, :]
    noise = rng.normal(0, 0.01, (n_days, n_stocks))
    ret = pd.DataFrame(alpha + noise, index=idx, columns=cols)
    return score, ret


# ── 基本形状与归一化 ────────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["markowitz", "risk_parity",
                                    "black_litterman"])
def test_panel_shape_and_axis(method):
    score, ret = _synth_panels()
    w = optimize_portfolio_panel(score, ret, method=method, n_top=4,
                                 window=40, rebalance=5)
    assert w.shape == score.shape
    assert list(w.columns) == list(score.columns)
    assert list(w.index) == list(score.index)


def test_long_short_leverage_two():
    score, ret = _synth_panels()
    w = optimize_portfolio_panel(score, ret, method="risk_parity", n_top=4,
                                 window=40, rebalance=5, long_short=True)
    active = w.abs().sum(axis=1).dropna()
    # 多空：Σ|w| = 2
    assert float(active.median()) == pytest.approx(2.0, abs=0.05)


def test_long_only_leverage_one():
    score, ret = _synth_panels()
    w = optimize_portfolio_panel(score, ret, method="markowitz", n_top=4,
                                 window=40, rebalance=5, long_short=False)
    active = w.sum(axis=1).dropna()
    # 纯多头：Σw = 1
    assert float(active.median()) == pytest.approx(1.0, abs=0.05)


def test_risk_parity_long_weights_positive():
    score, ret = _synth_panels()
    w = optimize_portfolio_panel(score, ret, method="risk_parity", n_top=4,
                                 window=40, rebalance=5, long_short=False)
    active_rows = w[w.abs().sum(axis=1) > 0]
    assert float(active_rows.min().min()) >= -1e-8   # 纯多头无负权重


# ── 持有期语义 ──────────────────────────────────────────────────────────

def test_rebalance_hold_semantics():
    score, ret = _synth_panels(n_days=60)
    w = optimize_portfolio_panel(score, ret, method="markowitz", n_top=4,
                                 window=40, rebalance=5)
    idx = list(w.index)
    # 第 0-4 天权重必须完全相同（区间内持有）
    r0 = w.iloc[0].values
    for j in range(1, 5):
        assert np.allclose(w.iloc[j].values, r0, equal_nan=True)
    # 前 30 天应为空仓（协方差窗口 <30 天 → NaN；第 31 天起有权重）
    assert w.iloc[:30].isna().all().all()


# ── 防前视：权重只用 t 及以前的收益 ─────────────────────────────────────

def test_no_lookahead_window():
    score, ret = _synth_panels(n_days=80)
    # 篡改：把最后 10 天收益放大 100 倍——若权重泄漏未来，中段权重会变化
    w_base = optimize_portfolio_panel(score, ret, method="markowitz",
                                      n_top=4, window=40, rebalance=10)
    ret2 = ret.copy()
    ret2.iloc[-10:] *= 100.0
    w_fut = optimize_portfolio_panel(score, ret2, method="markowitz",
                                     n_top=4, window=40, rebalance=10)
    # 第 40-70 天（不受未来篡改影响）权重应完全一致
    for j in range(40, 70):
        assert np.allclose(w_base.iloc[j].values, w_fut.iloc[j].values,
                           equal_nan=True), f"第 {j} 天出现未来泄漏"


# ── BL 观点生效 ────────────────────────────────────────────────────────

def test_black_litterman_views_affect_weights():
    score, ret = _synth_panels()
    w_bl = optimize_portfolio_panel(score, ret, method="black_litterman",
                                    n_top=4, window=40, rebalance=10)
    # BL 多空：Top 池应整体为正、Bottom 池为负（观点驱动）
    active = w_bl[w_bl.abs().sum(axis=1) > 0]
    cols = list(score.columns)
    long_med = active[cols[:4]].stack().median()
    short_med = active[cols[-4:]].stack().median()
    assert long_med > 0
    assert short_med < 0


# ── NaN 安全 ───────────────────────────────────────────────────────────

def test_insufficient_history_nan():
    score, ret = _synth_panels(n_days=20)          # 窗口不足 30 天
    w = optimize_portfolio_panel(score, ret, method="markowitz", n_top=4,
                                 window=40, rebalance=5)
    assert w.isna().all().all()                     # 全部空仓，不抛异常


# ── 底层优化器回归（已有测试的补充快照）────────────────────────────────

def test_optimizer_primitives_still_work():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, (120, 5))
    w = markowitz(r, risk_aversion=2.0)
    assert np.allclose(np.abs(w).sum(), 1.0, atol=1e-6)
    w_rp = risk_parity(r)
    assert np.all(w_rp > 0) and np.allclose(w_rp.sum(), 1.0)
    w_bl = black_litterman(r, np.full(5, 0.2), views={0: 0.05},
                           view_conf=0.5)
    assert np.allclose(np.abs(w_bl).sum(), 1.0, atol=1e-6)
