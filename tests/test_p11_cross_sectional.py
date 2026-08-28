"""
tests/test_p11_cross_sectional.py — P11 横截面认证测试（机构范式）

对标 Qlib/华泰：因子质量以"跨股票普适性"认证——
候选公式跨股票池执行 → 每个交易日截面 RankIC → 块自助 p 值。
覆盖：
  1. _cross_sectional_rankics：多股票对齐 → 截面 RankIC 序列
  2. _certify_batch：真动量公式跨 10 只股票 → 通过；噪声公式 → 拒绝
  3. 股票池不足时降级单标的认证

运行：python -m pytest tests/test_p11_cross_sectional.py -v
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

from data_pipeline.store.kline_store import KlineStore
from scripts.mine_full_market import (
    _certify_batch, _certify_single_series, _cross_sectional_rankics,
    MarketPool,
)


# ── 工具：构造 AR(1) 动量行情（过去收益可预测未来）─────────────────────────

def _ar1_close(T=600, phi=0.30, seed=0):
    """收益 AR(1) 正自相关 → 动量因子（过去收益）对 AR(1) 未来收益有预测力。"""
    rng = np.random.default_rng(seed)
    rets = np.empty(T)
    rets[0] = rng.normal(0, 0.01)
    for t in range(1, T):
        rets[t] = phi * rets[t - 1] + rng.normal(0, 0.01)
    return 100 * np.cumprod(1 + rets)


def _stock_df(close):
    T = len(close)
    return pd.DataFrame({
        "ts": (1700000000 + np.arange(T) * 86400).astype(np.int64),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.abs(np.random.default_rng(1).normal(1e6, 2e5, T)),
    })


def _populate_store(tmp_path, n_stocks=30, phi=0.80):
    store = KlineStore(tmp_path)
    for i in range(n_stocks):
        sym = f"sh6000{i:02d}"
        store.update(sym, "1d", _stock_df(_ar1_close(seed=i, phi=phi)))
    return store


# 动量公式：A=ret, window=10, mode=1, mode1=Momentum（过去 10 日收益变化）
_MOMENTUM_CHROM = [6, 0, 2, 0, 0, 0, 0, 12, 0, 0]
# 噪声公式：A=close, mode1=Last（水平类，截面无普适预测力）
_LAST_CHROM = [0, 0, 2, 0, 0, 0, 0, 9, 0, 0]


def _cand(chrom, source="sh600000", engine="gp", five=0.8, factor=None,
          ret=None, ts=None):
    from model_core.formula_dsl import chrom_to_formula
    f = chrom_to_formula(chrom)
    return {
        "source": source, "engine": engine, "kind": "param",
        "chrom": list(chrom), "desc": f.describe(), "n_trials": 100,
        "factor": factor, "ret": ret, "ts": ts, "five_total": five,
    }


# ── 1. 截面 RankIC 序列 ─────────────────────────────────────────────────────

def test_cross_sectional_rankics_alignment():
    """多股票 (ts, factor, ret) → 按共同交易日对齐的截面 RankIC 序列。"""
    ts = np.arange(300, dtype=np.int64)
    rng = np.random.default_rng(0)
    series = []
    for i in range(30):
        f = rng.normal(0, 1, 300)
        r = rng.normal(0, 0.01, 300)
        series.append((ts, f, r))
    rankics, days = _cross_sectional_rankics(series)
    assert days == 300
    assert len(rankics) == 300
    # 独立噪声 → 均值接近 0
    assert abs(np.mean(rankics)) < 0.1


def test_cross_sectional_rankics_strong_signal():
    """跨股票一致的强信号 → 截面 RankIC 显著为正。"""
    ts = np.arange(300, dtype=np.int64)
    series = []
    for i in range(30):
        r = 0.02 * np.sin(ts / 20.0 + i)          # 各股票共享的周期模式
        f = r + np.random.default_rng(i).normal(0, 0.002, 300)  # 因子≈未来收益
        series.append((ts, f, r))
    rankics, _ = _cross_sectional_rankics(series)
    assert np.mean(rankics) > 0.5


# ── 2. 批级认证（10 只股票池）──────────────────────────────────────────────

class _Cfg:
    def __init__(self, store_dir):
        self.store_dir = str(store_dir)
        self.tf = "1d"
        self.bars = 600
        self.horizon = 5
        self.oos_frac = 0.25
        self.min_oos_rankic = 0.02
        self.min_oos_p = 0.05
        self.dsr_gate = 0.0


def test_certify_batch_momentum_passes(tmp_path):
    """AR(1) 行情下，动量公式跨 30 只股票应通过横截面认证。"""
    _populate_store(tmp_path, n_stocks=30, phi=0.80)
    cfg = _Cfg(tmp_path)
    pool = MarketPool(tmp_path)
    ctx = {"pool": pool}
    symbols = [f"sh6000{i:02d}" for i in range(30)]
    cands = [_cand(_MOMENTUM_CHROM, source=symbols[0])]
    accepted, stats = _certify_batch(cands, symbols, "1d", cfg, pool, ctx)
    assert stats["n_stocks"] == 30
    assert len(accepted) == 1, f"动量公式应通过，stats={stats}"
    assert accepted[0]["cert"]["mode"] == "cross_sectional"
    assert accepted[0]["cert"]["rankic"] > 0.02
    assert accepted[0]["cert"]["p"] <= 0.05
    assert accepted[0]["cert"]["stocks"] == 30
    # M1 修复后横截面认证只评估每只股票的 OOS 段（oos_frac=0.25 → 600 根
    # K 线的 OOS 段约 250 天，有效截面日 245）。门槛随 OOS 段自适应
    # （min(250, max(200, oos_len-10))），故此处断言有效截面日 ≥ 200 即可。
    assert accepted[0]["cert"]["days"] >= 200


def test_certify_batch_noise_rejected(tmp_path):
    """随机游走市（无预测性）下，Last 水平公式跨股票应被拒绝。"""
    _populate_store(tmp_path, n_stocks=30, phi=0.0)
    cfg = _Cfg(tmp_path)
    pool = MarketPool(tmp_path)
    ctx = {"pool": pool}
    symbols = [f"sh6000{i:02d}" for i in range(30)]
    cands = [_cand(_LAST_CHROM, source=symbols[0])]
    accepted, stats = _certify_batch(cands, symbols, "1d", cfg, pool, ctx)
    assert len(accepted) == 0, f"随机游走下噪声公式应被拒绝，stats={stats}"
    assert stats["n_reject"] >= 1 or stats["n_cross"] == 0


def test_certify_single_series_fallback(tmp_path):
    """股票池不足（<8）→ 降级单标的认证：强信号通过、噪声拒绝。"""
    store = KlineStore(tmp_path)
    df = _stock_df(_ar1_close(T=600, phi=0.30, seed=0))
    store.update("sh600000", "1d", df)
    cfg = _Cfg(tmp_path)

    from scripts.mine_full_market import _build_ret
    close = df["close"].values.astype(float)
    ret = _build_ret(close, 5)
    ts = df["ts"].values.astype(np.int64)
    n = len(ret)

    # 强候选：因子 = 未来收益标签 + 小噪声（同位上帝因子，OOS 显著）
    f_strong = ret + np.random.default_rng(0).normal(0, 0.002, n)
    c_strong = _cand(_MOMENTUM_CHROM, factor=f_strong, ret=ret, ts=ts)
    ok = _certify_single_series(c_strong, cfg)
    assert ok is not None and ok["cert"]["mode"] == "single_series"

    # 噪声候选：常数因子
    c_noise = _cand(_LAST_CHROM, factor=np.zeros(n), ret=ret, ts=ts)
    bad = _certify_single_series(c_noise, cfg)
    assert bad is None
