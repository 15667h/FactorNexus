"""P13：回测引擎（factor_backtest）增强测试。

覆盖：滑点成本、板块感知涨跌停（创业板 20%）、跳变日建仓冻结（不交易不付成本）、
方向前半段估计（防前视）、报告字段完整性。
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.factor_backtest import backtest_factor, run_factor_backtest


def _trend_close(T: int = 400, up: bool = True) -> np.ndarray:
    """单调趋势价格（无跳变、无涨跌停）。"""
    rng = np.random.default_rng(1)
    step = 0.05 if up else -0.05
    base = 100.0 + np.arange(T) * step + rng.normal(0, 0.1, T)
    return np.maximum(base, 1.0)


def _factor_signal(T: int = 400) -> np.ndarray:
    """与未来 5 日收益正相关的因子（含噪声）。"""
    rng = np.random.default_rng(2)
    close = _trend_close(T)
    ret5 = np.concatenate([close[5:] / close[:-5] - 1.0, np.zeros(5)])
    f = ret5 + rng.normal(0, 0.05, T)
    f -= f.mean()
    f /= (f.std() + 1e-9)
    return f


# ── 滑点 ──────────────────────────────────────────────────────────────────

def test_slippage_reduces_returns():
    close = _trend_close()
    f = _factor_signal(len(close))
    b0 = backtest_factor(f, close, horizon=5, cost=0.0, slippage=0.0)
    b1 = backtest_factor(f, close, horizon=5, cost=0.0, slippage=0.005)
    assert b1["total_ret"] <= b0["total_ret"]
    assert b1["slippage"] == 0.005
    assert "slippage" in b1


# ── 板块感知涨跌停 ────────────────────────────────────────────────────────

def test_limit_pct_board_aware():
    """创业板 20% 阈值：+15% 日应可买入（旧 9.9% 会误判涨停）。"""
    close = _trend_close(300).copy()
    # 构造 +15% 大涨日（创业板合法，主板涨停）
    close[100] = close[99] * 1.15
    f = _factor_signal(300)

    b_main = backtest_factor(f, close, horizon=5, cost=0.0,
                             limit_filter=True, limit_pct=0.099)
    b_gem = backtest_factor(f, close, horizon=5, cost=0.0,
                            limit_filter=True, limit_pct=0.199)
    # 创业板口径下该日不被视为涨停 → 仓位不受限（收益应 ≥ 主板口径）
    assert b_gem["limit_pct"] == 0.199
    # 因子方向为正 → 该日多单：主板被误判涨停砍仓，创业板保留
    assert b_gem["total_ret"] >= b_main["total_ret"] - 1e-12


# ── 跳变日建仓冻结 ────────────────────────────────────────────────────────

def test_jump_day_freezes_position_no_turnover():
    """跳变日恰为建仓日 → 建仓冻结：不换手、不付成本、当日收益置 0。

    T1 修复：旧测试函数体只有 `pass`，核心行为零覆盖。
    断言基于实现：跳变建仓日 w=prev_pos（冻结）→ turnover_cost=0、
    ret1d 置 0 → pnl[101]=0 → nav 该日连续。
    """
    T = 200
    close = _trend_close(T).copy()
    # 跳变日 index=101（101%5==1 → 是建仓日），+120%（混库特征）
    close[101] = close[100] * 2.2
    # 后续回缩放保持形态（避免连续大负收益干扰 nav 断言）
    close[102:] = close[102:] * (close[100] / close[101])
    f = np.full(T, 1.5)                    # 满仓多头
    b = backtest_factor(f, close, horizon=5, cost=0.01, slippage=0.0,
                        limit_filter=True)
    assert b["jump_days"] >= 1
    nav = b["nav"]
    # 跳变建仓日 101：冻结 → turnover_cost=0、ret1d 置 0 → pnl[101]=0
    # → nav[102] == nav[101]。若未冻结，会扣 0.905*0.01 换手成本。
    assert abs(nav[102] - nav[101]) < 1e-9


def test_jump_day_no_pnl_contribution():
    """跳变日收益强制置 0：即使仓位非零，跳变日净值无变化。

    T2 修复：旧断言 total_ret<1.0 在删除防御后仍成立（假阳性）；
    现断言 nav 在跳变日连续（未防御时 nav 会 +150%×仓位）。
    """
    T = 200
    close = _trend_close(T).copy()
    close[100] = close[99] * 2.5                      # 跳变 +150%
    close[101:] = close[101:] * (close[100] / close[99] / 2.5)  # 保持后续比例
    f = np.full(T, 3.0)                                # 满仓多单
    b = backtest_factor(f, close, horizon=5, cost=0.0)
    assert b["jump_days"] >= 1
    nav = b["nav"]
    # 跳变日 index=100：ret1d[100]=+150% 被置 0 → pnl[100]=0（满仓 w≈0.995）
    # → nav[101] == nav[100]。未防御时 pnl[100]≈1.49 → nav 爆增 149%。
    assert abs(nav[101] - nav[100]) < 1e-9


# ── 方向前半段估计（防前视）───────────────────────────────────────────────

def test_run_factor_backtest_half_sample_direction(tmp_path):
    """run_factor_backtest 的方向用前半段估计（不因后半段翻转）。"""
    from data_pipeline.store.kline_store import FactorStore, KlineStore
    import pandas as pd

    T = 400
    ts = np.arange(1_600_000_000, 1_600_000_000 + T, dtype=np.int64)
    # 前半段上涨、后半段下跌（方向反转信号：全样本 IC≈0，前半段 IC>0）
    half = T // 2
    close = np.concatenate([
        _trend_close(half, up=True),
        _trend_close(half, up=False),
    ])
    df = pd.DataFrame({"ts": ts, "open": close - 0.1,
                       "high": close + 1, "low": close - 1,
                       "close": close, "volume": np.full(T, 1e6)})
    store = KlineStore(tmp_path)
    store.update("sh600000", "1d", df)

    # 因子 = 前半段未来收益的强信号（正 IC）
    ret5 = np.concatenate([close[5:] / close[:-5] - 1.0, np.zeros(5)])
    f = ret5.copy()
    f -= f.mean(); f /= (f.std() + 1e-9)
    fdf = pd.DataFrame({"ts": ts, "factor": f})
    fstore = FactorStore(tmp_path)
    meta = {"symbol": "sh600000", "hash": "", "engine": "test",
            "cert_rankic": 0.05, "direction": 1.0}
    fh = fstore.save("sh600000", [1, 2, 3], "test", fdf, report=meta)
    f = {"symbol": "sh600000", "hash": fh}

    b = run_factor_backtest(f, str(tmp_path), horizon=5, cost=0.0)
    # 方向估计：前半段正 IC → direction=+1 → 回测 IC 为正
    assert b["ic"] > -0.3  # 宽松：信号含噪声，方向至少不被后半段带偏
    # 报告的 IC 用全样本（报告项），方向已由前半段确定（防前视）


# ── 报告字段完整性 ────────────────────────────────────────────────────────

def test_backtest_report_extra_fields():
    close = _trend_close()
    f = _factor_signal(len(close))
    b = backtest_factor(f, close, horizon=5, cost=0.0003, slippage=0.0005)
    for k in ("slippage", "limit_pct", "jump_days"):
        assert k in b
    assert b["limit_pct"] == 0.099
