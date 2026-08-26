"""
tests/test_p10_institutional_spec.py — P10 机构规范落地测试

依据 docs/INSTITUTIONAL_SPEC.md（微软 Qlib / 华泰金工 / Bailey & López de Prado）：
  D3 数据健康检查（重复日期/大跳变/缺列、跳变日收益置 0）
  E2 分层回测（十分组单调性 + 多空收益）
  E3 IC 衰减曲线（lag 1..10）
  B3 涨跌停不可成交限制（Qlib 中国模式 limit=9.9%）

运行：python -m pytest tests/test_p10_institutional_spec.py -v
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

from data_pipeline.quality import check_series, clean_series, mask_jump_returns
from model_core.eval.report import build_factor_report
from scripts.factor_backtest import backtest_factor


# ── D3 数据健康检查 ─────────────────────────────────────────────────────────

def _df(close_ret=None, n=300, seed=0):
    rng = np.random.default_rng(seed)
    rets = np.random.default_rng(seed).normal(0.0002, 0.01, n) \
        if close_ret is None else np.asarray(close_ret, float)
    close = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({
        "ts": np.arange(n, dtype=np.int64) * 86400 + 1700000000,
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.abs(rng.normal(1e6, 2e5, n)),
    })


def test_check_series_healthy():
    assert check_series(_df()) == []


def test_check_series_detects_duplicates_and_jumps():
    df = _df()
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)  # 重复日期
    # 人为造一个大跳变（20% 跌停后的 30% 反弹 → 复权瑕疵型跳变）
    df = df.copy()
    df.loc[50, "close"] = df.loc[49, "close"] * 1.30
    issues = check_series(df)
    joined = " ".join(issues)
    assert "重复日期" in joined
    assert "异常跳变" in joined


def test_check_series_missing_columns():
    issues = check_series(pd.DataFrame({"ts": [1], "close": [1.0]}))
    assert any("缺列" in i for i in issues)


def test_clean_series_and_jump_mask():
    df = _df()
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
    df = df.copy()
    df.loc[50, "close"] = df.loc[49, "close"] * 1.30   # 30% 跳变（跳上）
    cleaned, jumps = clean_series(df)
    assert len(cleaned) == len(df) - 1          # 重复剔除
    # 30% 跳变会产生两个异常日（跳上 + 回落镜像），都应标记
    assert len(jumps) == 2
    # 跳变日收益标签置 0
    ret = np.ones(len(cleaned))
    masked = mask_jump_returns(ret, cleaned["ts"].values, jumps)
    assert masked[50] == 0.0 and masked[51] == 0.0 and masked[49] == 1.0


# ── E2 / E3 评估报告（IC 衰减 + 分层回测）──────────────────────────────────

def test_report_ic_decay_and_groups():
    rng = np.random.default_rng(0)
    T = 500
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, T))
    ret = np.zeros(T)
    ret[:T - 5] = close[5:] / close[:-5] - 1.0
    # 有预测力的因子（未来收益 + 噪声）
    factor = ret + rng.normal(0, 0.01, T)
    rep = build_factor_report(factor, ret, [1] * 10, "测试", n_trials=50)
    decay = rep.meta.get("ic_decay", [])
    assert len(decay) == 10                     # lag 1..10
    assert all(np.isfinite(decay))
    groups = rep.meta.get("group_returns", [])
    assert len(groups) == 10
    assert rep.meta.get("group_monotonicity", 0.0) > 0.5   # 强因子分组单调
    assert rep.meta.get("long_short_ret", 0.0) > 0         # 多空为正


# ── B3 涨跌停不可成交 ───────────────────────────────────────────────────────

def test_backtest_limit_filter_blocks_limit_up():
    """涨停日（≥9.9%）无法买入：涨停后连续暴跌的构造下，限制躲过净亏损。

    注：持仓期 [涨停日, +4] 的收益 = +10%（涨停当天）+ 4×(-5%) ≈ -10% < 0，
    故"无法买入"（仓位 0）优于"假装买入"。
    """
    rng = np.random.default_rng(0)
    n = 200
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.005, n))
    close[101] = close[100] * 1.10   # 涨停
    close[102:] = close[101] * np.cumprod(1 + rng.normal(-0.05, 0.005, n - 102))
    factor = np.full(n, 5.0)   # 强烈看多

    b_on = backtest_factor(factor, close, horizon=5, cost=0.0, limit_filter=True)
    b_off = backtest_factor(factor, close, horizon=5, cost=0.0, limit_filter=False)
    # 涨停日建仓失败 → 躲过 [涨停日, +4] 的净亏损
    assert b_on["total_ret"] >= b_off["total_ret"] - 1e-9


def test_backtest_limit_filter_blocks_limit_down():
    """跌停日（≤-9.9%）无法卖出：跌停后连续暴涨的构造下，做空信号被限制（躲过反弹亏损）。"""
    rng = np.random.default_rng(0)
    n = 200
    close = 100 * np.cumprod(1 + rng.normal(-0.0005, 0.005, n))
    close[101] = close[100] * 0.90   # 跌停
    close[102:] = close[101] * np.cumprod(1 + rng.normal(0.05, 0.005, n - 102))
    factor = np.full(n, -5.0)        # 强烈看空（做空信号）
    b_on = backtest_factor(factor, close, horizon=5, cost=0.0, limit_filter=True)
    b_off = backtest_factor(factor, close, horizon=5, cost=0.0, limit_filter=False)
    # 跌停日无法卖出（空头仓位置 0）→ 躲过后续反弹的做空亏损
    assert b_on["total_ret"] >= b_off["total_ret"] - 1e-9
