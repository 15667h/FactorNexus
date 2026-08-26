"""P14：组合层测试（中性化 / 正交化 / 合成 / 组合回测 / 归因）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_core.portfolio.combination import (
    combine_equal, combine_icir, combine_ml, icir_weights,
)
from model_core.portfolio.neutralization import (
    build_style_features, neutralize_panel,
)
from model_core.portfolio.orthogonalization import (
    incremental_rankic, orthogonalize_panel, orthogonalize_series,
)
from model_core.portfolio.portfolio import (
    backtest_portfolio, build_portfolio, performance, risk_model,
)
from model_core.portfolio.attribution import (
    brinson_attribution, style_attribution,
)


def _synthetic_panel(n_days: int = 120, n_stocks: int = 40,
                     seed: int = 0, signal: float = 0.05) -> pd.DataFrame:
    """合成因子面板：index=ts(日), columns=symbol, 含噪声信号。"""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n_days, freq="B")
    symbols = [f"sz{i:06d}" for i in range(n_stocks)]
    base = np.zeros((n_days, n_stocks))
    for t in range(1, n_days):
        base[t] = base[t - 1] + rng.normal(0, signal, n_stocks)
    noise = rng.normal(0, 0.5, (n_days, n_stocks))
    return pd.DataFrame(base + noise, index=ts, columns=symbols)


def _synthetic_klines(symbols: list[str], n_days: int = 120,
                      seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n_days, freq="B")
    out = {}
    for i, s in enumerate(symbols):
        close = 50.0 + rng.normal(0.05, 0.8, n_days).cumsum() + i
        close = np.maximum(close, 5.0)
        out[s] = pd.DataFrame({
            "ts": ts.astype("int64") // 10 ** 9,
            "open": close - 0.2, "high": close + 1.0,
            "low": close - 1.0, "close": close,
            "volume": rng.uniform(1e6, 5e6, n_days),
        })
    return out


def _synthetic_ret_panel(n_days: int = 120, n_stocks: int = 40,
                         seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n_days, freq="B")
    symbols = [f"sz{i:06d}" for i in range(n_stocks)]
    return pd.DataFrame(rng.normal(0.0005, 0.02, (n_days, n_stocks)),
                        index=ts, columns=symbols)


# ── 中性化 ────────────────────────────────────────────────────────────────

def test_build_style_features():
    df = _synthetic_klines(["sz000001"], n_days=80)["sz000001"]
    st = build_style_features("sz000001", df)
    assert set(st) == {"mcap", "ret20", "vol20", "turn20"}
    assert st["ret20"] is not None


def test_neutralize_panel_structure():
    panel = _synthetic_panel(n_days=60, n_stocks=30)
    klines = _synthetic_klines(list(panel.columns), n_days=60)
    neutral, report = neutralize_panel(panel, klines, industry_map={})
    assert report["degraded"] is True          # 无行业 → 风格中性化
    assert report["n_stocks"] >= 10
    assert neutral.shape == panel.shape
    # 中性化后均值应接近 0（截面 zscore）
    vals = neutral.values[np.isfinite(neutral.values)]
    assert abs(float(vals.mean())) < 0.5


def test_neutralize_with_industry():
    panel = _synthetic_panel(n_days=60, n_stocks=30)
    klines = _synthetic_klines(list(panel.columns), n_days=60)
    ind = {s: f"ind{i % 3}" for i, s in enumerate(panel.columns)}
    neutral, report = neutralize_panel(panel, klines, industry_map=ind)
    assert report["industries"] == 3
    assert report["degraded"] is False


# ── 正交化 ────────────────────────────────────────────────────────────────

def test_orthogonalize_series_removes_benchmark():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 300)
    bench = rng.normal(0, 1, 300)
    f = 3.0 * bench + x                    # 因子 = 3×基准 + 独立成分
    resid = orthogonalize_series(f, bench)
    # 残差与基准相关性应接近 0
    corr = np.corrcoef(resid[100:], bench[100:])[0, 1]
    assert abs(corr) < 0.15


def test_orthogonalize_panel_dims():
    p1 = _synthetic_panel(n_days=40, n_stocks=20, seed=3)
    p2 = _synthetic_panel(n_days=40, n_stocks=20, seed=4)
    out = orthogonalize_panel(p1, [p2])
    assert out.shape == p1.shape
    assert out.columns.tolist() == p1.columns.tolist()


def test_incremental_rankic_returns():
    panel = _synthetic_panel(n_days=60, n_stocks=30, seed=5)
    bench = _synthetic_panel(n_days=60, n_stocks=30, seed=6)
    ret = _synthetic_ret_panel(n_days=60, n_stocks=30, seed=7)
    r = incremental_rankic(panel, [bench], ret)
    assert set(r) == {"raw_rankic", "orth_rankic", "incremental"}
    assert np.isfinite(r["raw_rankic"])


# ── 合成 ──────────────────────────────────────────────────────────────────

def test_icir_weights_normalized():
    p1 = _synthetic_panel(n_days=80, n_stocks=30, seed=8)
    p2 = _synthetic_panel(n_days=80, n_stocks=30, seed=9)
    ret = _synthetic_ret_panel(n_days=80, n_stocks=30, seed=10)
    w = icir_weights([p1, p2], ret, window=40)
    assert len(w) == 2
    assert abs(float(np.abs(w).sum()) - 1.0) < 1e-6


def test_combine_icir_shape():
    p1 = _synthetic_panel(n_days=60, n_stocks=20, seed=11)
    p2 = _synthetic_panel(n_days=60, n_stocks=20, seed=12)
    ret = _synthetic_ret_panel(n_days=60, n_stocks=20, seed=13)
    comp, w = combine_icir([p1, p2], ret, window=30)
    assert comp.shape == p1.shape
    assert len(w) == 2


def test_combine_ml_smoke():
    p1 = _synthetic_panel(n_days=90, n_stocks=15, seed=14)
    p2 = _synthetic_panel(n_days=90, n_stocks=15, seed=15)
    ret = _synthetic_ret_panel(n_days=90, n_stocks=15, seed=16)
    comp, report = combine_ml([p1, p2], ret, window=50, n_estimators=20)
    assert comp.shape == p1.shape
    assert "n_estimators" in report


def test_combine_equal_shape():
    p1 = _synthetic_panel(n_days=40, n_stocks=15, seed=17)
    p2 = _synthetic_panel(n_days=40, n_stocks=15, seed=18)
    comp = combine_equal([p1, p2])
    assert comp.shape == p1.shape


# ── 组合构建 + 回测 ───────────────────────────────────────────────────────

def test_build_portfolio_long_short():
    panel = _synthetic_panel(n_days=60, n_stocks=40, seed=19)
    w = build_portfolio(panel, n_top=5, weights="equal", long_short=True)
    row = w.iloc[30]
    assert abs(float(row.sum())) < 1e-9        # 多空权重和 ≈ 0
    # 多空组合：多头暴露 1 + 空头暴露 1 = 总暴露 2
    assert float(row.abs().sum()) == pytest.approx(2.0, abs=1e-6)


def test_build_portfolio_long_only():
    panel = _synthetic_panel(n_days=40, n_stocks=20, seed=20)
    w = build_portfolio(panel, n_top=5, long_short=False)
    row = w.iloc[10]
    assert float(row.sum()) == pytest.approx(1.0, abs=1e-6)
    assert (row >= 0).all()


def test_backtest_portfolio_no_crash():
    panel = _synthetic_panel(n_days=80, n_stocks=30, seed=21)
    w = build_portfolio(panel, n_top=5)
    ret = _synthetic_ret_panel(n_days=80, n_stocks=30, seed=22)
    b = backtest_portfolio(w, ret, cost=0.0003)
    assert b["n"] > 0
    assert np.isfinite(b["total_ret"])
    assert "sharpe" in b and "max_dd" in b


def test_backtest_cost_reduces_return():
    panel = _synthetic_panel(n_days=80, n_stocks=30, seed=23)
    w = build_portfolio(panel, n_top=5)
    ret = _synthetic_ret_panel(n_days=80, n_stocks=30, seed=24)
    b0 = backtest_portfolio(w, ret, cost=0.0)
    b1 = backtest_portfolio(w, ret, cost=0.005)
    assert b1["total_ret"] <= b0["total_ret"] + 1e-9


def test_risk_model_and_performance():
    rng = np.random.default_rng(25)
    daily = rng.normal(0.0005, 0.01, 100)
    ex = rng.normal(0.5, 0.1, (100, 3))
    sr = rng.normal(0.0002, 0.005, (100, 3))
    rm = risk_model(daily, ex, sr)
    assert "vol" in rm and "var" in rm
    assert "style_risk" in rm
    perf = performance({"daily_ret": daily, "nav": np.cumprod(1 + daily),
                        "total_ret": 0.1}, bench_ret=daily * 0.5)
    assert "excess_ret" in perf and "info_ratio" in perf


# ── 归因 ──────────────────────────────────────────────────────────────────

def test_brinson_attribution():
    pw = np.array([0.3, 0.5, 0.2])
    bw = np.array([0.4, 0.4, 0.2])
    ind_ret = np.array([0.02, -0.01, 0.03])
    att = brinson_attribution(0.015, 0.008, pw, bw, ind_ret)
    assert set(att) == {"allocation", "selection", "interaction", "total"}
    assert abs(att["total"] - (0.015 - 0.008)) < 1e-9


def test_style_attribution():
    rng = np.random.default_rng(26)
    ex = rng.normal(0.5, 0.1, (100, 2))
    sr = rng.normal(0.0003, 0.004, (100, 2))
    daily = (ex * sr).sum(axis=1) + rng.normal(0, 0.001, 100)
    att = style_attribution(daily, ex, sr)
    assert "style_contrib" in att and "idiosyncratic" in att
    assert att["r2"] > 0.5               # 构造数据风格应解释大部分
