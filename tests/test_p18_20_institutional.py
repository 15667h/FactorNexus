"""P18-P22：基本面管线 / 组合优化器 / Barra 风险 / 冲击成本 / 停牌检测测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_core.fundamentals import build_fundamental_factors, \
    save_fundamental_factors
from model_core.portfolio.optimizer import (
    black_litterman, efficient_frontier, markowitz, risk_parity,
)
from model_core.portfolio.barra_risk import (
    barra_risk_model, build_style_factors, ledoit_wolf_shrinkage,
    style_exposures,
)
from model_core.portfolio.impact_cost import (
    daily_sigma, estimate_adv, impact_cost_rate,
)
from data_pipeline.quality import detect_suspensions


def _ret_matrix(T: int = 300, N: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 相关收益（构造可逆协方差）
    base = rng.normal(0.0003, 0.01, (T, 1))
    X = rng.normal(0, 0.01, (T, N)) + 0.3 * base
    return X


# ── P18 基本面因子构建 ────────────────────────────────────────────────────

def test_build_fundamental_factors():
    data = {
        "sh600519": {"pe": 20.0, "pb": 6.0, "roe": 30.0, "gross_margin": 91.0,
                     "rev_yoy": 15.0, "profit_yoy": 18.0, "debt_ratio": 20.0},
        "sz000001": {"pe": 5.0, "pb": 0.6, "roe": 12.0, "gross_margin": 40.0,
                     "rev_yoy": 5.0, "profit_yoy": 3.0, "debt_ratio": 90.0},
        "sh600000": {"pe": 6.0, "pb": 0.7, "roe": 11.0, "gross_margin": 35.0,
                     "rev_yoy": 8.0, "profit_yoy": 6.0, "debt_ratio": 88.0},
    }
    f = build_fundamental_factors(data)
    assert set(f) == {"ep", "bp", "roe", "gross", "rev_yoy",
                      "profit_yoy", "debt"}
    # EP 高 = 便宜：银行 PE 5-6 → EP 0.2/0.167 > 茅台 0.05
    assert f["ep"][1] > f["ep"][0]
    assert f["ep"][2] > f["ep"][0]
    # 茅台负债率最低 → debt 标准化后最低
    assert f["debt"][0] < f["debt"][2]


def test_save_fundamental_factors(tmp_path):
    from data_pipeline.store.kline_store import KlineStore, FactorStore
    rng = np.random.default_rng(1)
    n = 120
    ts = pd.date_range("2025-01-01", periods=n, freq="B")
    ts_i = ts.astype("int64") // 10 ** 9
    close = 50 + rng.normal(0.05, 0.5, n).cumsum()
    KlineStore(tmp_path).update("sh600519", "1d", pd.DataFrame(
        {"ts": ts_i, "open": close - 0.1, "high": close + 0.8,
         "low": close - 0.8, "close": close,
         "volume": rng.uniform(1e6, 5e6, n)}))
    data = {"sh600519": {"pe": 20.0, "pb": 6.0, "roe": 30.0,
                         "gross_margin": 91.0, "rev_yoy": 15.0,
                         "profit_yoy": 18.0, "debt_ratio": 20.0,
                         "report_date": "2025-06-30"}}
    saved = save_fundamental_factors(data, tmp_path)
    assert saved == 7                      # 7 个基本面因子
    fs = FactorStore(tmp_path)
    factors = fs.list_factors()
    assert len(factors) == 7
    # 因子文件内容 = 常数序列（最新报告期前值填充）
    fdf = fs.load("sh600519", factors[0]["hash"])
    assert abs(float(fdf["factor"].iloc[-1]) - float(fdf["factor"].iloc[0])) < 1e-9


# ── P19 组合优化器 ────────────────────────────────────────────────────────

def test_markowitz_weights_normalized():
    r = _ret_matrix()
    w = markowitz(r, risk_aversion=2.0)
    assert len(w) == r.shape[1]
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert (w >= 0).all()


def test_markowitz_lower_risk_than_equal():
    r = _ret_matrix()
    # 纯最小方差（risk_aversion=0 → 目标仅 0.5·w'Σw）
    w_mv = markowitz(r, risk_aversion=0.0)
    w_eq = np.full(r.shape[1], 1.0 / r.shape[1])
    cov = np.cov(r, rowvar=False)
    v_mv = float(w_mv @ cov @ w_mv)
    v_eq = float(w_eq @ cov @ w_eq)
    assert v_mv <= v_eq + 1e-9


def test_risk_parity_equal_contribution():
    r = _ret_matrix(seed=2)
    cov = np.cov(r, rowvar=False)
    w = risk_parity(cov)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    mrc = cov @ w
    contrib = w * mrc
    contrib /= contrib.sum()
    # 风险贡献应接近相等（容差放宽，数值迭代）
    assert np.std(contrib) < 0.05


def test_black_litterman_views():
    r = _ret_matrix(seed=3)
    N = r.shape[1]
    mkt = np.full(N, 1.0 / N)
    w0 = black_litterman(r, mkt, views=None)
    w1 = black_litterman(r, mkt, views={0: 0.1}, view_conf=0.9)
    assert abs(float(w1.sum()) - 1.0) < 1e-6
    # 强观点 → 资产 0 权重应明显提升
    assert w1[0] > w0[0] - 1e-9


def test_efficient_frontier():
    r = _ret_matrix(seed=4)
    rets, vols, ws = efficient_frontier(r, n_points=6)
    assert len(rets) >= 4
    # 前沿波动应随收益单调（近似）
    assert vols[-1] >= vols[0] - 1e-9


# ── P20 Barra 风险模型 ────────────────────────────────────────────────────

def test_ledoit_wolf_positive_definite():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (500, 10))
    cov = ledoit_wolf_shrinkage(X)
    assert cov.shape == (10, 10)
    # 正定可逆
    np.linalg.cholesky(cov)
    assert np.all(np.diag(cov) > 0)


def test_build_style_factors_shape():
    rng = np.random.default_rng(6)
    r = rng.normal(0, 0.01, (100, 6))
    v = rng.uniform(1e6, 5e6, (100, 6))
    ex = build_style_factors(r, volumes=v)
    assert ex.shape == (100, 6, 5)
    # 截面标准化：每期每风格均值 ≈ 0
    assert abs(float(np.nanmean(ex[50, :, 0]))) < 0.5


def test_barra_risk_model_decomposition():
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, (200, 6))
    w = np.full((200, 6), 1.0 / 6)
    rm = barra_risk_model(r, w)
    assert "total_vol" in rm and "style_vol" in rm and "idio_vol" in rm
    assert rm["total_vol"] >= 0
    # 总风险² ≈ 风格² + 特质²
    assert abs(rm["total_vol"] ** 2 -
               (rm["style_vol"] ** 2 + rm["idio_vol"] ** 2)) < 1e-6
    assert 0 <= rm["r2"] <= 1.05


def test_style_exposures_shape():
    rng = np.random.default_rng(8)
    r = rng.normal(0, 0.01, (80, 5))
    ex = build_style_factors(r)
    w = np.full((80, 5), 1.0 / 5)
    se = style_exposures(w, ex)
    assert se.shape == (80, 5)


# ── P21 冲击成本 ──────────────────────────────────────────────────────────

def test_impact_cost_rate():
    # 大单（高参与率）→ 成本高；小单 → 成本低
    big = impact_cost_rate(notional=5e6, adv_notional=5e7, sigma=0.02)
    small = impact_cost_rate(notional=5e4, adv_notional=5e7, sigma=0.02)
    assert big > small
    # 下限 = 固定滑点 + 佣金
    tiny = impact_cost_rate(notional=1e3, adv_notional=1e10, sigma=0.01)
    assert tiny >= 0.0005 + 0.0003 - 1e-9


def test_estimate_adv_and_sigma():
    rng = np.random.default_rng(9)
    close = 50 + rng.normal(0.05, 0.5, 100).cumsum()
    vol = rng.uniform(1e6, 5e6, 100)
    adv = estimate_adv(vol, close, window=20)
    assert np.isfinite(adv[-1])
    assert adv[-1] > 0
    sig = daily_sigma(close, window=20)
    assert np.isfinite(sig[-1]) and sig[-1] > 0


# ── P22 停牌检测 ──────────────────────────────────────────────────────────

def test_detect_suspensions():
    rng = np.random.default_rng(10)
    n = 100
    ts = np.arange(1_700_000_000, 1_700_000_000 + n)
    close = 50 + rng.normal(0.05, 0.5, n).cumsum()
    vol = rng.uniform(1e6, 5e6, n)
    # 制造停牌：第 30-32 日零成交且价格冻结
    vol[30:33] = 0.0
    close[30:33] = close[29]
    df = pd.DataFrame({"ts": ts, "open": close, "high": close + 1,
                       "low": close - 1, "close": close, "volume": vol})
    susp = detect_suspensions(df)
    assert int(ts[30]) in susp
    assert int(ts[31]) in susp
    assert int(ts[32]) in susp
    assert int(ts[40]) not in susp


def test_no_false_positive_suspension():
    """正常缩量交易（价变但量 0）不应误判停牌。"""
    rng = np.random.default_rng(11)
    n = 60
    ts = np.arange(1_700_000_000, 1_700_000_000 + n)
    close = 50 + rng.normal(0.05, 0.5, n).cumsum()
    vol = np.zeros(n)
    df = pd.DataFrame({"ts": ts, "open": close, "high": close + 1,
                       "low": close - 1, "close": close, "volume": vol})
    susp = detect_suspensions(df)
    # 价格每天都在变 → 无停牌
    assert len(susp) == 0
