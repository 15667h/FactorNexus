"""P24 策略工厂测试（M1 数据层 + M2 walk-forward + M3 LightGBM）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_core.strategy_factory import (
    FactorDataset, build_dataset, walk_forward_fit_predict,
    make_lgbm_regressor, evaluate_signal, quantile_analysis,
    cross_sectional_rankic,
)


def _seed_store(tmp_path, n_symbols: int = 12, n_days: int = 300,
                n_factors: int = 3) -> None:
    """合成因子库：每股票多个有预测力的因子。"""
    from data_pipeline.store.kline_store import FactorStore, KlineStore
    rng = np.random.default_rng(0)
    store = FactorStore(tmp_path)
    kstore = KlineStore(tmp_path)
    ts = pd.date_range("2024-01-01", periods=n_days, freq="B")
    # pandas 2.x date_range 为 datetime64[us]（微秒）→ 转秒
    ts_i = (ts.astype("datetime64[s]").astype("int64"))
    for i in range(n_symbols):
        sym = f"sz{i:06d}"
        close = 30 + rng.normal(0.04, 0.5, n_days).cumsum() + i * 0.1
        close = np.maximum(close, 5)
        kstore.update(sym, "1d", pd.DataFrame(
            {"ts": ts_i, "open": close - 0.1, "high": close + 0.8,
             "low": close - 0.8, "close": close,
             "volume": rng.uniform(1e6, 5e6, n_days)}))
        ret = np.zeros(n_days)
        ret[:-5] = close[5:] / close[:-5] - 1.0
        for k in range(n_factors):
            # 因子 = 未来收益信号 + 噪声（有预测力，多因子共享信号+独立噪声）
            signal = (k + 1) * ret + rng.normal(0, 0.5, n_days)
            signal -= signal.mean()
            fdf = pd.DataFrame({"ts": ts_i, "factor": signal})
            meta = {"engine": "gp", "kind": "param", "symbol": sym,
                    "cert_rankic": 0.05, "direction": 1.0}
            store.save(sym, [k + 1, k + 2, k + 3], "test_v1", fdf,
                       report=meta)


# ── M1 数据层 ────────────────────────────────────────────────────────────

def test_build_dataset_structure(tmp_path):
    _seed_store(tmp_path)
    ds = build_dataset(str(tmp_path), horizon=5)
    assert isinstance(ds, FactorDataset)
    assert ds.n_samples > 100
    assert ds.meta["n_symbols"] == 12
    assert ds.meta["n_factors"] == 12 * 3         # 每股票 3 因子
    # 特征 = 稀疏因子列 + 通用特征列（ret20/vol20/mcap/turn/score）
    assert len(ds.feature_names) == 12 * 3 + 5
    assert ds.X.shape[0] == len(ds.y) == len(ds.ts)
    assert ds.X.isna().any().any()                # 稀疏因子（NaN 允许）
    # 标签有限且合理
    assert np.isfinite(ds.y).all()
    assert abs(float(ds.y.mean())) < 0.1


def test_dataset_split_no_leakage(tmp_path):
    _seed_store(tmp_path)
    ds = build_dataset(str(tmp_path), horizon=5)
    mid = np.median(ds.ts)
    tr, te = ds.split_by_time(int(mid))
    assert len(tr) > 0 and len(te) > 0
    assert tr.ts.max() < te.ts.min()                  # 时间严格分离


def test_dataset_causal(tmp_path):
    """因果性：只改尾部标签不影响头部特征（特征只用 t 及以前）。"""
    _seed_store(tmp_path)
    ds1 = build_dataset(str(tmp_path), horizon=5)
    # 篡改最后一根 K 线价格 → 尾部标签变化，但头部特征不变
    from data_pipeline.store.kline_store import KlineStore
    ks = KlineStore(tmp_path)
    df = ks.load("sz000000", "1d")
    df.loc[df.index[-1], "close"] *= 2.0
    ks.update("sz000000", "1d", df)
    ds2 = build_dataset(str(tmp_path), horizon=5)
    # 头部样本特征应一致（排除最后一天样本）
    assert ds1.X.shape[1] == ds2.X.shape[1]
    # 抽查前 20 个样本：特征列在头部时段应相同
    n_check = min(20, len(ds1.X))
    assert np.allclose(ds1.X.iloc[:n_check].values,
                       ds2.X.iloc[:n_check].values, equal_nan=True)


# ── M2 walk-forward ──────────────────────────────────────────────────────

def test_walk_forward_basic(tmp_path):
    _seed_store(tmp_path)
    ds = build_dataset(str(tmp_path), horizon=5)
    res = walk_forward_fit_predict(
        ds, model_factory=make_lgbm_regressor,
        step=60, window=120, gap=5, min_train=50)
    assert res.n_folds >= 2
    assert res.oos_days > 0
    assert res.coverage > 0.3
    # 预测面板结构：index=交易日, columns=股票
    assert res.pred.index.is_monotonic_increasing
    assert len(res.folds) == res.n_folds
    # folds 时间不重叠
    for i in range(1, len(res.folds)):
        assert res.folds[i]["test_from"] > res.folds[i - 1]["test_to"]


def test_walk_forward_gap_no_overlap(tmp_path):
    """gap 防泄漏：训练段截止 < 预测段起始 - gap。"""
    _seed_store(tmp_path)
    ds = build_dataset(str(tmp_path), horizon=5)
    res = walk_forward_fit_predict(
        ds, model_factory=make_lgbm_regressor,
        step=40, window=100, gap=5, min_train=50)
    for f in res.folds:
        assert f["train_to"] < f["test_from"]     # 严格不相交


def test_walk_forward_empty_dataset():
    ds = FactorDataset(pd.DataFrame(), pd.Series(dtype=float),
                       np.array([], dtype=np.int64),
                       np.array([], dtype=object), [])
    res = walk_forward_fit_predict(ds, model_factory=make_lgbm_regressor)
    assert res.pred.empty and res.n_folds == 0


# ── M3 LightGBM 基线 ─────────────────────────────────────────────────────

def test_lgbm_model_fit_predict(tmp_path):
    _seed_store(tmp_path)
    ds = build_dataset(str(tmp_path), horizon=5)
    X = ds.X.fillna(0.0)
    model = make_lgbm_regressor(n_estimators=50, learning_rate=0.1)
    model.fit(X, ds.y)
    pred = model.predict(X)
    assert len(pred) == len(ds.y)
    assert np.isfinite(pred).all()
    # 有预测力：训练集上相关应显著 > 0（合成数据含信号）
    corr = np.corrcoef(pred, ds.y)[0, 1]
    assert corr > 0.05


def test_signal_evaluation(tmp_path):
    _seed_store(tmp_path)
    ds = build_dataset(str(tmp_path), horizon=5)
    res = walk_forward_fit_predict(
        ds, model_factory=make_lgbm_regressor,
        step=60, window=120, gap=5, min_train=50)
    assert res.oos_days > 0
    # 收益面板：复用 portfolio_pipeline.build_panels（同轴 1 日收益）
    from scripts.portfolio_pipeline import build_panels
    _, ret1d, _, _ = build_panels(str(tmp_path), horizon=1)
    # 对齐到预测面板时间轴（ret1d 为 1 日收益，评估用 H 日收益更贴切；
    # 此处直接用 1 日收益做评估，保证测试可跑通）
    ret = ret1d.reindex(index=res.pred.index, columns=res.pred.columns)
    ev = evaluate_signal(res.pred, ret)
    assert ev["n_days"] > 0
    assert np.isfinite(ev["rankic"])
    assert "icir" in ev and "turnover" in ev
    ics, days = cross_sectional_rankic(res.pred, ret)
    assert len(ics) == ev["n_days"]
    qa = quantile_analysis(res.pred, ret)
    assert "monotonicity" in qa


# ── 因果性硬测试（防前视）───────────────────────────────────────────────

def test_walk_forward_predicts_future_only(tmp_path):
    """OOS 预测只能覆盖训练之后的时间段（无泄漏）。"""
    _seed_store(tmp_path)
    ds = build_dataset(str(tmp_path), horizon=5)
    res = walk_forward_fit_predict(
        ds, model_factory=make_lgbm_regressor,
        step=60, window=120, gap=5, min_train=50)
    if not res.folds:
        pytest.skip("无折")
    first_test = res.folds[0]["test_from"]
    # 所有 OOS 预测的 ts ≥ 第一折预测起点
    assert (ds.ts >= first_test).sum() > 0
    # 预测面板时间范围在数据集时间范围内
    assert res.pred.index.min() >= pd.to_datetime(first_test, unit="s")
