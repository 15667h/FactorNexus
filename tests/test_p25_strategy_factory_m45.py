"""
tests/test_p25_strategy_factory_m45.py — M4 模型池 + M5 集成 回归测试

覆盖：MLP / S4 冒烟（NaN 处理、fit/predict 形状）、rank_average 权重正确性、
bagging seed 多样性、异质集成、stacking 输出、_hold_weights 持有语义。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_core.strategy_factory import (
    make_bagging_factory, make_ensemble_factory, make_mlp_regressor,
    make_s4_regressor, rank_average, stacking_fit_predict,
)
from model_core.strategy_factory.dataset import FactorDataset
from model_core.strategy_factory.models.lgbm_model import make_lgbm_regressor


def _synth(n: int = 600, f: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    X[::7, 3] = np.nan                    # 稀疏 NaN（模拟因子缺失）
    y = X[:, 0] * 0.5 + X[:, 1] * 0.2 + rng.normal(scale=0.1, size=n)
    return X, y


# ── M4：模型池 ────────────────────────────────────────────────────────────

def test_mlp_smoke_fit_predict():
    X, y = _synth()
    m = make_mlp_regressor(epochs=2, batch_size=128, device="cpu")
    m.fit(X, y)
    p = m.predict(X)
    assert p.shape == (len(X),)
    assert np.isfinite(p).sum() >= len(X) * 0.95   # NaN 输入不应产生 NaN 输出
    assert float(np.std(p)) > 1e-6                  # 有区分度


def test_s4_smoke_fit_predict():
    X, y = _synth()
    m = make_s4_regressor(epochs=2, batch_size=128, device="cpu")
    m.fit(X, y)
    p = m.predict(X)
    assert p.shape == (len(X),)
    assert np.isfinite(p).sum() >= len(X) * 0.95


def test_mlp_missing_x_fit_error():
    m = make_mlp_regressor(epochs=1, device="cpu")
    with pytest.raises(ValueError):
        m.fit(np.zeros((10, 5)), np.zeros(9))       # 形状不匹配


# ── M5：集成 ──────────────────────────────────────────────────────────────

def test_rank_average_equal_weights():
    idx = pd.to_datetime(["2026-01-01", "2026-01-02"])
    p1 = pd.DataFrame({"a": [1.0, 3.0], "b": [3.0, 1.0]}, index=idx)
    p2 = pd.DataFrame({"a": [3.0, 1.0], "b": [1.0, 3.0]}, index=idx)
    out = rank_average([p1, p2])
    # 第 1 行：a=(0+1)/2=0.5, b=(1+0)/2=0.5（M4 归一化秩等权平均）
    assert np.allclose(out.iloc[0].values, [0.5, 0.5])
    assert np.allclose(out.iloc[1].values, [0.5, 0.5])


def test_rank_average_weighted_and_nan():
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    p1 = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, 1.0, 2.0]},
                      index=idx)
    p2 = pd.DataFrame({"a": [2.0, 1.0, 3.0], "b": [1.0, 3.0, 2.0]},
                      index=idx)
    out = rank_average([p1, p2], weights=[0.7, 0.3])
    # 第 1 行：a=0.7*0+0.3*1=0.3, b=0.7*1+0.3*0=0.7（归一化秩加权）
    assert np.allclose(out.iloc[0].values, [0.3, 0.7])
    # 第 3 行 a 只有 p2 有值：p2 a=3 行内 rank=2→归一化 1.0 → 0.3*1/0.3=1.0
    assert np.allclose(out.iloc[2]["a"], 1.0)


def test_bagging_seed_diversity():
    X, y = _synth()
    f = lambda seed=42: make_lgbm_regressor(n_estimators=20, seed=seed)  # noqa: E731
    b = make_bagging_factory(f, n_models=3)()
    b.fit(X, y)
    assert len(b._models) == 3
    p0 = b._models[0].predict(X)
    p1 = b._models[1].predict(X)
    assert float(np.abs(p0 - p1).max()) > 1e-8     # seed 多样性生效


def test_ensemble_rank_avg_model():
    X, y = _synth()
    ef = make_ensemble_factory([
        lambda: make_lgbm_regressor(n_estimators=10),       # noqa: E731
        lambda: make_mlp_regressor(epochs=1, device="cpu"),  # noqa: E731
    ], method="rank_avg")
    m = ef()
    m.fit(X, y)
    p = m.predict(X)
    assert p.shape == (len(X),)
    assert np.isfinite(p).sum() >= len(X) * 0.95
    assert float(np.nanstd(p)) > 1e-6


def test_stacking_fit_predict_shape():
    X, y = _synth(n=900)
    ds = FactorDataset(
        X=pd.DataFrame(X), y=pd.Series(y),
        ts=np.arange(len(X)),
        symbol=np.array(["s1"] * 300 + ["s2"] * 300 + ["s3"] * 300,
                        dtype=object),
        feature_names=list(range(X.shape[1])))
    from sklearn.linear_model import Ridge
    st = stacking_fit_predict(
        ds, [lambda: make_lgbm_regressor(n_estimators=10)],  # noqa: E731
        lambda: Ridge(alpha=1.0), split_frac=0.7, gap=1)
    assert not st.pred.empty
    assert st.pred.shape[0] >= 30
    assert st.n_folds == 1


# ── M6：组合持有平滑 ──────────────────────────────────────────────────────

def test_hold_weights_semantics():
    from scripts.mine_signal import _hold_weights
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03",
                          "2026-01-04", "2026-01-05"])
    w = pd.DataFrame([[0.2, -0.2], [0.1, -0.1], [0.3, -0.3],
                      [0.0, 0.0], [0.2, -0.2]], index=idx, columns=["a", "b"])
    h = _hold_weights(w, 2)
    assert np.allclose(h.iloc[0].values, [0.2, -0.2])
    assert np.allclose(h.iloc[1].values, [0.2, -0.2])   # 区间内持有
    assert np.allclose(h.iloc[2].values, [0.3, -0.3])
    assert np.allclose(h.iloc[4].values, [0.2, -0.2])
    # hold=1 原样返回
    h1 = _hold_weights(w, 1)
    assert np.allclose(h1.values, w.values)


# ── T4 回归：walk-forward gap 防泄漏 + 公式编解码往返 ──────────────────────

def test_walk_forward_gap_defaults_to_horizon():
    """gap=None 时自动取 horizon（防训练标签泄漏）；gap<horizon 告警。"""
    import inspect
    from model_core.strategy_factory.walk_forward import (
        walk_forward_fit_predict, _train_predict)
    sig = inspect.signature(walk_forward_fit_predict)
    assert sig.parameters["gap"].default is None      # 默认自动
    # 源码含防泄漏逻辑
    src = inspect.getsource(walk_forward_fit_predict)
    assert "ds.meta.get(\"horizon\", 5)" in src
    assert "gap < horizon" in src                     # 过小 gap 告警
    # _train_predict 支持 ts 透传（逐日截面集成，M3）
    sig2 = inspect.signature(_train_predict)
    assert "ts_te" in sig2.parameters


def test_walk_forward_gap_leak_prevented():
    """horizon=10 时 gap=0（自动→10）不泄漏；显式 gap=2 触发告警且结果含泄漏警告。"""
    from model_core.strategy_factory.walk_forward import walk_forward_fit_predict
    rng = np.random.default_rng(0)
    n = 400
    ts = np.arange(1_700_000_000, 1_700_000_000 + n, dtype=np.int64)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    # 强信号：特征0 的滞后项预测未来
    y = pd.Series(np.concatenate([np.zeros(10), X["a"].values[:-10]]),
                  dtype=np.float64)
    ds = FactorDataset(X=X, y=y, ts=ts, symbol=np.full(n, "s1", dtype=object),
                       feature_names=list("abcd"),
                       meta={"horizon": 10, "n_factors": 4, "n_symbols": 1,
                             "n_samples": n})
    # gap=None → 自动取 horizon=10，正常滚动训练
    res = walk_forward_fit_predict(
        ds, model_factory=lambda: make_lgbm_regressor(
            n_estimators=10, learning_rate=0.1),
        step=40, window=120, gap=None, min_train=30)
    assert res.n_folds >= 1


def test_formula_chrom_roundtrip_consistency():
    """公式编解码往返恒等（H9 回归：200 随机种子 0 不一致）。"""
    import random
    from model_core.formula_dsl import (random_chrom, chrom_to_formula,
                                        normalize_chrom)
    bad = 0
    for seed in range(200):
        random.seed(seed)
        c = random_chrom(random)
        f = chrom_to_formula(c)
        if f.to_chrom() != normalize_chrom(c):
            bad += 1
    assert bad == 0, f"编解码不一致 {bad}/200"
