"""P16+P17：组合层流水线 + 因子监控测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.portfolio_pipeline import build_panels, run_pipeline
from model_core.eval.factor_monitor import (
    compute_decay_track, monitor_all, monitor_factor,
)


# ── P16 组合层流水线 ──────────────────────────────────────────────────────

def _seed_store(tmp_path, n_factors: int = 6, n_days: int = 300) -> None:
    """写入合成因子库 + K 线（多标的，含真实信号）。"""
    from data_pipeline.store.kline_store import FactorStore, KlineStore
    rng = np.random.default_rng(0)
    store = FactorStore(tmp_path)
    kstore = KlineStore(tmp_path)
    ts = pd.date_range("2024-01-01", periods=n_days, freq="B")
    ts_i = ts.astype("int64") // 10 ** 9
    syms = [f"sz{i:06d}" for i in range(n_factors)]
    for i, sym in enumerate(syms):
        close = 30 + rng.normal(0.04, 0.5, n_days).cumsum() + i * 0.1
        close = np.maximum(close, 5)
        kdf = pd.DataFrame({"ts": ts_i, "open": close - 0.1,
                            "high": close + 0.8, "low": close - 0.8,
                            "close": close,
                            "volume": rng.uniform(1e6, 5e6, n_days)})
        kstore.update(sym, "1d", kdf)
        # 因子 = 未来收益信号 + 噪声（有预测力）
        ret = np.zeros(n_days)
        ret[:-5] = close[5:] / close[:-5] - 1.0
        signal = 2.0 * ret + rng.normal(0, 0.5, n_days)
        signal -= signal.mean()
        fdf = pd.DataFrame({"ts": ts_i, "factor": signal})
        meta = {"engine": "gp", "kind": "param", "symbol": sym,
                "cert_rankic": 0.05, "direction": 1.0}
        store.save(sym, [1, 2, 3], "test_v1", fdf, report=meta)


def test_build_panels(tmp_path):
    _seed_store(tmp_path)
    score, ret_panel, klines, per_stock = build_panels(str(tmp_path))
    assert score.shape[1] == 6                     # 6 只股票
    assert ret_panel is not None and not ret_panel.empty
    assert len(klines) == 6
    assert len(per_stock) == 6
    assert all(v["factors"] == 1 for v in per_stock.values())


def test_run_pipeline_smoke(tmp_path):
    _seed_store(tmp_path)
    class Args:
        n_top = 3
        long_only = False
        weight = "equal"
        window = 40
        ml = False
        ml_window = 60
        ml_trees = 20
        horizon = 5
        bars = 0
        cost = 0.0003
        no_limit_filter = False
        industry = False
        report = ""
        store_dir = str(tmp_path)
    r = run_pipeline(Args())
    assert r["factors"] == 6
    assert r["stocks"] == 6
    assert "total_ret" in r["backtest"]
    assert np.isfinite(r["backtest"]["sharpe"])
    assert "excess_ret" in r["performance"]


def test_run_pipeline_long_only(tmp_path):
    _seed_store(tmp_path)
    class Args:
        n_top = 3
        long_only = True
        weight = "score"
        window = 40
        ml = False
        ml_window = 60
        ml_trees = 20
        horizon = 5
        bars = 0
        cost = 0.0003
        no_limit_filter = False
        industry = False
        report = ""
        store_dir = str(tmp_path)
    r = run_pipeline(Args())
    assert "total_ret" in r["backtest"]


# ── P17 因子监控 ──────────────────────────────────────────────────────────

def test_compute_decay_track():
    rng = np.random.default_rng(1)
    f = rng.normal(0, 1, 400)
    r = 0.1 * f + rng.normal(0, 1, 400)     # 恒强信号
    track = compute_decay_track(f, r, seg=100)
    assert len(track) == 4                  # 400 / 100
    assert all(np.isfinite(x) for x in track)
    assert all(abs(x) > 0.02 for x in track)  # 信号稳定存在


def test_monitor_factor_healthy(tmp_path):
    _seed_store(tmp_path, n_factors=1)
    d = monitor_factor("sz000000", "xxxx", str(tmp_path))
    # hash 不存在 → error（先取真实 hash）
    from data_pipeline.store.kline_store import FactorStore
    f = FactorStore(tmp_path).list_factors()[0]
    d = monitor_factor(f["symbol"], f["hash"], str(tmp_path), recent=100)
    assert d["status"] in ("healthy", "warning")
    assert d["bars"] >= 60
    assert "decay_track" in d


def test_monitor_factor_detects_flip(tmp_path):
    """构造方向翻转因子：实时段符号与入库方向相反 → warning。"""
    from data_pipeline.store.kline_store import FactorStore, KlineStore
    rng = np.random.default_rng(2)
    store = FactorStore(tmp_path)
    kstore = KlineStore(tmp_path)
    n = 300
    ts = pd.date_range("2024-01-01", periods=n, freq="B")
    ts_i = ts.astype("int64") // 10 ** 9
    close = 30 + rng.normal(0.04, 0.5, n).cumsum()
    close = np.maximum(close, 5)
    kstore.update("sz000001", "1d", pd.DataFrame(
        {"ts": ts_i, "open": close - 0.1, "high": close + 0.8,
         "low": close - 0.8, "close": close,
         "volume": rng.uniform(1e6, 5e6, n)}))
    # 因子：前 200 根正相关，后 100 根强负相关（方向翻转）
    ret = np.zeros(n)
    ret[:-5] = close[5:] / close[:-5] - 1.0
    f = np.zeros(n)
    f[:200] = 2.0 * ret[:200] + rng.normal(0, 0.3, 200)
    f[200:] = -3.0 * ret[200:] + rng.normal(0, 0.3, 100)
    f -= f.mean()
    fdf = pd.DataFrame({"ts": ts_i, "factor": f})
    meta = {"engine": "gp", "kind": "param", "symbol": "sz000001",
            "cert_rankic": 0.08, "direction": 1.0}
    h = store.save("sz000001", [1, 2, 3], "test_v1", fdf, report=meta)
    d = monitor_factor("sz000001", h, str(tmp_path), recent=100)
    assert d["status"] == "warning"
    assert any("方向翻转" in a for a in d["alerts"])


def test_monitor_all_structure(tmp_path):
    _seed_store(tmp_path, n_factors=4)
    rep = monitor_all(str(tmp_path), recent=100)
    assert rep["total"] == 4
    assert rep["healthy"] + rep["warning"] + rep["stale"] == 4
    assert "by_engine" in rep
    assert rep["by_engine"]["gp"]["total"] == 4
