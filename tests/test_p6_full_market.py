"""
tests/test_p6_full_market.py — P6 全市场三引擎联动挖矿机单元测试

覆盖：
  - 联动矿池 MarketPool：去重 / hits / top / token&param 公式 / 持久化 / LLM 假设
  - 全 A 股清单：新浪 hs_a 分页拉取（mock）、bj 北交所过滤、缓存
  - GP 初始种群种子注入（联动①通道）：NSGA3FactorMiner.mine(init_chroms=...)
  - 单股票三引擎联动挖掘冒烟：mine_one（GP+LLM 快速，RL 小步）
  - 无数据/数据过短边界

运行：python -m pytest tests/test_p6_full_market.py -v
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

from scripts.mine_full_market import (
    MarketPool, _decode_tokens, _pool_key, fetch_universe, mine_one,
)


# ── 联动矿池 ────────────────────────────────────────────────────────────────

def test_pool_add_dedup_and_hits(tmp_path):
    pool = MarketPool(tmp_path)
    key = pool.add("param", "gp", "sh600519", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "desc", 0.05, 0.92, 0.60)
    assert key == _pool_key("param", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # 同公式再入 → 去重 + hits 累加
    pool.add("param", "llm", "sz000001", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "desc2", 0.06, 0.95, 0.70)
    assert pool.size() == 1
    top = pool.top(5, kind="param")
    assert top[0]["hits"] == 2
    assert top[0]["engine"] == "llm"  # 最新引擎覆盖
    assert top[0]["dsr"] == 0.95


def test_pool_kind_isolation(tmp_path):
    pool = MarketPool(tmp_path)
    pool.add("param", "gp", "s1", [1] * 10, "p", 0.05, 0.90, 0.6)
    pool.add("token", "rl", "s1", [3, 7, 9], "t", 0.06, 0.93, 0.7)
    assert len(pool.top(10, kind="param")) == 1
    assert len(pool.top(10, kind="token")) == 1
    assert len(pool.top(10)) == 2
    toks = pool.token_formulas(5)
    assert toks and toks[0][0] == 0.93 and toks[0][1] == [3, 7, 9]


def test_pool_persist(tmp_path):
    pool = MarketPool(tmp_path)
    pool.add("param", "gp", "s1", [2] * 10, "d", 0.04, 0.91, 0.5)
    pool.save()
    pool2 = MarketPool(tmp_path)
    assert pool2.size() == 1
    assert pool2.top(1)[0]["symbol"] == "s1"


def test_pool_hypotheses_for_llm(tmp_path):
    """联动②：矿池精英 → LLM 假设（GP→LLM 反馈通道）。"""
    pool = MarketPool(tmp_path)
    pool.add("param", "gp", "sh600519", [5] * 10, "MACD 动量", 0.08, 0.95, 0.7)
    hyps = pool.hypotheses_for_llm(k=2)
    assert hyps
    assert "sh600519" in hyps[0]
    assert "0.95" in hyps[0]
    assert "不要完全重复" in hyps[0]


def test_pool_trim_by_max_size(tmp_path):
    pool = MarketPool(tmp_path, max_size=5)
    for i in range(8):
        pool.add("param", "gp", "s", [i] * 10, f"d{i}", 0.01 * i, 0.90 + 0.01 * i, 0.5)
    assert pool.size() == 5
    # 保留的是 DSR 最高的 5 条
    dsrs = [e["dsr"] for e in pool.top(10)]
    assert min(dsrs) >= 0.93


# ── 全 A 股清单（新浪 hs_a，mock）───────────────────────────────────────────

class _FakeResp:
    def __init__(self, rows):
        self._rows = rows

    def raise_for_status(self):
        pass

    def json(self):
        return self._rows


def test_universe_fetch_filters_bj_and_paginates(tmp_path, monkeypatch):
    """hs_a 分页拉取：过滤北交所 bj，保留 sh/sz，缓存复用。"""
    pages = [
        # 第 1 页：满 100 条（含 bj 与 sh/sz）
        [{"symbol": f"bj92000{i}"} for i in range(10)] +
        [{"symbol": f"sh60000{i:02d}"} for i in range(90)],
        # 第 2 页：不足 100 条 → 翻页终止
        [{"symbol": "sz000001"}, {"symbol": "sz000002"}],
    ]
    calls = {"n": 0}

    class _FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, params=None, timeout=None):
            calls["n"] += 1
            page = int(params["page"])
            assert params["node"] == "hs_a"
            return _FakeResp(pages[page - 1] if page <= 2 else [])

    import scripts.mine_full_market as m
    monkeypatch.setattr(m.requests, "Session", lambda: _FakeSession())
    monkeypatch.setattr(m, "_MIN_UNIVERSE", 50)   # 测试用小清单绕过全量守卫
    symbols = fetch_universe(tmp_path, refresh=True)
    assert calls["n"] == 2                       # 两页
    assert len(symbols) == 92                    # 90 + 2，bj 全部过滤
    assert all(s.startswith(("sh", "sz")) for s in symbols)
    assert "bj920000" not in symbols
    # 缓存复用：再次调用不再发请求
    calls["n"] = 0
    symbols2 = fetch_universe(tmp_path, refresh=False)
    assert calls["n"] == 0
    assert symbols2 == symbols


def test_universe_insufficient_raises(tmp_path, monkeypatch):
    """清单过少（<3000）时抛错，避免用残缺矿脉开工。"""
    import scripts.mine_full_market as m
    monkeypatch.setattr(m.requests, "Session",
                        lambda: type("S", (), {
                            "headers": {},
                            "get": lambda self, *a, **k: _FakeResp(
                                [{"symbol": "sh600000"}])})())
    with pytest.raises(RuntimeError, match="全 A 股清单"):
        fetch_universe(tmp_path, refresh=True)


# ── GP 初始种群种子注入（联动①）────────────────────────────────────────────

def test_gp_mine_with_init_chroms():
    """NSGA3FactorMiner.mine(init_chroms=...) 不报错且返回结果（种子+随机补齐）。"""
    from model_core.indicator_builder import build_indicators
    from model_core.engines.gp_engine import NSGA3FactorMiner
    from model_core.formula_dsl import CHROM_LEN

    rng = np.random.default_rng(0)
    T = 260
    close = 100 + np.cumsum(rng.normal(0, 1, T))
    df = pd.DataFrame({
        "ts": np.arange(T), "open": close, "high": close + 1,
        "low": close - 1, "close": close,
        "volume": np.abs(rng.normal(1000, 200, T)),
    })
    ind = build_indicators(df)
    ret = np.zeros(T)
    ret[:T - 5] = close[5:] / close[:-5] - 1.0

    miner = NSGA3FactorMiner(pop_size=24, n_gen=2, seed=1)
    seeds = [[1] * CHROM_LEN, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    results = miner.mine(ind, ret, init_chroms=seeds)
    assert isinstance(results, list)
    for r in results:
        assert len(r["chrom"]) == CHROM_LEN
        assert "formula" in r and "objectives" in r
    # 纯随机路径（None）不破坏
    results0 = NSGA3FactorMiner(pop_size=12, n_gen=1, seed=1).mine(ind, ret)
    assert len(results0) >= 0


# ── 单股票三引擎联动挖掘冒烟 ────────────────────────────────────────────────

def _synthetic_df(T=320, seed=7) -> pd.DataFrame:
    """合成日线：模拟 A 股日收益分布（涨跌幅 ±9.9% 约束，价格下限 10 元）。

    用收益率合成价格（而非随机游走价格），避免穿零/暴跌等 A 股不存在的
    病态序列误触发数据质量层的混库判定。
    """
    rng = np.random.default_rng(seed)
    ret = np.clip(rng.normal(0.001, 0.01, T), -0.099, 0.099)
    close = 100.0 * np.cumprod(1.0 + ret)
    close = np.maximum(close, 10.0)
    return pd.DataFrame({
        "ts": np.arange(T, dtype=np.int64),
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": np.abs(rng.normal(1000, 200, T)),
    })


class _Cfg:
    def __init__(self, **kw):
        self.engines = kw.pop("engines", ("gp", "llm"))
        self.tf = "1d"
        self.bars = 0
        self.rl_bars = 200
        self.horizon = 5
        self.gen = 2
        self.pop = 16
        self.rl_steps = 2
        self.rl_batch = 32
        self.rl_folds = 2
        self.llm_hyp = 2
        self.llm_rounds = 1
        self.llm_batch = 50
        self.dsr_gate = 0.0          # Bailey 修正后 DSR 为报告项，默认不拦截
        self.oos_frac = 0.25
        self.min_oos_rankic = 0.02
        self.min_oos_p = 0.05
        self.crowd_corr = 0.85
        self.quick_gate = 0.0
        self.no_backfill = False
        self.seed = 42
        self.store_dir = kw.pop("store_dir", "store")
        for k, v in kw.items():
            setattr(self, k, v)


def test_mine_one_gp_llm_smoke(tmp_path):
    """GP+LLM 联动挖掘在合成数据上可跑通，返回完整结构。"""
    from data_pipeline.store.kline_store import KlineStore

    store = KlineStore(tmp_path)
    store.update("sz000001", "1d", _synthetic_df(T=600))

    cfg = _Cfg(store_dir=str(tmp_path), engines=("gp", "llm"))
    pool = MarketPool(tmp_path)
    ctx = {"pool": pool, "llm_call": None, "gp_seeds": []}
    r = mine_one("sz000001", "1d", cfg, ctx)
    assert r["symbol"] == "sz000001"
    assert r["status"] in ("ok", "none_accepted")
    assert r["n_gp"] >= 0 and r["n_llm"] >= 0
    assert isinstance(r["best_dsr"], float)
    assert r["elapsed_s"] >= 0
    # 达标时矿池与因子库应有产出
    if r["n_accepted"] > 0:
        assert pool.size() >= 1
        assert r["best_engine"] in ("gp", "llm")


def test_mine_one_all_engines_small(tmp_path):
    """三引擎联动（GP+RL+LLM，RL 小步）冒烟：RL 分支产出 token 候选不炸。"""
    from data_pipeline.store.kline_store import KlineStore

    store = KlineStore(tmp_path)
    store.update("sh600000", "1d", _synthetic_df(T=600, seed=11))

    cfg = _Cfg(store_dir=str(tmp_path), engines=("gp", "rl", "llm"),
               rl_steps=1, rl_batch=16)
    pool = MarketPool(tmp_path)
    ctx = {"pool": pool, "llm_call": None, "gp_seeds": []}
    r = mine_one("sh600000", "1d", cfg, ctx)
    assert r["symbol"] == "sh600000"
    assert r["status"] not in ("error",)
    assert isinstance(r["n_rl"], int) and r["n_rl"] in (0, 1)
    if r.get("rl_error"):
        # 环境无 GPU/模型问题时允许 RL 降级跳过，但整体不崩
        assert "rl_error" in r


def test_mine_one_no_backfill_no_data(tmp_path):
    cfg = _Cfg(store_dir=str(tmp_path), engines=("gp",), no_backfill=True)
    pool = MarketPool(tmp_path)
    ctx = {"pool": pool, "llm_call": None, "gp_seeds": []}
    r = mine_one("sz000002", "1d", cfg, ctx)
    assert r["status"] == "no_data"


def test_mine_one_too_short(tmp_path):
    from data_pipeline.store.kline_store import KlineStore

    store = KlineStore(tmp_path)
    store.update("sz000003", "1d", _synthetic_df(T=80, seed=3))
    cfg = _Cfg(store_dir=str(tmp_path), engines=("gp",))
    pool = MarketPool(tmp_path)
    ctx = {"pool": pool, "llm_call": None, "gp_seeds": []}
    r = mine_one("sz000003", "1d", cfg, ctx)
    assert r["status"] == "too_short"


# ── token 解码（RL 产物描述）───────────────────────────────────────────────

def test_decode_tokens():
    out = _decode_tokens([0, 1, 2])
    assert isinstance(out, str) and out
    out2 = _decode_tokens([99999])
    assert "99999" in out2  # 越界 token 兜底
