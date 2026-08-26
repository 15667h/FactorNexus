"""
tests/test_p7_factor_backtest.py — P7 终端因子库浏览 + 因子回测单元测试

运行：python -m pytest tests/test_p7_factor_backtest.py -v
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

from scripts.factor_backtest import (
    ascii_equity_curve, backtest_factor, format_backtest_report,
    format_factor_list, format_factor_profile, load_factors,
    resolve_backtest_selector, resolve_selection, run_factor_backtest,
    _rankdata,
)


# ── 回测绩效计算 ────────────────────────────────────────────────────────────

def _trend_close(T=400, seed=1, drift=0.0005):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, T)
    close = 100 * np.cumprod(1 + rets)
    return close


def test_backtest_factor_structure():
    from scripts.factor_backtest import _build_ret_from_close
    close = _trend_close()
    # 因子 = 未来收益标签（与回测同口径）+ 小噪声：同位正相关 → 收益必正
    ret_full = _build_ret_from_close(close, 5)
    noise = np.random.default_rng(7).normal(0, 0.002, len(close))
    factor = ret_full + noise
    b = backtest_factor(factor, close, horizon=5, cost=0.0003)
    for k in ("n", "total_ret", "annual_ret", "sharpe", "sortino", "max_dd",
              "calmar", "profit_factor", "win_rate", "turnover",
              "ic", "icir", "rankic", "nav", "horizon", "cost"):
        assert k in b, f"缺指标 {k}"
    assert b["n"] == len(factor)
    assert len(b["nav"]) == len(factor)
    assert b["horizon"] == 5 and b["cost"] == 0.0003
    assert b["total_ret"] > 0
    assert b["ic"] > 0.5
    assert b["sharpe"] > 0.5


def test_backtest_momentum_structure():
    """动量因子（随机游走下 IC≈0）只验证指标结构，不断言收益符号。"""
    close = _trend_close(seed=2)
    ret5 = close[5:] / close[:-5] - 1.0
    factor = np.concatenate([np.zeros(5), ret5])
    b = backtest_factor(factor, close, horizon=5, cost=0.0)
    assert b["n"] == len(factor)
    assert abs(b["ic"]) < 0.3  # 随机游走增量独立 → 动量无预测力


def test_backtest_alignment_tail():
    """因子短于 K 线时按尾部对齐（入库时因子是 K 线尾部窗口计算的）。"""
    close = _trend_close(T=500)
    factor = np.full(200, 1.0) + np.random.default_rng(0).normal(0, 0.1, 200)
    b = backtest_factor(factor, close, horizon=3, cost=0.0)
    assert b["n"] == 200
    assert len(b["nav"]) == 200


def test_backtest_short_raises():
    with pytest.raises(ValueError, match="过短"):
        backtest_factor(np.ones(10), _trend_close(), horizon=5)


def test_backtest_kline_insufficient():
    with pytest.raises(ValueError, match="K线不足"):
        backtest_factor(np.ones(300), _trend_close(T=100), horizon=5)


def test_rankdata_ties():
    a = np.array([3.0, 1.0, 2.0, 2.0])
    r = _rankdata(a)
    # 并列 2.0 占 rank 2、3 → 平均 2.5
    assert np.allclose(r, [4.0, 1.0, 2.5, 2.5])


# ── 终端渲染 ────────────────────────────────────────────────────────────────

def _fake_factors(n=3):
    out = []
    for i in range(n):
        out.append({
            "symbol": f"sh60000{i}", "hash": f"hash{i:012d}", "kind": "param",
            "engine": "gp", "formula": [i] * 10, "vocab_version": "param-v1",
            "describe": f"测试公式{i}", "ic": 0.05 + i * 0.01,
            "rankic": 0.04, "icir": 0.3, "dsr": 0.90 + i * 0.02,
            "pbo": 0.4, "cpcv": {}, "five_dim": {"total": 0.6, "pps": 0.8,
                                                 "stability": 0.5,
                                                 "robustness": 0.5,
                                                 "logic": 0.4,
                                                 "diversity": 0.9},
            "sharpe": 1.2, "max_dd": 0.15, "turnover": 0.3,
            "n_trials": 100, "mined_at": 1700000000.0, "path": "x.parquet",
        })
    return out


def test_format_list_contains_fields():
    s = format_factor_list(_fake_factors())
    for key in ("symbol", "IC", "五维", "Sharpe", "引擎", "hash"):
        assert key in s
    assert "DSR" not in s.splitlines()[0], "DSR 列已移除（全是 0 无信息量）"
    assert "sh600000" in s and "测试公式" in s


def test_format_profile_contains_fields():
    s = format_factor_profile(_fake_factors(1)[0])
    for key in ("引擎", "公式", "染色体", "DSR", "PBO", "预测力", "五维", "入库时间"):
        assert key in s


def test_ascii_curve():
    nav = np.linspace(1.0, 1.5, 100)
    s = ascii_equity_curve(nav, width=40, height=10)
    lines = s.splitlines()
    assert len(lines) == 10 + 2  # 曲线 10 行 + 轴 + 标签
    assert "1.500" in lines[-1]


# ── 选择解析 ────────────────────────────────────────────────────────────────

def test_resolve_selection_number():
    fs = _fake_factors(3)
    f, err = resolve_selection("2", fs)
    assert err is None and f["symbol"] == "sh600001"


def test_resolve_selection_symbol_hash():
    fs = _fake_factors(3)
    f, err = resolve_selection("sh600000 hash0", fs)
    assert err is None and f["symbol"] == "sh600000"


def test_resolve_selection_invalid():
    fs = _fake_factors(3)
    assert resolve_selection("99", fs)[1] is not None
    assert resolve_selection("sz000001 xxxx", fs)[1] is not None
    assert resolve_selection("abc", fs)[1] is not None


# ── 回测选择器（编号 / 纯 symbol / symbol+hash）─────────────────────────────

def test_backtest_selector_number():
    """推荐用法：--backtest 5 → 列表第 5 个因子。"""
    fs = _fake_factors(5)
    f, err = resolve_backtest_selector("5", None, fs)
    assert err is None and f["symbol"] == "sh600004"


def test_backtest_selector_symbol_unique():
    """纯 symbol：该品种唯一因子直接命中。"""
    fs = [f for f in _fake_factors(3) if f["symbol"] == "sh600000"]
    f, err = resolve_backtest_selector("sh600000", None, fs)
    assert err is None and f["hash"] == "hash000000000000"


def test_backtest_selector_symbol_multiple_lists_numbers():
    """纯 symbol 但该品种多个因子 → 报错并列出编号（不猜）。"""
    fs = _fake_factors(3)  # sh600000/sh600001/sh600002 各一个
    fs.append({**fs[0], "hash": "hash0000000009"})  # sh600000 第二个因子
    f, err = resolve_backtest_selector("sh600000", None, fs)
    assert f is None and err is not None
    assert "编号" in err and "1" in err and "4" in err


def test_backtest_selector_symbol_hash_prefix():
    fs = _fake_factors(3)
    f, err = resolve_backtest_selector("sh600001", "hash0", fs)
    assert err is None and f["symbol"] == "sh600001"
    # 不存在的前缀
    f, err = resolve_backtest_selector("sh600001", "zzz", fs)
    assert f is None and err is not None


def test_backtest_selector_number_out_of_range():
    fs = _fake_factors(3)
    f, err = resolve_backtest_selector("9", None, fs)
    assert f is None and "超出范围" in err


# ── 因子库读取 + 端到端 ─────────────────────────────────────────────────────

def test_load_factors_from_store(tmp_path):
    """临时 store：保存一个因子 → load_factors 读回且字段齐全。"""
    from data_pipeline.store.kline_store import FactorStore

    fs = FactorStore(tmp_path)
    rep = {
        "describe": "测试因子", "ic": 0.08, "rankic": 0.06, "icir": 0.5,
        "dsr": 0.95, "pbo": 0.3, "cpcv": {}, "sharpe": 1.5, "max_dd": 0.1,
        "turnover": 0.2, "n_trials": 50, "mined_at": 1700000000.0,
        "five_dim": {"pps": 0.9, "stability": 0.6, "robustness": 0.7,
                     "logic": 0.5, "diversity": 0.8, "total": 0.7},
        "meta": {"symbol": "sz000001", "engine": "llm"},
    }
    fdf = pd.DataFrame({"factor": np.random.default_rng(0).normal(0, 1, 200)})
    fs.save("sz000001", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "param-v1", fdf,
            report=rep)
    factors = load_factors(tmp_path)
    assert len(factors) == 1
    f = factors[0]
    assert f["symbol"] == "sz000001" and f["engine"] == "llm"
    assert f["dsr"] == 0.95 and f["five_dim"]["total"] == 0.7
    assert f["kind"] == "param"


def test_run_factor_backtest_e2e(tmp_path):
    """端到端：临时 store 存 K线 + 因子 → 回测全链路。"""
    from data_pipeline.store.kline_store import FactorStore, KlineStore

    close = _trend_close(T=400, seed=3)
    df = pd.DataFrame({
        "ts": np.arange(400, dtype=np.int64),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.abs(np.random.default_rng(0).normal(1e6, 2e5, 400)),
    })
    KlineStore(tmp_path).update("sh600000", "1d", df)

    ret5 = close[5:] / close[:-5] - 1.0
    factor = np.concatenate([np.zeros(5), ret5])
    fs = FactorStore(tmp_path)
    rep = {"describe": "动量", "ic": 0.05, "dsr": 0.9, "five_dim": {"total": 0.6},
           "meta": {"symbol": "sh600000", "engine": "gp"}}
    fh = fs.save("sh600000", [1] * 10, "param-v1",
                 pd.DataFrame({"factor": factor}), report=rep)

    factors = load_factors(tmp_path)
    b = run_factor_backtest(factors[0], tmp_path, horizon=5, cost=0.0003)
    assert b["n"] == len(factor)
    assert b["total_ret"] > 0
    report_txt = format_backtest_report(b)
    assert "总收益" in report_txt and "夏普" in report_txt
