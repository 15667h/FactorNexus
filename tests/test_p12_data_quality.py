"""P12：数据质量层（quality.py / kline_store.py 增强）测试。

覆盖：OHLC 一致性、粘滞价格、未来时间戳、跳变分类（混库 vs 单向）、
clean_series 的 OHLC 修复与未来戳剔除、KlineStore 冲突检测增强（异常比例法）、
来源元数据 kline_sources.json、audit_all 全库健康审计。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.quality import (
    check_ohlc_consistency,
    check_series,
    classify_jumps,
    clean_series,
    count_future_timestamps,
    count_zero_volume_trading,
    detect_sticky_prices,
    mask_jump_returns,
)
from data_pipeline.store.kline_store import KlineStore


def _df(n: int = 120, start_ts: int = 1_600_000_000) -> pd.DataFrame:
    """合成健康日线：模拟 A 股日收益（±9.9% 涨跌幅约束，价格下限 10 元）。"""
    rng = np.random.default_rng(0)
    ret = np.clip(rng.normal(0.001, 0.01, n), -0.099, 0.099)
    close = 100.0 * np.cumprod(1.0 + ret)
    close = np.maximum(close, 10.0)
    ts = np.arange(start_ts, start_ts + n, dtype=np.int64)
    return pd.DataFrame({
        "ts": ts,
        "open": close - 0.2, "high": close + 1.0,
        "low": close - 1.0, "close": close,
        "volume": rng.uniform(1e6, 5e6, n),
    })


# ── OHLC 一致性 ───────────────────────────────────────────────────────────

def test_check_ohlc_healthy():
    assert check_ohlc_consistency(_df()) == []


def test_check_ohlc_detects_inverted():
    df = _df()
    df.loc[5, "high"] = df.loc[5, "close"] - 5.0   # high < close
    df.loc[6, "low"] = df.loc[6, "close"] + 5.0    # low > close
    issues = check_ohlc_consistency(df)
    assert any("high<max" in s for s in issues)
    assert any("low>min" in s for s in issues)


def test_clean_series_fixes_ohlc():
    df = _df()
    df.loc[5, "high"] = df.loc[5, "close"] - 5.0
    df.loc[6, "low"] = df.loc[6, "close"] + 5.0
    cleaned, _ = clean_series(df)
    assert cleaned["high"].iloc[5] >= cleaned["close"].iloc[5]
    assert cleaned["low"].iloc[6] <= cleaned["close"].iloc[6]


# ── 粘滞价格 ──────────────────────────────────────────────────────────────

def test_detect_sticky_prices():
    df = _df()
    c = df["close"].values.astype(float)
    c[30:42] = c[30]                       # 12 日完全粘滞
    df["close"] = c
    total, runs = detect_sticky_prices(df)
    assert total >= 12
    assert len(runs) >= 1


def test_check_series_reports_sticky():
    df = _df()
    df.loc[30:39, "close"] = df.loc[30, "close"]
    assert any("粘滞" in s for s in check_series(df))


# ── 未来时间戳 / 成交量异常 ───────────────────────────────────────────────

def test_count_future_timestamps():
    df = _df()
    df.loc[0, "ts"] = int(time.time()) + 10 * 86400
    assert count_future_timestamps(df) == 1
    cleaned, _ = clean_series(df)
    assert len(cleaned) == len(df) - 1


def test_zero_volume_trading():
    df = _df()
    df.loc[10:50, "volume"] = 0.0          # 41 日价格变动但量 0
    assert count_zero_volume_trading(df) >= 1
    assert any("成交量" in s for s in check_series(df))


# ── 跳变分类：混库 vs 单向 ────────────────────────────────────────────────

def test_classify_jumps_clean():
    kind, info = classify_jumps(_df())
    assert kind == "clean"
    assert info["n"] == 0


def test_classify_jumps_mix_alternating():
    """交替涨跌大跳变（qfq/不复权交替）→ mix。"""
    df = _df()
    c = df["close"].values.astype(float)
    # 交替：×6.5 → ÷6.5 → ×6.5 → ÷6.5（方向交替占比 100%）
    for i, scale in zip(range(50, 58), [6.5, 1 / 6.5, 6.5, 1 / 6.5,
                                        6.5, 1 / 6.5, 6.5, 1 / 6.5]):
        c[i] = c[i - 1] * scale
    df["close"] = c
    kind, info = classify_jumps(df)
    assert kind == "mix"
    assert info["n"] >= 7
    assert info["alternations"] >= 6


def test_classify_jumps_one_way_single():
    """单次大跳变（除权/单点瑕疵）→ one_way（单点异常产生 2 次相邻反向跳变）。"""
    df = _df()
    df.loc[60, "close"] = df.loc[60, "close"] * 2.5   # 单点 +150%
    kind, info = classify_jumps(df)
    assert kind == "one_way"
    assert info["n"] == 2


# ── mask_jump_returns ─────────────────────────────────────────────────────

def test_mask_jump_returns_zeroes_jump_days():
    df = _df()
    df.loc[60, "close"] = df.loc[60, "close"] * 2.5
    _, jumps = clean_series(df)
    ret = np.ones(len(df))
    masked = mask_jump_returns(ret, df["ts"].values, jumps)
    # 单点异常产生两个跳变日：跳变发生日(+150%) 与回归日(-60%) 均置 0
    assert masked[60] == 0.0
    assert masked[61] == 0.0
    assert masked[62] == 1.0              # 非跳变日不受影响


# ── KlineStore：冲突检测增强（异常比例法）─────────────────────────────────

def test_update_conflict_detected_by_bad_ratio(tmp_path):
    """公共日期价格比 >2x/<0.5x 占比 >10% → 整体覆盖（即使中位数看似正常）。"""
    store = KlineStore(tmp_path)
    n = 100
    ts = np.arange(1_600_000_000, 1_600_000_000 + n, dtype=np.int64)
    base = 10.0 + np.arange(n) * 0.05
    df1 = pd.DataFrame({"ts": ts, "open": base, "high": base + 1,
                        "low": base - 1, "close": base, "volume": 1e6})
    store.update("sh600000", "1d", df1, source="sina", adjust="raw")
    # 混入：前 40 天 ×5（异常比例 40%>10%），后 60 天同口径
    base2 = base.copy()
    base2[:40] *= 5.0
    df2 = pd.DataFrame({"ts": ts, "open": base2, "high": base2 + 1,
                        "low": base2 - 1, "close": base2, "volume": 1e6})
    merged = store.update("sh600000", "1d", df2, source="tencent", adjust="qfq")
    # 整体覆盖 → 库 = df2（前 40 天价格 ×5）
    assert len(merged) == n
    assert abs(merged["close"].iloc[0] - base2[0]) < 1e-6
    # 元数据记录来源与冲突标记
    info = store.source_info("sh600000", "1d")
    assert info is not None
    assert info["source"] == "tencent"
    assert info["adjust"] == "qfq"
    assert info["conflict_overwrite"] is True


def test_update_merges_when_consistent(tmp_path):
    store = KlineStore(tmp_path)
    n = 100
    ts = np.arange(1_600_000_000, 1_600_000_000 + n, dtype=np.int64)
    base = 10.0 + np.arange(n) * 0.05
    df1 = pd.DataFrame({"ts": ts, "open": base, "high": base + 1,
                        "low": base - 1, "close": base, "volume": 1e6})
    store.update("sh600000", "1d", df1)
    # 追加 50 根（同口径，小波动）
    ts2 = np.arange(ts[-1] + 1, ts[-1] + 51, dtype=np.int64)
    base2 = base[-1] + 0.05 * np.arange(1, 51)
    df2 = pd.DataFrame({"ts": ts2, "open": base2, "high": base2 + 1,
                        "low": base2 - 1, "close": base2, "volume": 1e6})
    merged = store.update("sh600000", "1d", df2)
    assert len(merged) == n + 50


def test_audit_all_reports_pollution(tmp_path):
    store = KlineStore(tmp_path)
    n = 100
    ts = np.arange(1_600_000_000, 1_600_000_000 + n, dtype=np.int64)
    base = 10.0 + np.arange(n) * 0.05
    good = pd.DataFrame({"ts": ts, "open": base, "high": base + 1,
                         "low": base - 1, "close": base, "volume": 1e6})
    store.update("sh600000", "1d", good, source="tencent", adjust="qfq")
    # 混库标的：交替跳变
    base_mix = base.copy()
    for i, s in zip(range(10, 18), [5.0, 1 / 5.0] * 4):
        base_mix[i] = base_mix[i - 1] * s
    mix = pd.DataFrame({"ts": ts, "open": base_mix, "high": base_mix + 1,
                        "low": base_mix - 1, "close": base_mix, "volume": 1e6})
    store.update("sz000001", "1d", mix, source="sina", adjust="raw")
    audit = store.audit_all()
    assert audit["total"] == 2
    assert audit["dirty"] >= 1
    by_code = {a["code"]: a for a in audit["polluted"]}
    assert by_code["sz000001"]["jump_kind"] == "mix"
    assert by_code["sz000001"]["source"] == "sina"
    assert audit["by_source"].get("tencent") == 1


def test_audit_kline_clean(tmp_path):
    store = KlineStore(tmp_path)
    n = 100
    ts = np.arange(1_600_000_000, 1_600_000_000 + n, dtype=np.int64)
    base = 10.0 + np.arange(n) * 0.05
    good = pd.DataFrame({"ts": ts, "open": base, "high": base + 1,
                         "low": base - 1, "close": base, "volume": 1e6})
    store.update("sh600000", "1d", good, source="tencent", adjust="qfq")
    a = store.audit_kline("sh600000", "1d")
    assert a["clean"] is True
    assert a["source"] == "tencent"


import time  # noqa: E402  （用于未来时间戳测试，放末尾避免前置顺序困扰）
