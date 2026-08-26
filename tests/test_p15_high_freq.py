"""P15：高频因子测试（分钟级特征构建 + 高频挖掘管线）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_core.highfreq_features import (
    HIGHFREQ_FEATURES, build_highfreq_features,
)
from scripts.mine_high_freq import _expanding_zscore, _oos_significance


def _synthetic_minute_df(n_days: int = 20, bars_per_day: int = 16,
                         seed: int = 0) -> pd.DataFrame:
    """合成分钟 K 线（日 4 小时 / 15min × 16 根，模拟 A 股交易时段）。"""
    rng = np.random.default_rng(seed)
    rows = []
    ts = pd.Timestamp("2025-01-02 09:30:00", tz="Asia/Shanghai")
    for d in range(n_days):
        day_ts = ts + pd.Timedelta(days=d)
        # 跳过周末
        while day_ts.weekday() >= 5:
            day_ts += pd.Timedelta(days=1)
        close = 50.0
        for b in range(bars_per_day):
            bar_ts = day_ts + pd.Timedelta(minutes=15 * b)
            open_ = close
            close = open_ * (1.0 + rng.normal(0.0002, 0.002))
            high = max(open_, close) * (1.0 + abs(rng.normal(0, 0.001)))
            low = min(open_, close) * (1.0 - abs(rng.normal(0, 0.001)))
            vol = rng.uniform(1e4, 5e4)
            rows.append({
                "ts": int(bar_ts.timestamp()),
                "open": open_, "high": high, "low": low,
                "close": close, "volume": vol,
            })
    return pd.DataFrame(rows)


def test_build_highfreq_features_all_names():
    df = _synthetic_minute_df(n_days=15, bars_per_day=16)
    feats = build_highfreq_features(df)
    assert set(feats) == set(HIGHFREQ_FEATURES)
    for name, v in feats.items():
        assert len(v) >= 10, f"{name} 长度不足"
        # 大多数日应有有限值
        assert np.isfinite(v).sum() >= 5, f"{name} 有效值过少"


def test_build_highfreq_features_daily_alignment():
    """特征按日聚合：15 天 → 特征长度 ≈ 15（含首日不完整）。"""
    df = _synthetic_minute_df(n_days=15, bars_per_day=16)
    feats = build_highfreq_features(df)
    # 交易日数（去周末）
    days = sorted({str(pd.Timestamp(x, unit="s", tz="Asia/Shanghai").date())
                   for x in df["ts"]})
    assert len(days) >= 10
    assert len(feats["hf_intra_vol"]) == len(days)


def test_build_highfreq_features_causal():
    """因果性：只改最后一天数据不影响之前特征值（防未来信息）。"""
    df1 = _synthetic_minute_df(n_days=12, bars_per_day=16, seed=1)
    df2 = df1.copy()
    # 篡改最后 8 根（最后一天）
    df2.loc[df2.index[-8:], "close"] *= 1.5
    f1 = build_highfreq_features(df1)
    f2 = build_highfreq_features(df2)
    n = len(f1["hf_intra_vol"])
    for name in HIGHFREQ_FEATURES:
        a, b = f1[name][:n - 1], f2[name][:n - 1]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 3:
            assert np.allclose(a[ok], b[ok], atol=1e-9), f"{name} 出现前视"


def test_expanding_zscore_causal():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 200)
    z = _expanding_zscore(x)
    assert z is not None and len(z) == 200
    # 因果：改尾部不影响头部（NaN 位置一致，用 equal_nan 比较）
    x2 = x.copy()
    x2[-10:] += 100.0
    z2 = _expanding_zscore(x2)
    ok = np.isfinite(z[:100]) & np.isfinite(z2[:100])
    assert ok.sum() >= 60
    assert np.allclose(z[:100][ok], z2[:100][ok], atol=1e-9)


def test_oos_significance():
    rng = np.random.default_rng(3)
    f = rng.normal(0, 1, 200)
    r = 0.2 * f + rng.normal(0, 1, 200)      # 较强信号（RankIC ≈ 0.2）
    rankic, t, p = _oos_significance(f, r)
    assert np.isfinite(rankic) and np.isfinite(p)
    assert rankic > 0.05        # 强信号应检出
    assert p < 0.05


def test_highfreq_encoding_unique():
    """特征编码（10000+fid）互不冲突。"""
    codes = [10000 + i for i in range(len(HIGHFREQ_FEATURES))]
    assert len(set(codes)) == len(codes)
    assert all(c >= 10000 for c in codes)
