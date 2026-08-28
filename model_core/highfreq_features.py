"""
model_core/highfreq_features.py — 高频因子特征（P15，华泰因子工厂 2.0 风格）

从分钟级 K 线聚合构建「日频高频特征」（intraday 特征），每个特征
与日线因子同构（一日一个值），可直接参与认证/入库/组合。

特征清单（全部因果，t 日只用 t 日及以前的分钟数据）：
  hf_open_gap       开盘跳空（open/prev_close - 1）
  hf_intra_ampl     日内振幅（high-low）/open
  hf_intra_vol      日内分钟收益波动率（std）
  hf_intra_skew     日内分钟收益偏度
  hf_intra_kurt     日内分钟收益峰度
  hf_intra_ac1      日内分钟收益一阶自相关（反转/动量微观结构）
  hf_tail_mom       尾盘动量（最后 20% 分钟收益）
  hf_morning_ratio  上午成交量占比（<12:00 量 / 全天量）
  hf_vwap_dev       VWAP 偏离（close/vwap - 1）
  hf_vol_corr       分钟收益与分钟成交量相关
  hf_big_trade      大分钟单占比（量 > 2σ 的分钟数占比）
  hf_ret5m_min      日内 5 分钟最差收益（日内尾部风险）
  hf_ret5m_max      日内 5 分钟最好收益
  hf_range_pos      日内位置（(close-low)/(high-low)）

用法：
    from model_core.highfreq_features import build_highfreq_features
    feats = build_highfreq_features(df_minute)   # {name: np.ndarray[T_days]}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 特征名 -> 实现（供元数据与公式编码引用）
HIGHFREQ_FEATURES = (
    "hf_open_gap", "hf_intra_ampl", "hf_intra_vol", "hf_intra_skew",
    "hf_intra_kurt", "hf_intra_ac1", "hf_tail_mom", "hf_morning_ratio",
    "hf_vwap_dev", "hf_vol_corr", "hf_big_trade", "hf_ret5m_min",
    "hf_ret5m_max", "hf_range_pos",
)
_FEAT_INDEX = {name: i for i, name in enumerate(HIGHFREQ_FEATURES)}


def _day_key(ts: int) -> str:
    """分钟时间戳 -> 自然日（CST 日期串）。"""
    import datetime as dt
    return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone(
        dt.timedelta(hours=8))).strftime("%Y-%m-%d")


def _per_day(ts_arr: np.ndarray, values: np.ndarray, days: list[str],
             agg) -> np.ndarray:
    """按交易日聚合（日期对齐到 days 顺序）。"""
    out = np.full(len(days), np.nan)
    cur_day = None
    buf: list[float] = []
    day_idx = {d: i for i, d in enumerate(days)}
    for t, v in zip(ts_arr, values):
        d = _day_key(int(t))
        if cur_day is None:
            cur_day = d
        if d != cur_day:
            if cur_day in day_idx and buf:
                out[day_idx[cur_day]] = agg(np.array(buf))
            cur_day = d
            buf = []
        if np.isfinite(v):
            buf.append(float(v))
    if cur_day in day_idx and buf:
        out[day_idx[cur_day]] = agg(np.array(buf))
    return out


def build_highfreq_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """分钟 K 线 -> 日频高频特征 {name: np.ndarray[交易日]}。

    输入 df 列: ts/open/high/low/close/volume（分钟 bar，升序）。
    输出按自然日聚合（返回所有交易日，含首日不完整）。
    """
    if df is None or df.empty:
        return {}
    df = df.sort_values("ts").drop_duplicates(subset="ts")
    ts = df["ts"].values.astype(np.int64)
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)
    n = len(ts)
    days = sorted({_day_key(int(t)) for t in ts})
    nd = len(days)
    eps = 1e-9

    def agg_day(values, fn):
        return _per_day(ts, values, days, fn)

    # 分钟收益（含日内首根相对昨收——用前一根处理，首根跳过）
    ret = np.zeros(n)
    ret[1:] = close[1:] / (close[:-1] + eps) - 1.0

    feats: dict[str, np.ndarray] = {}
    # 开盘跳空 = 当日**第一根** bar 的 open/昨收 - 1（M19 修复：旧实现把
    # 全天每分钟 open 相对昨收的偏离取均值，混入盘中价格，非开盘跳空）
    prev_close = None
    gap_vals = np.full(n, np.nan)
    for i in range(n):
        d = _day_key(int(ts[i]))
        first_of_day = (i == 0) or (_day_key(int(ts[i - 1])) != d)
        if i > 0 and _day_key(int(ts[i - 1])) != d:
            prev_close = close[i - 1]
        if first_of_day and prev_close is not None:
            gap_vals[i] = open_[i] / (prev_close + eps) - 1.0
    feats["hf_open_gap"] = agg_day(gap_vals, np.mean)

    # 日内振幅 = (当日最高 - 当日最低) / 当日开盘（M20 修复：旧实现取
    # 每分钟 (H-L)/O 的最大值，是"分钟相对振幅最大值"而非日内振幅）
    day_high = _per_day(ts, high, days, np.max)
    day_low = _per_day(ts, low, days, np.min)
    day_open = _per_day(ts, open_, days,
                        lambda v: float(v[0]) if len(v) else np.nan)
    feats["hf_intra_ampl"] = (day_high - day_low) / (day_open + eps)
    feats["hf_intra_vol"] = agg_day(ret, np.std)
    feats["hf_intra_skew"] = _per_day(
        ts, ret, days,
        lambda v: float(pd.Series(v).skew()) if len(v) > 2 else np.nan)
    feats["hf_intra_kurt"] = _per_day(
        ts, ret, days,
        lambda v: float(pd.Series(v).kurt()) if len(v) > 3 else np.nan)

    # 一阶自相关（日内；1h 周期日 bar 数仅 4，min 4 根可算）
    def _ac1(v: np.ndarray) -> float:
        if len(v) < 4:
            return np.nan
        a = v[:-1] - v[:-1].mean()
        b = v[1:] - v[1:].mean()
        sd = np.sqrt((a ** 2).mean() * (b ** 2).mean())
        return float((a * b).mean() / sd) if sd > 1e-12 else 0.0
    feats["hf_intra_ac1"] = _per_day(ts, ret, days, _ac1)

    # 尾盘动量：日内最后 20% 分钟的累计收益（至少 4 根）
    def _tail(v: np.ndarray) -> float:
        if len(v) < 4:
            return np.nan
        k = max(1, len(v) // 5)
        return float(np.prod(1.0 + v[-k:]) - 1.0)
    feats["hf_tail_mom"] = _per_day(ts, ret, days, _tail)

    # 上午成交量占比（< 12:00 北京时间）
    import datetime as dt
    cst = dt.timezone(dt.timedelta(hours=8))
    am_mask = np.array([
        1 if dt.datetime.fromtimestamp(int(t), tz=cst).hour < 12 else 0
        for t in ts], dtype=float)
    am_vol = _per_day(ts, vol * am_mask, days, np.sum)
    day_vol = _per_day(ts, vol, days, np.sum)
    feats["hf_morning_ratio"] = am_vol / (day_vol + eps)

    # VWAP 偏离：close / Σ(价×量)/Σ量 - 1（用 (h+l+c)/3 近似均价），逐日累计
    typ = (high + low + close) / 3.0
    vwap_dev = np.full(n, np.nan)
    cur_day = None
    sum_pv = 0.0
    sum_v = 0.0
    for i in range(n):
        d = _day_key(int(ts[i]))
        if cur_day is None:
            cur_day = d
        if d != cur_day:
            cur_day = d
            sum_pv, sum_v = 0.0, 0.0
        sum_pv += typ[i] * vol[i]
        sum_v += vol[i]
        if sum_v > eps:
            vwap_dev[i] = close[i] / (sum_pv / sum_v + eps) - 1.0
    feats["hf_vwap_dev"] = agg_day(vwap_dev, np.mean)

    # 分钟收益与成交量相关（日内），逐日计算（1h 周期 4 根也可算）
    def _vol_corr(v_ret, v_vol):
        if len(v_ret) < 4:
            return np.nan
        return float(np.corrcoef(v_ret, v_vol)[0, 1]) \
            if np.std(v_ret) > 1e-12 and np.std(v_vol) > 1e-12 else 0.0
    vol_corr_vals = np.full(n, np.nan)
    cur_day = None
    buf_r: list[float] = []
    buf_v: list[float] = []
    for i in range(n):
        d = _day_key(int(ts[i]))
        if cur_day is None:
            cur_day = d
        if d != cur_day:
            if len(buf_r) >= 4:
                vol_corr_vals[i - 1] = _vol_corr(np.array(buf_r),
                                                 np.array(buf_v))
            cur_day = d
            buf_r, buf_v = [], []
        buf_r.append(ret[i])
        buf_v.append(vol[i])
    # M21 修复：循环结束后 flush 最后一天（旧实现最后一个交易日恒为 NaN）
    if len(buf_r) >= 4 and n > 0:
        vol_corr_vals[n - 1] = _vol_corr(np.array(buf_r), np.array(buf_v))
    feats["hf_vol_corr"] = _per_day(ts, vol_corr_vals, days, np.nanmean)

    # 大分钟单占比：量 > 日内均值+2σ 的分钟占比（至少 4 根）
    def _big(v: np.ndarray) -> float:
        if len(v) < 4:
            return np.nan
        m, s = float(np.mean(v)), float(np.std(v))
        return float(np.mean(v > m + 2 * s)) if s > 1e-12 else 0.0
    feats["hf_big_trade"] = _per_day(ts, vol, days, _big)

    # 日内 5 分钟最差/最好收益（用累计 5 根窗口；1h 周期降为 2 根窗口）
    def _worst(v: np.ndarray) -> float:
        if len(v) < 4:
            return np.nan
        c = np.cumprod(1.0 + v)
        c = np.concatenate([[1.0], c])
        k = min(5, max(2, len(v) // 2))
        w = np.min(c[k:] / c[:-k] - 1.0)
        return float(w)
    def _best(v: np.ndarray) -> float:
        if len(v) < 4:
            return np.nan
        c = np.cumprod(1.0 + v)
        c = np.concatenate([[1.0], c])
        k = min(5, max(2, len(v) // 2))
        return float(np.max(c[k:] / c[:-k] - 1.0))
    feats["hf_ret5m_min"] = _per_day(ts, ret, days, _worst)
    feats["hf_ret5m_max"] = _per_day(ts, ret, days, _best)

    # 日内位置
    feats["hf_range_pos"] = agg_day(
        (close - low) / (high - low + eps), np.mean)

    # 统一长度
    for name in HIGHFREQ_FEATURES:
        v = feats.get(name)
        if v is None:
            feats[name] = np.full(nd, np.nan)
        elif len(v) != nd:
            feats[name] = np.full(nd, np.nan)
    return feats


def highfreq_feature_names() -> list[str]:
    return list(HIGHFREQ_FEATURES)
