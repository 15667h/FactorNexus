"""
model_core/indicator_builder.py — 从 K 线 OHLCV 构造 ParamVM 指标库（P1.2）

对应 formula_dsl.INDICATORS 中定义的指标名，全部因果计算（t 只用 t 及以前数据）。
输入：df（列 ts/open/high/low/close/volume，可选 amount/oi），输出 {name: np.ndarray[T]}。
所有窗口指标返回与输入等长的序列（warm-up 期用 0 或中性值填充）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _roll(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.reshape(-1, 1)
    return np.lib.stride_tricks.sliding_window_view(x, w)


def _roll_full(x: np.ndarray, w: int, fill: float = 0.0, agg="mean") -> np.ndarray:
    """滑动窗口统计并补 warm-up 期（返回与 x 等长的一维序列）。

    agg: "mean"|"std"|"min"|"max"|"sum"（默认 mean）
    """
    if len(x) < w:
        return np.full(len(x), fill)
    wnd = _roll(x, w)  # [T-w+1, w]
    if agg == "mean":
        r = wnd.mean(axis=1)
    elif agg == "std":
        r = wnd.std(axis=1)
    elif agg == "min":
        r = wnd.min(axis=1)
    elif agg == "max":
        r = wnd.max(axis=1)
    elif agg == "sum":
        r = wnd.sum(axis=1)
    else:
        raise ValueError(f"未知聚合 {agg}")
    return np.concatenate([np.full(w - 1, fill), r])


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """因果 EMA（exact 递推）。"""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
    return out


def _linear_slope(x: np.ndarray, w: int) -> np.ndarray:
    """w 期线性回归斜率（因果，t 只用 t 及以前；warm-up 置 0）。"""
    T = len(x)
    if T < w or w < 2:
        return np.zeros(T)
    tidx = np.arange(w, dtype=np.float64)
    tx = tidx - tidx.mean()
    denom = (tx ** 2).sum() + 1e-9
    wnd = _roll(x, w)                       # [T-w+1, w]
    xm = wnd.mean(axis=1, keepdims=True)
    slopes = ((wnd - xm) * tx).sum(axis=1) / denom
    out = np.zeros(T)
    out[w - 1:] = slopes
    return out


def _rsi(close: np.ndarray, w: int = 14) -> np.ndarray:
    diff = np.diff(close, prepend=close[0])
    gains = np.maximum(diff, 0.0)
    losses = np.maximum(-diff, 0.0)
    ag = _roll_full(gains, w)
    al = _roll_full(losses, w)
    rs = (ag + 1e-9) / (al + 1e-9)
    return (100.0 - 100.0 / (1.0 + rs)) / 50.0 - 1.0  # 归一到 [-1,1]


def build_indicators(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """从 OHLCV DataFrame 构造指标库 {name: np.ndarray[T]}（因果）。"""
    if df.empty:
        return {}
    close = df["close"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)
    T = len(close)
    eps = 1e-9

    ret = np.zeros(T)
    ret[1:] = close[1:] / (close[:-1] + eps) - 1.0

    ind: dict[str, np.ndarray] = {}
    ind["close"] = close
    ind["open"] = open_
    ind["high"] = high
    ind["low"] = low
    ind["volume"] = vol
    # amount：无成交额列时用 close*volume 近似（量纲一致即可，因子大多用比例）
    ind["amount"] = df["amount"].values.astype(np.float64) if "amount" in df.columns \
        else close * vol

    ind["ret"] = ret
    ind["ret5"] = np.concatenate([np.zeros(5), close[5:] / (close[:-5] + eps) - 1.0]) \
        if T > 5 else np.zeros(T)
    ind["ret20"] = np.concatenate([np.zeros(20), close[20:] / (close[:-20] + eps) - 1.0]) \
        if T > 20 else np.zeros(T)

    # 波动类
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])),
                               np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
    ind["atr"] = _roll_full(tr, 14)
    ind["rvol"] = _roll_full(ret, 20, fill=0.0, agg="std")
    ind["hl_range"] = (high - low) / (close + eps)
    ma20 = _roll_full(close, 20, fill=close.mean())
    vol_ma20 = _roll_full(vol, 20, fill=vol.mean())
    ind["vol_regime"] = vol / (vol_ma20 + eps)

    # 动量/反转
    ind["rsi14"] = _rsi(close)
    ema12, ema26 = _ema(close, 12), _ema(close, 26)
    macd = ema12 - ema26
    ind["macd_hist"] = macd - _ema(macd, 9)
    std20 = np.sqrt(_roll_full((close - ma20) ** 2, 20, fill=0.0) + eps)
    ind["boll_pos"] = np.clip((close - (ma20 - 2 * std20)) / (4 * std20 + eps), 0, 1)
    ind["boll_width"] = (4 * std20) / (ma20 + eps)
    obv = np.cumsum(np.sign(ret) * vol)
    # M24 修复：obv_slope 应为 OBV 的 w 期线性回归斜率（原实现为滚动 std）。
    ind["obv_slope"] = _linear_slope(obv, 20)
    # M22 修复：mfi14 应为资金流量指标 MFI（正资金流/负资金流比值），
    # 原实现是"资金流强度均值"，与 features.py 的 _mfi 定义矛盾。
    typical = (high + low + close) / 3.0
    mf_flow = typical * vol
    pc = np.concatenate([[typical[0]], typical[:-1]])  # 前收盘（因果）
    pos_mf = np.where(typical > pc, mf_flow, 0.0)
    neg_mf = np.where(typical < pc, mf_flow, 0.0)
    pos_sum = _roll_full(pos_mf, 14) * 14
    neg_sum = _roll_full(neg_mf, 14) * 14
    mfr = pos_sum / (neg_sum + eps)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    ind["mfi14"] = (mfi - 50.0) / 50.0

    # 反转类
    hh14 = _roll_full(high, 14, agg="max")
    ll14 = _roll_full(low, 14, agg="min")
    ma14 = _roll_full(close, 14)
    std14 = np.sqrt(_roll_full((close - ma14) ** 2, 14, fill=0.0) + eps)
    ind["willr_14"] = -100.0 * (hh14 - close) / (hh14 - ll14 + eps)
    ind["cci_14"] = (close - ma14) / (0.015 * std14 + eps)
    ind["roc_12"] = np.concatenate([np.zeros(12), close[12:] / (close[:-12] + eps) - 1.0]) \
        if T > 12 else np.zeros(T)
    ind["typical_dev"] = close / (ma20 + eps) - 1.0

    # 趋势类
    ind["ema_ratio_12_26"] = ema12 / (ema26 + eps) - 1.0
    ind["trend_strength_50"] = np.concatenate(
        [np.zeros(49), close[49:] / (close[:-49] + eps) - 1.0]) if T > 49 else np.zeros(T)
    hh50 = _roll_full(high, 50, fill=close.max())
    ll50 = _roll_full(low, 50, fill=close.min())
    ind["price_pos_50"] = np.clip((close - ll50) / (hh50 - ll50 + eps), 0, 1)
    trix = _ema(_ema(_ema(close, 15), 15), 15)
    # 后向差分（因果：t 只用 t 及以前）。
    # 历史 bug：np.gradient 默认中心差分 gradient[i]=(trix[i+1]-trix[i-1])/2，
    # t 时刻用到了 t+1 未来值 → TRIX 因子前视泄漏、单品种回测虚高。
    # 与 features.py 的 _trix（后向差分）保持一致。
    trix_diff = np.concatenate([[0.0], trix[1:] - trix[:-1]])
    ind["trix_15"] = trix_diff / (trix + eps)
    ppo = (ema12 - ema26) / (ema26 + eps)
    ind["ppo"] = ppo
    # M23 修复：ult_osc 应为 3 周期(7/14/28)加权买压 Ultimate Oscillator，
    # 原实现是 28 期 (C-L)/(H-L) 位置比，与 features.py 的 _ult_osc 定义矛盾。
    pc_uo = np.concatenate([[close[0]], close[:-1]])   # 前收盘（因果）
    tl = np.minimum(low, pc_uo)
    th = np.maximum(high, pc_uo)
    bp_uo = close - tl                                  # buying pressure
    tr_uo = th - tl                                     # true range

    def _uo_avg(w: int) -> np.ndarray:
        return _roll_full(bp_uo, w) / (_roll_full(tr_uo, w) + eps)

    uo = (4.0 * _uo_avg(7) + 2.0 * _uo_avg(14) + _uo_avg(28)) / 7.0
    ind["ult_osc"] = np.clip(uo * 2.0 - 1.0, -1.0, 1.0)

    # 微观结构（华泰实证核心；无笔数数据时用近似，仅用于比例特征）
    num_trades = vol / 100.0  # 手数/100 视为笔数量级近似
    ind["num_trades"] = num_trades
    ind["amt_per_trade"] = ind["amount"] / (num_trades + eps)
    ind["trade_size_reg_intercept"] = ind["amt_per_trade"] / (ind["amount"] / (vol + eps) + eps)
    ind["amt_vol_euclid"] = np.sqrt((np.log1p(ind["amount"]) - np.log1p(vol)) ** 2)
    # 成交笔数 lag-1 自相关（滚动窗口）
    ind["num_trades_ac1"] = _roll_ac1(num_trades, 20)

    return ind


def _roll_ac1(x: np.ndarray, w: int) -> np.ndarray:
    """滚动 lag-1 自相关（因果，warm-up 0）。"""
    T = len(x)
    if T < w + 1:
        return np.zeros(T)
    out = np.zeros(T)
    wnd = _roll(x, w)  # [T-w+1, w]
    a, b = wnd[:, :-1], wnd[:, 1:]
    am, bm = a.mean(axis=1, keepdims=True), b.mean(axis=1, keepdims=True)
    cov = ((a - am) * (b - bm)).mean(axis=1)
    sa = ((a - am) ** 2).mean(axis=1) ** 0.5
    sb = ((b - bm) ** 2).mean(axis=1) ** 0.5
    # 滑动窗口第 i 行对应原序列 t = w-1+i，因此从 w-1 开始写入
    out[w - 1:] = cov / (sa * sb + 1e-9)
    return out


def indicators_from_store(code: str, timeframe: str = "1d") -> dict[str, np.ndarray]:
    """从 KlineStore 读 K 线并构造指标库（便捷入口）。"""
    from data_pipeline.store.kline_store import KlineStore
    df = KlineStore().load(code, timeframe)
    if df.empty:
        raise FileNotFoundError(f"K线库中无 {code}_{timeframe}.parquet，请先运行 backfill")
    return build_indicators(df)
