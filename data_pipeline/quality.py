"""
data_pipeline/quality.py — K 线数据健康检查与清洗（机构级 D3 标准）

对齐 Qlib check_data_health（缺失/大跳变/必需列/重复日期）：
  1. check_series   : 检查必需列、重复日期、缺失值、价格大跳变（复权瑕疵/停复牌）、
                      OHLC 一致性、粘滞价格、未来时间戳、成交量异常
  2. clean_series   : 清洗：去重复日期、非正/非有限价格前值填充、OHLC 关系修复、
                      剔除未来时间戳、标记跳变日（供收益标签置 0，避免伪收益）
  3. mask_jump_returns : 把跳变日的收益标签置 0（数据瑕疵不产生伪收益）
  4. classify_jumps : 跳变分类（交替方向=复权口径混库；单向=除权/单次异常）——
                      机构 D3 复权瑕疵诊断，供库健康审计与回测防御使用

跳变阈值：A股主板涨跌停 10%（创业板/科创板 20%），复权/数据瑕疵跳变通常 >22%，
取 22% 作为异常阈值（高于任何合法涨跌幅，见 Qlib limit_threshold=0.099 语境）。

用法：
    from data_pipeline.quality import check_series, clean_series
    issues = check_series(df)              # -> list[str]（空=健康）
    df, jump_dates = clean_series(df)      # -> (清洗后 df, 跳变日期 set)
    kind, detail = classify_jumps(df)      # -> ("mix"|"one_way"|"clean", 明细)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

REQUIRED_COLS = ("ts", "open", "high", "low", "close", "volume")
# 异常跳变阈值：高于创业板 20% 涨跌幅上限（Qlib 中国模式 limit 9.9%，加容差）
JUMP_THRESHOLD = 0.22
# A股面值退市线：低于 1 元的 bar 不参与跳变判定（仙股/穿零 ffill 产物，
# 正常波动即超阈值，会误报混库；A股真实日线 <1 元仅存于退市整理期）
MIN_PRICE = 1.0
# 粘滞价格判定：连续 >= 该天数收盘价完全相同 → 疑似停牌/数据冻结
STICKY_MIN_DAYS = 5
# 混库判定：跳变中方向交替占比（异号相邻跳变 / 总跳变）>= 该比例 → 复权口径混库
MIX_ALTERNATION_RATIO = 0.5
# 成交量异常：收盘价变动但成交量恒 0 的天数占比阈值
ZERO_VOL_RATIO = 0.1
# 未来时间戳容差（秒）：允许跨时区/半日差异，超过即视为脏数据
FUTURE_TOLERANCE_S = 2 * 86400


def check_series(df: pd.DataFrame) -> list[str]:
    """健康检查，返回问题列表（空 = 健康）。"""
    issues: list[str] = []
    if df is None or df.empty:
        return ["空数据"]
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        issues.append(f"缺列: {missing_cols}")
        return issues
    if df["ts"].duplicated().any():
        issues.append(f"重复日期 {int(df['ts'].duplicated().sum())} 个")
    n_nan = int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum())
    if n_nan:
        issues.append(f"缺失值 {n_nan} 个")
    # 非正/非有限价格（停牌日 0 价格、数据瑕疵）——机构 D3 必检项
    for col in ("open", "high", "low", "close"):
        v = df[col].values.astype(np.float64)
        n_bad = int((~np.isfinite(v) | (v <= 0)).sum())
        if n_bad:
            issues.append(f"{col} 非正/非有限 {n_bad} 个（疑似停牌 0 价格/瑕疵）")
    close = df["close"].values.astype(np.float64)
    if len(close) > 1:
        valid = (close[:-1] >= MIN_PRICE) & (close[1:] >= MIN_PRICE)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.abs(close[1:] / close[:-1] - 1.0)
        n_jump = int((valid & (ret > JUMP_THRESHOLD)).sum())
        if n_jump:
            issues.append(f"异常跳变 {n_jump} 个（>{JUMP_THRESHOLD:.0%}，疑似复权瑕疵）")
    issues.extend(check_ohlc_consistency(df))
    # 粘滞价格：连续多日收盘价完全相同（停牌/数据冻结，因子会退化为常数）
    n_sticky, sticky_runs = detect_sticky_prices(df)
    if n_sticky:
        issues.append(f"粘滞价格 {n_sticky} 日（连续≥{STICKY_MIN_DAYS}日收盘价相同，"
                      f"疑似停牌/数据冻结，{len(sticky_runs)} 段）")
    # 未来时间戳（脏数据/时区错位）
    n_future = count_future_timestamps(df)
    if n_future:
        issues.append(f"未来时间戳 {n_future} 个（晚于当前时间，疑似脏数据）")
    # 成交量异常：收盘价有变动但成交量恒 0（停牌日应同时价格冻结）
    n_zero_vol = count_zero_volume_trading(df)
    if n_zero_vol:
        issues.append(f"价格变动但成交量恒 0 的天 {n_zero_vol} 个（疑似缺量数据）")
    return issues


def check_ohlc_consistency(df: pd.DataFrame) -> list[str]:
    """OHLC 一致性：high >= max(open, close) 且 low <= min(open, close)。

    违反 = 数据源瑕疵（腾讯/新浪/通达信个别 bar 高低价倒挂），
    会在指标计算中产生伪信号，机构 D3 必检。
    """
    if df is None or df.empty:
        return []
    issues: list[str] = []
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    n = int((h + 1e-9 < np.maximum(o, c)).sum())
    if n:
        issues.append(f"OHLC 倒挂 high<max(o,c) {n} 个")
    n = int((l - 1e-9 > np.minimum(o, c)).sum())
    if n:
        issues.append(f"OHLC 倒挂 low>min(o,c) {n} 个")
    return issues


def detect_sticky_prices(df: pd.DataFrame,
                         min_days: int = STICKY_MIN_DAYS) -> tuple[int, list[tuple[int, int]]]:
    """检测粘滞价格段：连续 >= min_days 日收盘价完全相同。

    Returns:
        (粘滞总天数, [(起始ts, 结束ts), ...])    """
    if df is None or df.empty or "close" not in df.columns:
        return 0, []
    c = df["close"].values.astype(np.float64)
    ts = df["ts"].values.astype(np.int64)
    n = len(c)
    total = 0
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and c[j + 1] == c[i]:
            j += 1
        if j - i + 1 >= min_days:
            total += j - i + 1
            runs.append((int(ts[i]), int(ts[j])))
        i = j + 1
    return total, runs


def detect_suspensions(df: pd.DataFrame) -> set[int]:
    """停牌日识别（机构 D2）：成交量恒 0 且价格与前一日完全相同的交易日。

    数据源无停牌标记 → 用「零成交 + 价格冻结」双重条件近似（单条件误报高：
    缩量一字板成交量为 0 但价格会变；放量日价格也可能巧合相同）。
    Returns: 停牌日 ts 集合（调用方用于因子/标签剔除）。
    """
    if df is None or df.empty or "volume" not in df.columns:
        return set()
    v = df["volume"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    ts = df["ts"].values.astype(np.int64)
    out: set[int] = set()
    for i in range(1, len(c)):
        if v[i] <= 0 and c[i] == c[i - 1]:
            out.add(int(ts[i]))
    return out


def count_future_timestamps(df: pd.DataFrame,
                            tolerance_s: int = FUTURE_TOLERANCE_S) -> int:
    """未来时间戳计数（晚于 now + 容差）。"""
    if df is None or df.empty or "ts" not in df.columns:
        return 0
    now = time.time() + tolerance_s
    return int((df["ts"].values.astype(np.int64) > now).sum())


def count_zero_volume_trading(df: pd.DataFrame,
                              zero_vol_ratio: float = ZERO_VOL_RATIO) -> int:
    """收盘价有变动但成交量恒 0 的天数（缺量数据，无法支撑价格有效性）。"""
    if df is None or df.empty or "volume" not in df.columns:
        return 0
    c = df["close"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    if len(c) < 2:
        return 0
    price_moved = np.abs(c[1:] / np.maximum(c[:-1], 1e-9) - 1.0) > 1e-9
    zero_vol = v[1:] <= 0
    n = int((price_moved & zero_vol).sum())
    # 只有占比超阈值才算异常（新股/停牌期少量 0 量正常）
    return n if n >= zero_vol_ratio * len(c) else 0


def classify_jumps(df: pd.DataFrame,
                   threshold: float = JUMP_THRESHOLD) -> tuple[str, dict]:
    """跳变分类（复权瑕疵诊断，机构 D3）。

    跳变（|日收益| > threshold）的方向模式：
      - 交替方向（涨-跌-涨-跌…）占比 >= MIX_ALTERNATION_RATIO → "mix"：复权口径
        混库（qfq 与不复权价格交替），价格不可信，应整库重拉
      - 否则 → "one_way"：单次/单向大跳变，多为除权除息（不复权数据）或
        数据源单点瑕疵，跳变日收益置 0 即可
      - 无跳变 → "clean"

    Returns:
        (kind, {"n": 总跳变数, "alternations": 方向交替数,
                "big": 幅度>100% 的跳变数, "dates": [跳变发生日 ts, ...]})
    """
    if df is None or df.empty or "close" not in df.columns or len(df) < 2:
        return "clean", {"n": 0, "alternations": 0, "big": 0, "dates": []}
    c = df["close"].values.astype(np.float64)
    ts = df["ts"].values.astype(np.int64) if "ts" in df.columns else None
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = c[1:] / np.maximum(c[:-1], 1e-9) - 1.0
    # 价格有效性：两端均 >= MIN_PRICE 才参与跳变判定（仙股/穿零 ffill 免疫）
    valid = (c[:-1] >= MIN_PRICE) & (c[1:] >= MIN_PRICE)
    idx = np.where(valid & (np.abs(ret) > threshold))[0]
    n = int(len(idx))
    if n == 0:
        return "clean", {"n": 0, "alternations": 0, "big": 0, "dates": []}
    signs = np.sign(ret[idx])
    alternations = int((np.diff(signs) != 0).sum()) if len(signs) > 1 else 0
    big = int((np.abs(ret[idx]) > 1.0).sum())
    dates = [int(ts[i + 1]) for i in idx] if ts is not None else []
    # 混库判定：至少 3 次跳变（单点异常只产生 2 次相邻反向跳变，如 +150% 再
    # 跌回 -60%，不是口径交替）且方向交替占比达标
    ratio = alternations / max(n - 1, 1)
    kind = "mix" if n >= 3 and ratio >= MIX_ALTERNATION_RATIO else "one_way"
    return kind, {"n": n, "alternations": alternations,
                  "big": big, "dates": dates}


def clean_series(df: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    """清洗：去重复日期（保留最后一条）、非正/非有限价格前值填充、
    OHLC 关系修复、剔除未来时间戳、检测跳变日（返回日期 set）。

    跳变日由调用方处理：收益标签置 0（避免数据瑕疵产生伪收益），
    因子计算仍用原始价格（跳变是真实价格变动，因果计算不受影响）。
    """
    out = df.copy()
    if out.empty:
        return out, set()
    if "ts" in out.columns:
        out["ts"] = out["ts"].astype("int64")
        # 剔除未来时间戳（脏数据：时区错位/虚假 bar）
        now = time.time() + FUTURE_TOLERANCE_S
        n_future = int((out["ts"] > now).sum())
        if n_future:
            out = out[out["ts"] <= now]
        out = out.sort_values("ts").drop_duplicates(subset="ts", keep="last")
    # 非正/非有限价格 → 前值填充（首日异常则整行剔除）
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            continue
        v = out[col].values.astype(np.float64)
        bad = ~np.isfinite(v) | (v <= 0) if col != "volume" else ~np.isfinite(v)
        if bad.any():
            v = v.copy()
            v[bad] = np.nan
            # pandas 2.x Copy-on-Write 下 .values 可能返回只读视图 → 显式 copy
            v = pd.Series(v).ffill().to_numpy(dtype=np.float64, copy=True)
            v[~np.isfinite(v)] = 0.0 if col == "volume" else np.nan
            out[col] = v
    if "close" in out.columns:
        out = out[out["close"].notna() & (out["close"] > 0)].reset_index(drop=True)
    # OHLC 关系修复：high 取 max(high, open, close)、low 取 min(low, open, close)
    # （数据源个别 bar 高低价倒挂，修正后指标计算不产生伪信号）
    if all(c in out.columns for c in ("open", "high", "low", "close")):
        o = out["open"].values.astype(np.float64)
        h = out["high"].values.astype(np.float64)
        l = out["low"].values.astype(np.float64)
        c = out["close"].values.astype(np.float64)
        out["high"] = np.maximum.reduce([h, o, c])
        out["low"] = np.minimum.reduce([l, o, c])
    jump_dates: set[int] = set()
    if "close" in out.columns and len(out) > 1:
        close = out["close"].values.astype(np.float64)
        valid = (close[:-1] >= MIN_PRICE) & (close[1:] >= MIN_PRICE)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.abs(close[1:] / close[:-1] - 1.0)
        bad = np.where(valid & (ret > JUMP_THRESHOLD))[0]
        ts = out["ts"].values if "ts" in out.columns else None
        for i in bad:
            jump_dates.add(int(ts[i + 1]) if ts is not None else i + 1)
    return out.reset_index(drop=True), jump_dates


def mask_jump_returns(ret: np.ndarray, ts: np.ndarray,
                      jump_dates: set[int]) -> np.ndarray:
    """把跳变日的收益标签置 0（D3：数据瑕疵不产生伪收益）。"""
    out = ret.copy()
    if not jump_dates:
        return out
    for i, t in enumerate(ts):
        if int(t) in jump_dates:
            out[i] = 0.0
    return out
