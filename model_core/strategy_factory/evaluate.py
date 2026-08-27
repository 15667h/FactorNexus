"""
model_core/strategy_factory/evaluate.py — 信号评估（策略工厂验收指标）

对齐机构信号评估：横截面 RankIC / IC_IR / 十分组 / 换手 / 时间分段方向。

用法：
    from model_core.strategy_factory.evaluate import evaluate_signal
    rep = evaluate_signal(pred, ret_panel)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cross_sectional_rankic(pred: pd.DataFrame,
                           ret: pd.DataFrame) -> tuple[list[float], list]:
    """逐日横截面 RankIC（Spearman），返回 (IC 序列, 有效交易日)。"""
    ics: list[float] = []
    days: list = []
    for ts in pred.index:
        if ts not in ret.index:
            continue
        f = pred.loc[ts].astype(float)
        r = ret.loc[ts].astype(float)
        common = f.index.intersection(r.index)
        fv, rv = f[common].values, r[common].values
        ok = np.isfinite(fv) & np.isfinite(rv)
        if ok.sum() >= 10 and np.std(fv[ok]) > 1e-12 \
                and np.std(rv[ok]) > 1e-12:
            ic = spearmanr(fv[ok], rv[ok]).statistic
            if np.isfinite(ic):
                ics.append(float(ic))
                days.append(ts)
    return ics, days


def evaluate_signal(pred: pd.DataFrame, ret: pd.DataFrame,
                    ppy: int = 244) -> dict:
    """信号综合评估。

    Returns:
        {n_days, rankic, icir, ic_std, half_agree(前后半段方向一致),
         turnover(信号换手), coverage}
    """
    out: dict = {"n_days": 0, "rankic": 0.0, "icir": 0.0, "ic_std": 0.0,
                 "half_agree": False, "turnover": 0.0, "coverage": 0.0}
    if pred is None or pred.empty:
        return out
    ics, days = cross_sectional_rankic(pred, ret)
    out["n_days"] = len(ics)
    out["coverage"] = float(pred.notna().sum().sum()
                            / max(pred.size, 1))
    if not ics:
        return out
    arr = np.array(ics, dtype=np.float64)
    mean_ic, std_ic = float(arr.mean()), float(arr.std())
    out["rankic"] = mean_ic
    out["ic_std"] = std_ic
    out["icir"] = mean_ic / std_ic * np.sqrt(ppy) if std_ic > 1e-12 else 0.0
    # 时间分段方向一致（机构稳健性：前半/后半 RankIC 同号）
    half = len(arr) // 2
    if half >= 20:
        m1, m2 = float(arr[:half].mean()), float(arr[half:].mean())
        out["half_agree"] = bool(m1 * m2 > 0)
        out["half_rankic"] = (round(m1, 4), round(m2, 4))
    # 信号换手（截面排序变化的平均比例，Qlib turnover 口径）
    if pred.shape[0] >= 2:
        ranks = pred.rank(axis=1)
        diff = ranks.diff().abs().mean(axis=1).dropna()
        out["turnover"] = float(diff.mean()) if len(diff) else 0.0
    return out


def quantile_analysis(pred: pd.DataFrame, ret: pd.DataFrame,
                      n_groups: int = 10) -> dict:
    """十分组收益单调性（机构 E2）。"""
    out = {"group_returns": [], "monotonicity": 0.0, "long_short": 0.0}
    group_rets: list[float] = []
    for ts in pred.index:
        if ts not in ret.index:
            continue
        f = pred.loc[ts].astype(float)
        r = ret.loc[ts].astype(float)
        common = f.index.intersection(r.index)
        fv, rv = f[common], r[common]
        ok = np.isfinite(fv) & np.isfinite(rv)
        if ok.sum() < n_groups * 3:
            continue
        fv, rv = fv[ok], rv[ok]
        q = pd.qcut(fv, n_groups, labels=False, duplicates="drop")
        for g in range(n_groups):
            if g >= q.max() + 1:
                continue
            mask = q == g
            if mask.sum() >= 2:
                group_rets.append(float(rv[mask].mean()))
    if len(group_rets) >= n_groups:
        g = np.array(group_rets[:n_groups])
        out["group_returns"] = [round(float(x), 5) for x in g]
        out["monotonicity"] = float(spearmanr(
            np.arange(len(g)), g).statistic)
        out["long_short"] = round(float(g[-1] - g[0]), 5)
    return out
