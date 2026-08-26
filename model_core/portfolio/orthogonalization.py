"""
model_core/portfolio/orthogonalization.py — 因子正交化（P14）

对标华泰金工：增量信息挖掘必须正交化——新因子对既有因子做截面回归，
取残差作为"增量 alpha"（以残差收益率为目标的增量信息挖掘）。

实现：
  1. orthogonalize_panel : 逐日横截面 OLS 残差化（新因子 ~ 基准因子集）
  2. orthogonalize_series : 单标的时序 OLS 残差化（因子 ~ 基准因子）
  3. incremental_rankic  : 正交化后的增量 RankIC（相对残差收益率）

用法：
    from model_core.portfolio.orthogonalization import orthogonalize_panel
    inc = orthogonalize_panel(new_panel, bench_panels)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def orthogonalize_panel(new_panel: pd.DataFrame,
                        bench_panels: list[pd.DataFrame]) -> pd.DataFrame:
    """逐日横截面正交化：new ~ [1, bench1, bench2, ...] 取残差。

    Args:
        new_panel: 新因子面板（index=ts, columns=symbol）。
        bench_panels: 基准因子面板列表（已入库因子，同轴对齐）。

    Returns:
        残差面板（同轴）。基准缺失日/标的 → 该日残差 = 原始值（无基准可正交）。
    """
    if new_panel.empty:
        return new_panel
    if not bench_panels:
        return new_panel.copy()
    out = new_panel.copy()
    symbols = list(new_panel.columns)
    for ts in new_panel.index:
        y = new_panel.loc[ts].values.astype(np.float64)
        Xcols: list[np.ndarray] = []
        for bp in bench_panels:
            if ts in bp.index:
                row = bp.loc[ts].reindex(symbols).values.astype(np.float64)
                Xcols.append(row)
        if not Xcols:
            continue
        valid = np.isfinite(y)
        for x in Xcols:
            valid &= np.isfinite(x)
        if valid.sum() < 10:
            continue
        X = np.column_stack([np.ones(valid.sum())] +
                            [x[valid] for x in Xcols])
        yv = y[valid]
        try:
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            resid = yv - X @ beta
        except np.linalg.LinAlgError:
            resid = yv - yv.mean()
        sd = float(np.std(resid))
        if sd > 1e-9:
            resid = (resid - resid.mean()) / sd
        out.loc[ts, [s for s, v in zip(symbols, valid) if v]] = resid
        out.loc[ts, [s for s, v in zip(symbols, valid) if not v]] = np.nan
    return out


def orthogonalize_series(factor: np.ndarray,
                         bench: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """单标的时序正交化：factor ~ [1, bench...] OLS 残差（长度对齐）。"""
    f = np.asarray(factor, dtype=np.float64)
    if bench is None or (isinstance(bench, list) and not bench):
        return f.copy()
    benches = [bench] if not isinstance(bench, list) else bench
    benches = [np.asarray(b, dtype=np.float64) for b in benches]
    n = min(len(f), *(len(b) for b in benches))
    if n < 20:
        return f.copy()
    fv = f[-n:]
    X = np.column_stack([np.ones(n)] + [b[-n:] for b in benches])
    valid = np.isfinite(X).all(axis=1) & np.isfinite(fv)
    if valid.sum() < 20:
        return f.copy()
    try:
        beta, *_ = np.linalg.lstsq(X[valid], fv[valid], rcond=None)
        resid = fv - X @ beta
    except np.linalg.LinAlgError:
        resid = fv - fv.mean()
    out = f.copy()
    out[-n:] = resid
    return out


def incremental_rankic(new_panel: pd.DataFrame,
                       bench_panels: list[pd.DataFrame],
                       ret_panel: pd.DataFrame) -> dict:
    """正交化前后的 RankIC 对比（增量信息评估）。

    Returns:
        {"raw_rankic": 原始 RankIC 均值,
         "orth_rankic": 正交化后 RankIC 均值,
         "incremental": 增量 = orth - raw}
    """
    from scipy.stats import spearmanr

    def _mean_rankic(f_panel: pd.DataFrame, r_panel: pd.DataFrame) -> float:
        ics = []
        for ts in f_panel.index:
            if ts not in r_panel.index:
                continue
            f = f_panel.loc[ts].astype(float)
            r = r_panel.loc[ts].astype(float)
            common = f.index.intersection(r.index)
            fv = f[common].values
            rv = r[common].values
            ok = np.isfinite(fv) & np.isfinite(rv)
            if ok.sum() >= 10 and np.std(fv[ok]) > 1e-12:
                ics.append(spearmanr(fv[ok], rv[ok]).statistic)
        return float(np.mean(ics)) if ics else 0.0

    raw = _mean_rankic(new_panel, ret_panel)
    orth = orthogonalize_panel(new_panel, bench_panels)
    orth_ic = _mean_rankic(orth, ret_panel)
    return {"raw_rankic": raw, "orth_rankic": orth_ic,
            "incremental": orth_ic - raw}
