"""
model_core/portfolio/combination.py — 多因子合成（P14）

对标华泰金工：多因子合成用 IC_IR 加权（机构经典）或机器学习模型
（随机森林，强拟合 vs 过拟合权衡）。机构从不单因子交易——合成因子
才是组合层输入。

实现：
  1. icir_weights      : 滚动窗口 IC_IR 权重（只用于历史窗口，防前视）
  2. combine_icir      : IC_IR 加权合成（逐日横截面）
  3. combine_ml        : 随机森林合成（可选，sklearn；时序滚动训练防前视）
  4. combine_equal     : 等权合成（基准对照）

用法：
    from model_core.portfolio.combination import combine_icir, combine_ml
    composite = combine_icir(panels, ret_panel, window=60)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def icir_weights(panels: list[pd.DataFrame], ret_panel: pd.DataFrame,
                 window: int = 60) -> np.ndarray:
    """滚动窗口 IC_IR 权重（华泰：w ∝ IC_mean / IC_std，防前视）。

    对每个因子面板，用最近 window 天的横截面 RankIC 序列估计
    IC 均值与 IC 标准差 → ICIR = mean/std（NaN 安全），归一化权重。
    """
    from scipy.stats import spearmanr

    ics: list[list[float]] = []
    for p in panels:
        seq = []
        for ts in p.index[-window:]:
            if ts not in ret_panel.index:
                continue
            f = p.loc[ts].astype(float)
            r = ret_panel.loc[ts].astype(float)
            common = f.index.intersection(r.index)
            fv, rv = f[common].values, r[common].values
            ok = np.isfinite(fv) & np.isfinite(rv)
            if ok.sum() >= 10 and np.std(fv[ok]) > 1e-12:
                seq.append(spearmanr(fv[ok], rv[ok]).statistic)
        ics.append(seq)
    icir = []
    for seq in ics:
        if len(seq) < 10:
            icir.append(0.0)
            continue
        sd = float(np.std(seq))
        icir.append(float(np.mean(seq)) / sd if sd > 1e-12 else 0.0)
    w = np.array(icir, dtype=np.float64)
    w = np.clip(w, -1.0, 1.0)          # 负 ICIR 反向（方向翻转）
    s = float(np.abs(w).sum())
    if s <= 1e-12:
        return np.full(len(panels), 1.0 / max(len(panels), 1))
    return w / s


def combine_icir(panels: list[pd.DataFrame], ret_panel: pd.DataFrame,
                 window: int = 60) -> tuple[pd.DataFrame, np.ndarray]:
    """IC_IR 加权合成：composite_t = Σ w_i · zscore(f_i,t)。

    Returns: (合成面板, 权重数组)
    """
    if not panels:
        return pd.DataFrame(), np.array([])
    w = icir_weights(panels, ret_panel, window=window)
    out = None
    for p, wi in zip(panels, w):
        z = p.apply(lambda row: (row - row.mean()) / (row.std() + 1e-12), axis=1)
        term = z * wi
        out = term if out is None else out.add(term, fill_value=0.0)
    return out, w


def combine_ml(panels: list[pd.DataFrame], ret_panel: pd.DataFrame,
               window: int = 120, n_estimators: int = 100,
               seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """随机森林合成（华泰：ML 合成 + 时序交叉验证防过拟合）。

    滚动训练：每 20 个交易日用前 window 天训练 RF（特征=因子面板，
    标签=未来收益），预测后续 20 天合成得分。防前视：训练只用历史。

    Returns: (合成面板, {"n_estimators", "train_days", "feature_imp"})
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise RuntimeError("sklearn 未安装，无法 ML 合成") from exc

    if not panels:
        return pd.DataFrame(), {}
    ts_all = sorted(panels[0].index)
    symbols = list(panels[0].columns)
    out = pd.DataFrame(np.nan, index=ts_all, columns=symbols)
    step = 20
    importances: list[np.ndarray] = []
    n_train = 0
    for start in range(0, len(ts_all), step):
        end = min(start + step, len(ts_all))
        if end > len(ts_all):
            break
        test_ts = ts_all[start:end]
        train_ts = ts_all[max(0, start - window):start]
        if len(train_ts) < 60:
            continue
        # 特征矩阵 [T_train, N×F]：展平因子面板（按标的列对齐）
        Xs, ys = [], []
        for t in train_ts:
            row = []
            for p in panels:
                if t in p.index:
                    row.append(p.loc[t].reindex(symbols).values)
                else:
                    row.append(np.full(len(symbols), np.nan))
            Xs.append(np.concatenate(row))
            if t in ret_panel.index:
                ys.append(ret_panel.loc[t].reindex(symbols).values)
            else:
                ys.append(np.full(len(symbols), np.nan))
        X = np.array(Xs, dtype=np.float64)
        y = np.array(ys, dtype=np.float64)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
        if ok.sum() < 60:
            continue
        model = RandomForestRegressor(n_estimators=n_estimators,
                                      random_state=seed, n_jobs=-1)
        model.fit(X[ok], y[ok])
        importances.append(model.feature_importances_)
        n_train += int(ok.sum())
        # 预测测试期
        for t in test_ts:
            row = []
            for p in panels:
                if t in p.index:
                    row.append(p.loc[t].reindex(symbols).values)
                else:
                    row.append(np.full(len(symbols), np.nan))
            Xt = np.array([np.concatenate(row)], dtype=np.float64)
            pred = model.predict(Xt)[0]
            out.loc[t] = pred
    report = {"n_estimators": n_estimators, "train_days": n_train,
              "feature_imp": float(np.mean(importances)) if importances else 0.0}
    return out, report


def combine_equal(panels: list[pd.DataFrame]) -> pd.DataFrame:
    """等权合成（基准对照）。"""
    if not panels:
        return pd.DataFrame()
    out = None
    for p in panels:
        z = p.apply(lambda row: (row - row.mean()) / (row.std() + 1e-12), axis=1)
        out = z if out is None else out.add(z, fill_value=0.0)
    return out / len(panels)
