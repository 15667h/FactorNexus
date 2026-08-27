"""
model_core/strategy_factory/ensemble.py — M5 模型集成 + 评估接入

把模型池（LGBM / MLP / S4 …）集成成单一更强信号，机构标准做法：

  1. rank_average          —— 逐日横截面排名平均（Qlib/blend 常用，
                              对量纲差异鲁棒：ML 预测值本身无量纲语义）
  2. make_bagging_factory  —— 同模型多 seed 平均（模型不确定性消减）
  3. make_ensemble_factory —— 异质模型排名平均（fit 时各训一个，predict
                              时输出逐日截面 rank 均值；兼容 walk_forward
                              每折独立调用的接口约定）
  4. stacking_fit_predict  —— 时间分段两层（第一层 base 模型 → 第二层
                              meta 模型）；严格防前视：meta 只在第一层
                              的「训练段预测」上拟合，评估只用「OOS 段」

用法（walk_forward 内直接集成）：
    from model_core.strategy_factory.ensemble import make_ensemble_factory
    factory = make_ensemble_factory([
        lambda: make_lgbm_regressor(),
        lambda: make_mlp_regressor(epochs=15),
    ], method="rank_avg")

用法（独立 stacking，不依赖 walk_forward）：
    from model_core.strategy_factory.ensemble import stacking_fit_predict
    oos = stacking_fit_predict(ds, base_factories, meta_factory)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from model_core.strategy_factory.dataset import FactorDataset
from model_core.strategy_factory.walk_forward import (WalkForwardResult,
                                                      _train_predict)


# ── 1. 横截面排名平均 ────────────────────────────────────────────────────

def rank_average(preds: list[pd.DataFrame],
                 weights: list[float] | None = None) -> pd.DataFrame:
    """多模型 OOS 预测 → 逐日横截面排名加权平均面板。

    Args:
        preds: 预测面板列表（index=交易日, columns=股票）。
        weights: 各模型权重（None=等权）。

    Returns:
        同轴面板（NaN 传播：某日某模型缺预测 → 该模型该日不参与）。
    """
    if not preds:
        return pd.DataFrame()
    if weights is None:
        weights = [1.0 / len(preds)] * len(preds)
    all_idx = sorted(set().union(*[set(p.index) for p in preds]))
    all_cols = sorted(set().union(*[set(p.columns) for p in preds]))
    out = pd.DataFrame(np.nan, index=all_idx, columns=all_cols)
    rank_sum = pd.DataFrame(0.0, index=all_idx, columns=all_cols)
    w_sum = pd.DataFrame(0.0, index=all_idx, columns=all_cols)
    for p, w in zip(preds, weights):
        if p is None or p.empty:
            continue
        r = p.rank(axis=1)                     # 逐日截面排名（NaN→NaN）
        ok = r.notna()
        rank_sum = rank_sum.add((r * w).fillna(0.0), fill_value=0.0)
        w_sum = w_sum.add(ok.astype(float) * w, fill_value=0.0)
    mask = w_sum > 1e-12
    out[mask] = (rank_sum / w_sum.replace(0.0, np.nan))[mask]
    return out


# ── 2. bagging（同模型多 seed）───────────────────────────────────────────

class _BaggingModel:
    """同模型工厂多 seed 实例 → 预测平均（sklearn 风格，兼容 walk_forward）。"""

    def __init__(self, base_factory, n_models: int = 3,
                 seeds: list[int] | None = None) -> None:
        self.base_factory = base_factory
        self.n_models = max(1, int(n_models))
        self.seeds = seeds or [42 + i * 17 for i in range(self.n_models)]
        self._models: list = []

    def fit(self, X, y) -> "_BaggingModel":
        X = np.asarray(X)
        y = np.asarray(y)
        self._models = []
        for sd in self.seeds[:self.n_models]:
            m = self.base_factory(seed=sd) if _accepts_seed(self.base_factory) \
                else self.base_factory()
            m.fit(X, y)
            self._models.append(m)
        return self

    def predict(self, X) -> np.ndarray:
        if not self._models:
            raise RuntimeError("先 fit 再 predict")
        preds = [np.asarray(m.predict(X), dtype=np.float64)
                 for m in self._models]
        return np.nanmean(np.vstack(preds), axis=0)


def _accepts_seed(factory) -> bool:
    import inspect
    try:
        return "seed" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


def make_bagging_factory(base_factory, n_models: int = 3,
                         seeds: list[int] | None = None):
    """返回 model_factory（walk_forward 兼容）：每折训练 n 个同模型平均。"""
    return lambda: _BaggingModel(base_factory, n_models=n_models, seeds=seeds)


# ── 3. 异质模型排名平均集成 ──────────────────────────────────────────────

class _EnsembleRankModel:
    """多异质模型 fit；predict 输出逐样本排名平均（0-1 归一化等价）。"""

    def __init__(self, factories: list, method: str = "rank_avg",
                 weights: list[float] | None = None) -> None:
        self.factories = list(factories)
        self.method = method
        self.weights = weights
        self._models: list = []

    def fit(self, X, y) -> "_EnsembleRankModel":
        self._models = [f() for f in self.factories]
        for m in self._models:
            m.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if not self._models:
            raise RuntimeError("先 fit 再 predict")
        X = np.asarray(X)
        preds = [np.asarray(m.predict(X), dtype=np.float64)
                 for m in self._models]
        if self.method == "rank_avg":
            # 逐模型样本排名 → min-max 归一化 → 加权平均（NaN 样本不参与）
            w = self.weights or [1.0 / len(preds)] * len(preds)
            out = np.zeros(len(X))
            cnt = np.zeros(len(X))
            for p, wi in zip(preds, w):
                s = pd.Series(p)
                r = s.rank().values
                ok = np.isfinite(r)
                if not ok.any():
                    continue
                lo, hi = float(np.nanmin(r)), float(np.nanmax(r))
                rn = (r - lo) / (hi - lo) if hi > lo else \
                    np.zeros_like(r)
                out += wi * np.nan_to_num(rn, nan=0.0)
                cnt += wi * ok.astype(np.float64)
            return np.where(cnt > 1e-12, out / np.maximum(cnt, 1e-12),
                            np.nan)
        # 均值集成（量纲接近时）
        return np.nanmean(np.vstack(preds), axis=0)


def make_ensemble_factory(factories: list, method: str = "rank_avg",
                          weights: list[float] | None = None):
    """返回 model_factory：每折训练全部模型，predict 输出集成信号。

    Args:
        factories: 模型工厂列表（如 [lambda: make_lgbm_regressor(), ...]）。
        method: "rank_avg"（逐样本排名平均）| "mean"（数值平均）。
        weights: 各模型权重（None=等权）。
    """
    return lambda: _EnsembleRankModel(factories, method=method,
                                      weights=weights)


# ── 4. 时间分段 stacking ────────────────────────────────────────────────

def stacking_fit_predict(ds: FactorDataset, base_factories: list,
                         meta_factory, split_frac: float = 0.7,
                         gap: int = 5) -> WalkForwardResult:
    """两层 stacking（时间分段，防前视）。

    流程：
      1. cutoff = 前 split_frac 时间（按 ts 分位）
      2. 第一层：全部 base 模型在「训练段」拟合，预测全样本
         （训练段预测 = in-sample 用于 meta 拟合；OOS 段预测 = 评估输入）
      3. 第二层：meta 模型在训练段 base 预测上拟合（特征=各 base 预测）
      4. 输出：OOS 段的 meta 预测面板（WalkForwardResult 兼容）

    局限标注：第一层在训练段是 in-sample 预测（无 CV），meta 拟合存在
    轻微乐观偏差——这是时间分段 stacking 的标准代价；如需严格折内预测，
    请改用嵌套 walk-forward（成本 ×折数）。
    """
    ts_sorted = np.sort(np.unique(ds.ts))
    cutoff = ts_sorted[int(len(ts_sorted) * split_frac)] - 1
    tr_mask = ds.ts <= cutoff
    te_mask = ds.ts > cutoff
    X_tr, y_tr = ds.X[tr_mask], ds.y[tr_mask]
    if len(X_tr) < 120 or te_mask.sum() < 30:
        return WalkForwardResult(pred=pd.DataFrame(), folds=[])

    # 第一层：训练段拟合 → 全样本预测
    base_preds: list[np.ndarray] = []
    folds: list[dict] = []
    for f in base_factories:
        m = f()
        m.fit(X_tr.values, y_tr.values)
        p = np.asarray(m.predict(ds.X.values), dtype=np.float64)
        base_preds.append(p)
    # 第二层：meta 在训练段 base 预测上拟合
    P_tr = np.vstack([p[tr_mask] for p in base_preds]).T
    P_te = np.vstack([p[te_mask] for p in base_preds]).T
    meta = meta_factory()
    meta.fit(P_tr, y_tr.values)
    y_te = meta.predict(P_te)

    # 面板组装（OOS 段）
    te_ts = ds.ts[te_mask]
    te_sym = ds.symbol[te_mask]
    pred_rows: dict[int, dict[str, float]] = {}
    for t, s, v in zip(te_ts, te_sym, y_te):
        if np.isfinite(v):
            pred_rows.setdefault(int(t), {})[str(s)] = float(v)
    idx = pd.to_datetime(sorted(pred_rows.keys()), unit="s")
    symbols = sorted({str(s) for s in te_sym})
    pred = pd.DataFrame(np.nan, index=idx, columns=symbols)
    for t, row in pred_rows.items():
        pred.loc[pd.Timestamp(t, unit="s")] = row
    folds.append({
        "fold": 0, "method": "stacking_time_split",
        "train_to": int(cutoff), "test_from": int(ts_sorted[-1]),
        "n_train": int(len(X_tr)), "n_test": int(te_mask.sum()),
        "n_base": len(base_factories),
    })
    return WalkForwardResult(pred=pred, folds=folds)
