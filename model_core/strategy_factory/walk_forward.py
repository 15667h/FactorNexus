"""
model_core/strategy_factory/walk_forward.py — M2 walk-forward 训练框架

机构标准时序验证（Qlib 对齐）：滚动训练 → 预测下一段 → 收集 OOS 预测。

防前视三重保险：
  1. 特征全因果（dataset 保证）
  2. 训练段与预测段严格不相交，且留 gap 根（避免标签重叠泄漏——
     Qlib 标准做法：预测 H 日收益时，训练最后 H 根与预测首根标签重叠）
  3. 评估只用 OOS 预测（训练段绝不回看）

用法：
    from model_core.strategy_factory.walk_forward import walk_forward_fit_predict
    oos = walk_forward_fit_predict(ds, model_factory, step=60, window=240, gap=5)
    # oos: pd.DataFrame(index=交易日, columns=股票, 值=OOS 预测)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from model_core.strategy_factory.dataset import FactorDataset


@dataclass
class WalkForwardResult:
    """walk-forward 输出。"""
    pred: pd.DataFrame           # OOS 预测面板（index=交易日, columns=股票）
    folds: list[dict] = field(default_factory=list)   # 每折信息

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def oos_days(self) -> int:
        return int(self.pred.shape[0])

    @property
    def coverage(self) -> float:
        """OOS 预测覆盖率（非 NaN 样本占比）。"""
        if self.pred.empty:
            return 0.0
        return float(self.pred.notna().sum().sum()
                     / max(self.pred.size, 1))


def _train_predict(model, X_tr, y_tr, X_te, ts_te=None):
    """训练 + 预测（模型接口统一：fit(X, y) → predict(X)）。

    ts_te: 测试样本时间戳（可选）。支持逐日截面排名的模型（如
    _EnsembleRankModel 的 predict_with_ts）用它做正确的逐日排名（M3）。
    """
    model.fit(X_tr, y_tr)
    if ts_te is not None and hasattr(model, "predict_with_ts"):
        return np.asarray(model.predict_with_ts(X_te, ts_te), dtype=np.float64)
    return np.asarray(model.predict(X_te), dtype=np.float64)


def walk_forward_fit_predict(ds: FactorDataset, model_factory,
                             step: int = 60, window: int = 240,
                             gap: int | None = None, min_train: int = 120,
                             progress: bool = True) -> WalkForwardResult:
    """walk-forward 滚动训练预测。

    Args:
        ds: 训练数据集（含 ts 轴）。
        model_factory: 无参可调用 → 返回新模型实例（如 lambda: LGBMRegressor()）。
        step: 每折预测长度（交易日）。
        window: 训练窗口（rolling；None=expanding）。
        gap: 训练尾与预测首之间的间隔（防标签重叠泄漏）。
             None=自动取 ds.meta["horizon"]（默认，推荐）；
             显式传入过小值（gap < horizon）时打印告警——标签是未来 H 日收益，
             训练最后样本的标签会读到 close[ts+H]，若 gap < H 则读到测试期价格。
        min_train: 最少训练样本数（不足跳过该折）。

    Returns:
        WalkForwardResult：pred 面板（OOS 预测，index=交易日升序，
        columns=股票，值=预测得分）；folds 记录每折 train/test 区间与样本数。
    """
    # gap 防泄漏核心（M1/H3 修复）：默认必须 = 预测周期 horizon。
    # 历史 bug：硬编码 gap=5，horizon≠5 时训练标签越过测试首日形成前视。
    if gap is None:
        gap = int(ds.meta.get("horizon", 5))
    horizon = int(ds.meta.get("horizon", 5))
    if gap < horizon:
        print(f"[警告] walk_forward gap={gap} < horizon={horizon}，"
              f"训练标签可能读到测试期价格（建议 gap >= horizon）")
    ts_sorted = np.sort(np.unique(ds.ts))
    all_ts = ts_sorted
    symbols = sorted(set(ds.symbol))
    pred_rows: dict[int, dict[str, float]] = {}
    folds: list[dict] = []

    # 折边界：以预测段为单位滚动
    n = len(all_ts)
    starts = list(range(0, n, step))
    if starts and starts[-1] + step < n:
        starts.append(n - step)     # 保证覆盖末尾
    for fold_i, te_start in enumerate(starts):
        te_end = min(te_start + step, n)
        if te_end <= te_start:
            continue
        # 训练段：te_start 之前，留 gap
        tr_start_idx = 0 if window is None else max(0, te_start - window)
        te_start_ts = all_ts[te_start]
        # gap：训练段最后允许的交易日 = 预测首日 - gap 根
        gap_idx = max(0, te_start - gap)
        tr_ts_bound = all_ts[gap_idx - 1] if gap_idx > 0 else all_ts[0] - 1

        tr_mask = (ds.ts <= tr_ts_bound) & (ds.ts >= all_ts[tr_start_idx])
        te_mask = (ds.ts >= te_start_ts) & (ds.ts <= all_ts[te_end - 1])
        # M5 修复：数据集行序按股票分组而非全局时间序——MLP/S4 内部按行序取
        # 尾 10% 作验证集，会取到"最后一只股票的尾部"而非最近期时段。
        # 训练样本按 ts 升序重排，使 NN 验证集为最近期时段（防泄漏声明成立）。
        tr_idx = np.flatnonzero(tr_mask)
        _order = np.argsort(ds.ts[tr_idx], kind="stable")
        tr_idx = tr_idx[_order]
        X_tr = ds.X.iloc[tr_idx]
        y_tr = ds.y.iloc[tr_idx]
        X_te = ds.X[te_mask]
        if len(X_tr) < min_train or len(X_te) < 10:
            continue
        try:
            model = model_factory()
            pred = _train_predict(model, X_tr, y_tr, X_te,
                                  ts_te=ds.ts[te_mask])
        except Exception:  # noqa: BLE001
            continue
        te_ts = ds.ts[te_mask]
        te_sym = ds.symbol[te_mask]
        for t, s, p in zip(te_ts, te_sym, pred):
            if np.isfinite(p):
                pred_rows.setdefault(int(t), {})[str(s)] = float(p)
        folds.append({
            "fold": fold_i,
            "train_from": int(all_ts[tr_start_idx]),
            "train_to": int(tr_ts_bound),
            "test_from": int(te_start_ts),
            "test_to": int(all_ts[te_end - 1]),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        })

    idx = pd.to_datetime(sorted(pred_rows.keys()), unit="s")
    pred = pd.DataFrame(
        np.nan, index=idx,
        columns=[s for s in symbols if any(s in row
                                          for row in pred_rows.values())])
    for t, row in pred_rows.items():
        pred.loc[pd.Timestamp(t, unit="s")] = row
    return WalkForwardResult(pred=pred, folds=folds)
