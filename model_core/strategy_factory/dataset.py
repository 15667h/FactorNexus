"""
model_core/strategy_factory/dataset.py — M1 数据层

把因子库（单标的时序因子）组织成机器学习训练样本：
  特征矩阵 X：(交易日 × 股票) 行，列为「该股票的因子值 + 风格特征」
  标签 y   ：未来 H 日收益（与挖矿 horizon 一致）

设计（对齐 Qlib 横截面 ML 范式）：
  - 行 = (t, symbol) 样本；特征 = 该股票在 t 日的全部因子值
    （单标的因子 → 每股票只有自己的列有值，其余 NaN；GBDT 原生支持缺失）
  - 风格特征：ret20 / vol20（因果滚动，从 K 线直接计算）
  - 标签 = close[t+H]/close[t] - 1（未来 H 日收益，与挖矿认证口径一致）
  - 全因果：任何特征只用 t 及以前数据

用法：
    from model_core.strategy_factory.dataset import build_dataset
    ds = build_dataset("store", horizon=5)
    # ds.X: 特征 DataFrame（index=MultiIndex[(ts, symbol)], columns=特征）
    # ds.y: 标签 Series；ds.ts / ds.symbol 辅助列
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.store.kline_store import FactorStore, KlineStore
from data_pipeline.quality import clean_series, mask_jump_returns


@dataclass
class FactorDataset:
    """机器学习训练数据集（特征 + 标签 + 轴信息）。"""
    X: pd.DataFrame            # 特征矩阵 [样本, 特征]
    y: pd.Series               # 标签 [样本]
    ts: np.ndarray             # 每样本交易日（int64）
    symbol: np.ndarray         # 每样本股票代码
    feature_names: list        # 特征列名
    meta: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def split_by_time(self, cutoff_ts: int) -> tuple["FactorDataset",
                                                       "FactorDataset"]:
        """按时间切分（防前视核心：train 只用 cutoff 以前，test 用以后）。"""
        tr_mask = self.ts < cutoff_ts
        te_mask = self.ts >= cutoff_ts
        return (self._sub(tr_mask), self._sub(te_mask))

    def _sub(self, mask: np.ndarray) -> "FactorDataset":
        return FactorDataset(
            X=self.X[mask], y=self.y[mask],
            ts=self.ts[mask], symbol=self.symbol[mask],
            feature_names=self.feature_names, meta=self.meta,
        )


def _causal_style_features(close: np.ndarray, vol: np.ndarray,
                           win: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """因果 ret20 / vol20（t 只用 t 及以前；与 indicator_builder 同口径）。"""
    n = len(close)
    ret20 = np.zeros(n)
    if n > win:
        ret20[win:] = close[win:] / close[:-win] - 1.0
    vol20 = np.zeros(n)
    for t in range(win, n):
        seg = close[t - win + 1:t + 1] / close[t - win:t] - 1.0
        vol20[t] = float(np.std(seg))
    return ret20, vol20


def build_dataset(store_dir: str = "store", horizon: int = 5,
                  bars: int = 0,
                  top_factors_per_stock: int = 0) -> FactorDataset:
    """从因子库构建训练数据集（全因果）。

    特征选择（机构标准，2026-08-28 新增）：
      - 高频因子（kind=highfreq）聚合为**共享特征列**（hf_xxx，全股票共用，
        Qlib 横截面特征式）——1407 个 per-symbol 高频列 → 14 个共享列；
      - 日线因子保持 per-symbol 列，可按认证 RankIC 做 **Top-K 筛选**
        （top_factors_per_stock>0 时每股票只保留最强 K 个）；
      - 理由：特征列数决定 LGBM 训练时间与过拟合风险——机构是"大因子库 +
        分层筛选"，不是"挖多少用多少"。

    Args:
        store_dir: 存储根目录（因子库 + K 线库）。
        horizon: 标签预测周期（未来 H 日收益）。
        bars: K 线窗口（0=全历史）。
        top_factors_per_stock: 每股票日线因子 Top-K 筛选（0=全部；高频共享列不参与）。

    Returns:
        FactorDataset：X 列 = {因子列...} ∪ {ret20, vol20}，
        y = 未来 H 日收益。
    """
    store, kstore = FactorStore(store_dir), KlineStore(store_dir)
    factors = store.list_factors()

    # ── 1. 按股票收集因子值（特征选择 + 列名设计）────────────────────
    # 列名规则：
    #   高频因子 → 共享列 "hf_{feature}"（全股票共用，Qlib 横截面特征式）
    #   日线因子 → per-symbol 列 "{symbol}_{hash[:10]}"
    # 认证强度：日线在 report.meta，高频在 report 顶层（双路径读取）
    per_sym: dict[str, list] = {}
    for f in factors:
        sym = f["symbol"]
        rep = f.get("report") or {}
        inner = rep.get("meta") or {}
        kind = rep.get("kind", "")
        fdf = store.load(sym, f["hash"])
        if fdf is None or "factor" not in fdf.columns or fdf.empty:
            continue
        if kind == "highfreq":
            feat = rep.get("feature") or ""
            if not feat:
                continue
            # feature 名已带 hf_ 前缀（如 hf_morning_ratio），直接作共享列
            col = feat
        else:
            col = f"{sym}_{f['hash'][:10]}"
        rankic = float(inner.get("cert_rankic",
                                 rep.get("cert_rankic", 0.0)))
        per_sym.setdefault(sym, []).append((col, rankic, fdf))

    factor_map: dict[str, dict[int, dict[str, float]]] = {}
    factor_cols: list[str] = []
    # M6 修复：每只股票 K 线在收集因子前先「截断 + 清洗」并缓存，因子对齐
    # （无 ts 日线因子按行序对齐）与样本行索引使用同一份清洗后 K 线，消除
    # 清洗/截断导致的因子-交易日错位（历史 bug：因子对齐未清洗 kdf0 尾部，
    # 样本行却用清洗后 kdf 的 t_idx，清洗删行后整体平移）。jump_dates 一并
    # 缓存供 M7 标签掩码使用。
    kdf_cache: dict[str, tuple[pd.DataFrame, set[int]]] = {}
    for sym, cands in per_sym.items():
        # 高频共享列全保留；日线 per-symbol 列按 |认证 RankIC| Top-K
        hf = [c for c in cands if c[0].startswith("hf_")]
        dl = [c for c in cands if not c[0].startswith("hf_")]
        dl.sort(key=lambda c: -abs(c[1]))
        if top_factors_per_stock > 0:
            dl = dl[:top_factors_per_stock]
        keep = hf + dl
        if not keep:
            continue
        bucket = factor_map.setdefault(sym, {})
        kdf_raw = kstore.load(sym, "1d")
        if kdf_raw.empty:
            continue
        if bars > 0 and len(kdf_raw) > bars:
            kdf_raw = kdf_raw.iloc[-bars:]
        kdf, jump_dates = clean_series(kdf_raw)
        if kdf.empty:
            continue
        kdf_cache[sym] = (kdf, jump_dates)
        ts_clean = kdf["ts"].values.astype(np.int64)
        for col, _rankic, fdf in keep:
            if col not in factor_cols:
                factor_cols.append(col)
            vals = fdf["factor"].values.astype(np.float64)
            if "ts" in fdf.columns:
                ts_arr = fdf["ts"].values.astype(np.int64)
                for t, v in zip(ts_arr, vals):
                    if np.isfinite(v):
                        bucket.setdefault(int(t), {})[col] = float(v)
            else:
                # 行序对齐清洗后 K 线尾部：factor[i] ↔ 清洗后 K 线尾部第 i 根
                n = min(len(vals), len(ts_clean))
                ts_tail = ts_clean[-n:]
                for t, v in zip(ts_tail, vals[-n:]):
                    if np.isfinite(v):
                        bucket.setdefault(int(t), {})[col] = float(v)

    # ── 2. K 线清洗 + 风格特征 + 标签 ────────────────────────────────
    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []
    rows_ts: list[int] = []
    rows_sym: list[str] = []
    # 通用特征列（每股票每行都有值，解决单标的因子稀疏问题——Qlib 横截面范式）
    common_cols = ["ret20", "vol20", "mcap_proxy", "turn_proxy", "score"]
    base_n = len(factor_cols) + len(common_cols)
    # 预计算每股票综合得分（IC_IR 加权，复用组合层逻辑的轻量版）
    comp_map = _stock_composite(factor_map, factor_cols)
    for sym, bucket in factor_map.items():
        if not bucket:
            continue
        if sym not in kdf_cache:
            continue
        kdf, jump_dates = kdf_cache[sym]   # 与因子对齐同一份清洗后 K 线（M6）
        close = kdf["close"].values.astype(np.float64)
        vol = kdf["volume"].values.astype(np.float64)
        t_arr = kdf["ts"].values.astype(np.int64)
        t_idx = {int(t): i for i, t in enumerate(t_arr)}
        ret20, vol20 = _causal_style_features(close, vol)
        # 市值代理（成交额滚动）与量能代理（因果）
        amt = close * vol
        mcap_proxy = np.zeros(len(close))
        turn_proxy = np.zeros(len(close))
        for t in range(20, len(close)):
            mcap_proxy[t] = float(np.log(np.mean(amt[t - 19:t + 1]) + 1e-9))
            turn_proxy[t] = float(vol[t] / (np.mean(vol[t - 19:t]) + 1e-9))
        # 标签：未来 H 日收益（因果标签，与挖矿一致）
        label = np.full(len(close), np.nan)
        if len(close) > horizon:
            label[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
        # M7 修复：跳变日收益标签置 0（clean_series 返回的 jump_dates 之前被
        # 丢弃，复权跳变/停牌起复产生的 ±数百收益会作为巨大离群点主导损失）。
        label = mask_jump_returns(label, t_arr, jump_dates)
        if len(close) > horizon:
            # 尾部无未来收益的 NaN 保持 NaN（避免 mask 把尾部伪 0 当真实样本）
            label[-horizon:] = np.nan
        comp = comp_map.get(sym)
        for t, fdict in bucket.items():
            j = t_idx.get(int(t))
            if j is None or j + 1 >= len(close):
                continue
            if not np.isfinite(label[j]):
                continue
            row = np.full(base_n, np.nan)
            for col, v in fdict.items():
                if col in factor_cols:
                    row[factor_cols.index(col)] = v
            # 通用特征（非稀疏）
            off = len(factor_cols)
            row[off] = ret20[j]
            row[off + 1] = vol20[j]
            row[off + 2] = mcap_proxy[j]
            row[off + 3] = turn_proxy[j]
            row[off + 4] = comp[int(t)] if comp is not None \
                and int(t) in comp else np.nan
            rows_x.append(row)
            rows_y.append(float(label[j]))
            rows_ts.append(int(t))
            rows_sym.append(sym)
    if not rows_x:
        return FactorDataset(pd.DataFrame(), pd.Series(dtype=float),
                             np.array([], dtype=np.int64),
                             np.array([], dtype=object),
                             factor_cols + common_cols)

    X = pd.DataFrame(np.vstack(rows_x),
                     columns=factor_cols + common_cols)
    y = pd.Series(rows_y, dtype=np.float64)
    return FactorDataset(
        X=X, y=y,
        ts=np.array(rows_ts, dtype=np.int64),
        symbol=np.array(rows_sym, dtype=object),
        feature_names=list(X.columns),
        meta={"n_factors": len(factor_cols), "horizon": horizon,
              "n_symbols": len(factor_map), "n_samples": len(rows_x)},
    )


def _stock_composite(factor_map: dict, factor_cols: list[str],
                     window: int = 60) -> dict[str, dict[int, float]]:
    """每股票每日综合得分（因子均值 + 近期 IC_IR 加权，轻量版）。

    解决单标的因子稀疏问题：为每股票生成一条「通用信号列」，
    让 GBDT 能学到跨股票结构（对齐 Qlib 横截面范式）。
    性能：窗口内对每个因子用"增量均值 IC"近似 IC_IR（避免逐点 spearmanr）。
    """
    out: dict[str, dict[int, float]] = {}
    for sym, bucket in factor_map.items():
        if not bucket:
            continue
        dates = sorted(bucket.keys())
        mat = np.full((len(dates), len(factor_cols)), np.nan)
        for i, t in enumerate(dates):
            for col, v in bucket[t].items():
                if col in factor_cols:
                    mat[i, factor_cols.index(col)] = v
        comp = np.full(len(dates), np.nan)
        for i in range(window, len(dates)):
            seg = mat[i - window:i]
            w = np.zeros(mat.shape[1])
            for k in range(mat.shape[1]):
                f_k = seg[:, k]
                ok = np.isfinite(f_k)
                if ok.sum() >= 30 and np.std(f_k[ok]) > 1e-12:
                    # 增量 Pearson IC 近似（O(n) 免去滚动 spearmanr）
                    fv = np.nan_to_num(f_k - np.nanmean(f_k))
                    sd = float(np.std(fv))
                    if sd > 1e-12:
                        # 用因子自相关近似的"信号强度"：|IC| ≈ |mean(f)| / std
                        w[k] = abs(float(np.nanmean(f_k))) / sd
            s = np.abs(w).sum()
            if s > 1e-12:
                w /= s
                row = mat[i]
                ok = np.isfinite(row)
                if ok.any():
                    comp[i] = float(np.nansum(row[ok] * w[ok]))
        out[sym] = {dates[i]: comp[i] for i in range(len(dates))
                    if np.isfinite(comp[i])}
    return out


def _dataset_fingerprint(store_dir: str, horizon: int,
                         bars: int, top_factors_per_stock: int) -> str:
    """数据集指纹：因子 hash 集合 + 每股票 K 线 (长度, 末 ts) + 筛选参数。

    任一因子入库/删除、K 线更新或 Top-K 参数变化 → 指纹变化 → 缓存失效。
    """
    import hashlib
    store, kstore = FactorStore(store_dir), KlineStore(store_dir)
    factors = sorted((f["symbol"], f["hash"]) for f in store.list_factors())
    kl = []
    for sym in sorted({s for s, _ in factors}):
        kdf = kstore.load(sym, "1d")
        if not kdf.empty:
            kl.append((sym, len(kdf), int(kdf["ts"].iloc[-1])))
    raw = repr((horizon, bars, top_factors_per_stock, factors, kl))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_dataset_cached(store_dir: str = "store", horizon: int = 5,
                         bars: int = 0,
                         top_factors_per_stock: int = 0,
                         cache_dir: str | None = None,
                         verbose: bool = True) -> FactorDataset:
    """带缓存的 build_dataset（指纹失效）。

    因子库膨胀后 build_dataset 需数十分钟（_stock_composite 全因子重算）；
    本函数以 (因子集合 + K 线状态 + 筛选参数) 指纹缓存数据集，指纹不变时
    秒级加载，供策略工厂反复实验使用。

    Args:
        top_factors_per_stock: 每股票日线因子 Top-K 筛选（0=全部）。
        cache_dir: 缓存目录（默认 {store_dir}/cache/dataset）。
        verbose: 打印缓存命中/重建信息。
    """
    import time as _time
    cache_dir = cache_dir or str(Path(store_dir) / "cache" / "dataset")
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    fp = _dataset_fingerprint(store_dir, horizon, bars,
                              top_factors_per_stock)
    meta_path = cdir / f"ds_{fp}.json"
    npz_path = cdir / f"ds_{fp}.npz"
    if meta_path.exists() and npz_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            data = np.load(npz_path, allow_pickle=True)
            X = pd.DataFrame(data["X"], columns=meta["feature_names"])
            y = pd.Series(data["y"], dtype=np.float64)
            ds = FactorDataset(
                X=X, y=y,
                ts=data["ts"].astype(np.int64),
                symbol=data["symbol"].astype(object),
                feature_names=meta["feature_names"], meta=meta["meta"])
            if verbose:
                print(f"[数据集缓存] 命中 {fp}（{ds.n_samples} 样本 × "
                      f"{len(ds.feature_names)} 特征）")
            return ds
        except Exception:  # noqa: BLE001
            pass
    t0 = _time.time()
    ds = build_dataset(store_dir, horizon=horizon, bars=bars,
                       top_factors_per_stock=top_factors_per_stock)
    if ds.n_samples == 0:
        return ds
    try:
        np.savez_compressed(
            npz_path, X=ds.X.values, y=ds.y.values, ts=ds.ts,
            symbol=ds.symbol)
        meta_path.write_text(json.dumps({
            "fingerprint": fp, "feature_names": ds.feature_names,
            "meta": ds.meta,
        }, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    if verbose:
        print(f"[数据集缓存] 重建 {fp}（{ds.n_samples} 样本 × "
              f"{len(ds.feature_names)} 特征，{_time.time() - t0:.0f}s）")
    return ds
