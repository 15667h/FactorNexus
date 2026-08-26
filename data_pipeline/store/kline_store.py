"""
data_pipeline/store/ — 本地四库分层存储（K线库 / 因子库 / 标签库 / 元数据）

存储约定：
  K线库   store/kline/{code}_{tf}.parquet    # 后复权 OHLCV，按 code 分区
  因子库   store/factors/{symbol}_{hash}.parquet  # 因子矩阵 + 元数据
  标签库   store/labels/{symbol}_label.parquet     # 未来收益标签
  元数据   store/meta/  （trade_days.json / quote_{date}.json / factors_index.parquet）
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STORE_DIR = Path("store")
DEFAULT_KLINE_DIR = "kline"
DEFAULT_FACTOR_DIR = "factors"
DEFAULT_LABEL_DIR = "labels"


# ── K 线库 ───────────────────────────────────────────────────────────────

class KlineStore:
    """K线库：Parquet 分区存储 + 增量更新（幂等，无重复 bar）。"""

    def __init__(self, store_dir: str | Path | None = None) -> None:
        self.root = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self.kline_dir = self.root / DEFAULT_KLINE_DIR
        self.kline_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, code: str, tf: str) -> Path:
        return self.kline_dir / f"{code}_{tf}.parquet"

    def exists(self, code: str, tf: str) -> bool:
        return self._path(code, tf).exists()

    def load(self, code: str, tf: str) -> pd.DataFrame:
        """读取整库（升序）。列: ts, open, high, low, close, volume[, oi]"""
        p = self._path(code, tf)
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p)

    def update(self, code: str, tf: str, df: pd.DataFrame,
               source: str | None = None, adjust: str | None = None) -> pd.DataFrame:
        """增量更新：按 ts 去重合并，原子写回。返回合并后的完整库。

        复权口径冲突防护（2026-08-26 审计）：腾讯 qfq（复权价）与通达信/新浪
        （不复权价）价格量级可差数倍——若混入同一库，价格会在两个口径间
        来回跳变（实测茅台 -85%/+590%），彻底污染因子计算。
        检测公共日期逐日价格比：偏离 1 超过 2x 或 0.5x 的日期占比 >10%，
        或中位数偏离超 2x/0.5x → 口径冲突，以新数据整体覆盖（新源为准）。
        至少 3 个公共日期即可检测（旧实现要求 10 个，会放过小重叠混入）。

        source/adjust: 数据来源（tencent/sina/tongdaxin）与复权口径（qfq/hfq/raw），
        写入 store/meta/kline_sources.json 供库健康审计与数据溯源。
        """
        if df.empty:
            return self.load(code, tf)
        incoming = df.copy()
        incoming["ts"] = incoming["ts"].astype("int64")
        incoming = incoming.sort_values("ts").drop_duplicates(subset="ts", keep="last")
        existing = self.load(code, tf)
        merged = incoming
        conflict = False
        if not existing.empty:
            common = np.intersect1d(existing["ts"].values, incoming["ts"].values)
            if len(common) >= 3:
                ex_m = existing.set_index("ts").loc[common, "close"].astype(float)
                in_m = incoming.set_index("ts").loc[common, "close"].astype(float)
                ratio = in_m.values / np.maximum(ex_m.values, 1e-9)
                med_ratio = float(np.median(ratio))
                # 异常比例法：>2x/<0.5x 的公共日期占比（中位数可能被交替跳变掩盖）
                bad_ratio = float((np.abs(np.log(ratio)) > np.log(2.0)).mean())
                if bad_ratio > 0.10 or med_ratio > 2.0 or med_ratio < 0.5:
                    merged = incoming  # 复权口径冲突 → 新数据整体覆盖
                    conflict = True
                else:
                    merged = pd.concat([existing, incoming], ignore_index=True)
                    merged = merged.sort_values("ts").drop_duplicates(
                        subset="ts", keep="last")
            else:
                merged = pd.concat([existing, incoming], ignore_index=True)
                merged = merged.sort_values("ts").drop_duplicates(
                    subset="ts", keep="last")
        p = self._path(code, tf)
        tmp = p.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(p)
        if source or adjust:
            self._note_source(code, tf, merged, source=source, adjust=adjust,
                              conflict=conflict)
        return merged

    # ── 数据溯源元数据（机构 D3：可审计）────────────────────────────────

    def _source_meta_path(self) -> Path:
        return self.root / "meta" / "kline_sources.json"

    def _load_sources(self) -> dict:
        p = self._source_meta_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_sources(self, meta: dict) -> None:
        p = self._source_meta_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(p)

    def _note_source(self, code: str, tf: str, merged: pd.DataFrame,
                     source: str | None, adjust: str | None,
                     conflict: bool) -> None:
        meta = self._load_sources()
        key = f"{code}_{tf}"
        meta[key] = {
            "source": source, "adjust": adjust,
            "bars": int(len(merged)),
            "first_ts": int(merged["ts"].iloc[0]) if len(merged) else None,
            "last_ts": int(merged["ts"].iloc[-1]) if len(merged) else None,
            "conflict_overwrite": conflict,
            "updated_ts": int(time.time()),
        }
        self._save_sources(meta)

    def source_info(self, code: str, tf: str) -> dict | None:
        """查询某标的 K 线数据来源元数据（无记录返回 None）。"""
        return self._load_sources().get(f"{code}_{tf}")

    def audit_kline(self, code: str, tf: str) -> dict:
        """单标的 K 线健康审计：健康检查 + 跳变分类 + 来源元数据。"""
        from data_pipeline.quality import check_series, classify_jumps

        df = self.load(code, tf)
        if df.empty:
            return {"code": code, "tf": tf, "bars": 0, "clean": True,
                    "issues": ["无数据"]}
        issues = check_series(df)
        kind, jinfo = classify_jumps(df)
        src = self.source_info(code, tf) or {}
        return {
            "code": code, "tf": tf, "bars": int(len(df)),
            "clean": not issues and kind == "clean",
            "issues": issues, "jump_kind": kind, "jump": jinfo,
            "source": src.get("source"), "adjust": src.get("adjust"),
        }

    def audit_all(self, tf: str = "1d") -> dict:
        """全库健康审计：扫描所有 K 线文件，返回统计与污染清单。

        Returns:
            {"total": 文件数, "dirty": 污染文件数,
             "polluted": [审计详情, ...], "by_source": {source: 文件数}}
        """
        out = []
        for p in sorted(self.kline_dir.glob(f"*_{tf}.parquet")):
            code = p.stem.rpartition("_")[0]
            out.append(self.audit_kline(code, tf))
        polluted = [a for a in out if not a["clean"]]
        by_source: dict = {}
        for a in out:
            s = a.get("source") or "unknown"
            by_source[s] = by_source.get(s, 0) + 1
        return {"total": len(out), "dirty": len(polluted),
                "polluted": polluted, "by_source": by_source}

    def list_cached(self) -> list[dict]:
        out = []
        for p in self.kline_dir.glob("*.parquet"):
            code, _, tf = p.stem.rpartition("_")
            out.append({"code": code, "timeframe": tf, "path": str(p)})
        return out


# ── 因子库 ───────────────────────────────────────────────────────────────

class FactorStore:
    """因子库：因子矩阵 + 完整元数据（公式/版本/评分），DuckDB 索引 JSON。"""

    def __init__(self, store_dir: str | Path | None = None) -> None:
        self.root = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self.factor_dir = self.root / DEFAULT_FACTOR_DIR
        self.factor_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "meta" / "factors_index.json"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def factor_hash(formula: list | tuple, vocab_version: str) -> str:
        """因子唯一标识：公式 + 词表版本的 sha256 前缀。"""
        raw = json.dumps([int(t) for t in formula], separators=(",", ":")) + "|" + vocab_version
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def save(self, symbol: str, formula: list, vocab_version: str,
             factor_df: pd.DataFrame, report: dict | None = None) -> str:
        """保存因子矩阵与元数据，返回因子哈希。"""
        fh = self.factor_hash(formula, vocab_version)
        p = self.factor_dir / f"{symbol}_{fh}.parquet"
        tmp = p.with_suffix(".parquet.tmp")
        factor_df.to_parquet(tmp, index=False)
        tmp.replace(p)

        meta = {
            "symbol": symbol, "hash": fh, "formula": [int(t) for t in formula],
            "vocab_version": vocab_version,
            "path": str(p), "report": report or {},
        }
        index = self._load_index()
        index[f"{symbol}_{fh}"] = meta
        self._save_index(index)
        return fh

    def load(self, symbol: str, fh: str) -> pd.DataFrame | None:
        p = self.factor_dir / f"{symbol}_{fh}.parquet"
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def _load_index(self) -> dict:
        if not self.index_path.exists():
            return {}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_index(self, index: dict) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.index_path)

    def list_factors(self) -> list[dict]:
        return list(self._load_index().values())


# ── 标签库 ───────────────────────────────────────────────────────────────

class LabelStore:
    """标签库：未来收益标签（如 5 日收益），与因子库同构存储。"""

    def __init__(self, store_dir: str | Path | None = None) -> None:
        self.root = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self.label_dir = self.root / DEFAULT_LABEL_DIR
        self.label_dir.mkdir(parents=True, exist_ok=True)

    def save(self, symbol: str, horizon: int, label_df: pd.DataFrame) -> None:
        p = self.label_dir / f"{symbol}_label_{horizon}d.parquet"
        tmp = p.with_suffix(".parquet.tmp")
        label_df.to_parquet(tmp, index=False)
        tmp.replace(p)

    def load(self, symbol: str, horizon: int) -> pd.DataFrame | None:
        p = self.label_dir / f"{symbol}_label_{horizon}d.parquet"
        if not p.exists():
            return None
        return pd.read_parquet(p)
