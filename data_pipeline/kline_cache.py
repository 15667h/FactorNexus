"""
data_pipeline/kline_cache.py — 本地 K 线缓存管理器

设计：
  - 每个品种存一个 Parquet 文件：D:/K线数据/{symbol}_H1.parquet
  - 列：time(int64 Unix秒), open, high, low, close, tick_volume
  - 首次：从 MT5 拉全量（BARS_COUNT 根）写入
  - 后续：只拉本地最新 bar 之后的增量，追加
  - 无 MT5 连接时：直接读本地文件（供查询/分析使用）

用法：
    cache = KlineCache()
    df = cache.get(symbol)          # 优先读本地，按需增量更新
    df = cache.get(symbol, force_refresh=True)  # 强制重拉全量
    cache.update_all(symbols)       # 批量更新
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None

try:
    from config import Config
    _TIMEFRAME = Config.TIMEFRAME
    _BARS_COUNT = Config.BARS_COUNT
except ImportError:
    _TIMEFRAME = 16385   # H1
    _BARS_COUNT = 12000


def _default_cache_dir() -> Path:
    try:
        from config import Config
        return Path(getattr(Config, "KLINE_CACHE_DIR", r"D:\K线数据"))
    except ImportError:
        return Path(r"D:\K线数据")

_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]


class KlineCache:
    """本地 K 线缓存管理器，支持增量更新。"""

    def __init__(
        self,
        cache_dir:  str | Path | None = None,
        timeframe:  int         = _TIMEFRAME,
        bars_count: int         = _BARS_COUNT,
    ) -> None:
        self.cache_dir  = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self.timeframe  = timeframe
        self.bars_count = bars_count
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str) -> Path:
        tf_name = {16385: "H1", 16388: "H4", 16408: "D1", 1: "M1", 5: "M5"}.get(
            self.timeframe, f"TF{self.timeframe}"
        )
        return self.cache_dir / f"{symbol}_{tf_name}.parquet"

    # ── 公开接口 ─────────────────────────────────────────────────────────

    def get(
        self,
        symbol:        str,
        force_refresh: bool = False,
        mt5_connected: bool = True,
    ) -> Optional[pd.DataFrame]:
        """获取品种 K 线数据（本地优先，自动增量更新）。

        Args:
            symbol:        MT5 品种名
            force_refresh: True = 忽略本地缓存，强制重拉全量
            mt5_connected: False = 只读本地，不尝试连接 MT5

        Returns:
            DataFrame（time, open, high, low, close, tick_volume），
            或 None（本地无数据且无 MT5 连接时）。
        """
        path = self._cache_path(symbol)

        if force_refresh or not path.exists():
            if mt5_connected and _MT5_AVAILABLE:
                return self._full_download(symbol)
            elif path.exists():
                logger.info(f"[Cache] {symbol}: 读取本地缓存（无 MT5 连接）")
                return pd.read_parquet(path)
            else:
                logger.warning(f"[Cache] {symbol}: 无本地缓存且无 MT5 连接")
                return None

        # 读本地
        local_df = pd.read_parquet(path)
        if local_df.empty:
            return self._full_download(symbol) if mt5_connected else None

        if not mt5_connected or not _MT5_AVAILABLE:
            return local_df

        # 增量更新
        last_time = int(local_df["time"].iloc[-1])
        updated = self._incremental_update(symbol, local_df, last_time)
        return updated

    def update_all(
        self,
        symbols: list[str],
        mt5_connected: bool = True,
    ) -> dict[str, int]:
        """批量更新多个品种，返回 {symbol: new_bars_added}。"""
        results = {}
        for sym in symbols:
            try:
                before = 0
                path = self._cache_path(sym)
                if path.exists():
                    before = len(pd.read_parquet(path))
                df = self.get(sym, mt5_connected=mt5_connected)
                after = len(df) if df is not None else 0
                added = max(0, after - before)
                results[sym] = added
                logger.info(f"[Cache] {sym}: {after} bars total, +{added} new")
            except Exception as exc:
                logger.error(f"[Cache] {sym}: update failed: {exc}")
                results[sym] = -1
        return results

    def list_cached(self) -> list[dict]:
        """列出所有已缓存的品种和数据量。"""
        out = []
        for p in sorted(self.cache_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                last_t = pd.to_datetime(df["time"].iloc[-1], unit="s", utc=True)
                out.append({
                    "file":     p.name,
                    "bars":     len(df),
                    "last_bar": str(last_t),
                    "size_kb":  round(p.stat().st_size / 1024, 1),
                })
            except Exception:
                pass
        return out

    def read_local(self, symbol: str) -> Optional[pd.DataFrame]:
        """直接读本地文件，不尝试 MT5（离线使用）。"""
        return self.get(symbol, mt5_connected=False)

    # ── 内部方法 ─────────────────────────────────────────────────────────

    def _full_download(self, symbol: str) -> Optional[pd.DataFrame]:
        """从 MT5 下载全量历史数据并保存。"""
        if not _MT5_AVAILABLE or mt5 is None:
            return None
        logger.info(f"[Cache] {symbol}: 全量下载（{self.bars_count} bars）...")
        t0 = time.time()
        rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, self.bars_count)
        if rates is None or len(rates) == 0:
            logger.warning(f"[Cache] {symbol}: MT5 返回空数据")
            return None
        df = pd.DataFrame(rates)[_COLUMNS].astype({
            "time": "int64", "open": "float32", "high": "float32",
            "low": "float32", "close": "float32", "tick_volume": "int64",
        })
        path = self._cache_path(symbol)
        df.to_parquet(path, index=False)
        elapsed = time.time() - t0
        logger.info(
            f"[Cache] {symbol}: {len(df)} bars 已保存 → {path.name}  ({elapsed:.1f}s)"
        )
        return df

    def _incremental_update(
        self,
        symbol:    str,
        local_df:  pd.DataFrame,
        last_time: int,
    ) -> pd.DataFrame:
        """拉取 last_time 之后的新 bar，追加写入本地。"""
        if not _MT5_AVAILABLE or mt5 is None:
            return local_df

        # 拉最近 200 根 bar（足够覆盖增量）
        rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, 200)
        if rates is None or len(rates) == 0:
            return local_df

        new_df = pd.DataFrame(rates)[_COLUMNS].astype({
            "time": "int64", "open": "float32", "high": "float32",
            "low": "float32", "close": "float32", "tick_volume": "int64",
        })
        # 只取比 last_time 更新的 bar
        new_rows = new_df[new_df["time"] > last_time]
        if new_rows.empty:
            logger.debug(f"[Cache] {symbol}: 已是最新，无增量")
            return local_df

        merged = pd.concat([local_df, new_rows], ignore_index=True)
        merged = merged.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        path = self._cache_path(symbol)
        merged.to_parquet(path, index=False)
        logger.info(f"[Cache] {symbol}: +{len(new_rows)} 新 bar，共 {len(merged)} bars")
        return merged
