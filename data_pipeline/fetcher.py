"""
data_pipeline/fetcher.py — MT5 数据获取模块

通过 MetaTrader5 Python API 连接 MT5 终端并获取历史 OHLCV 数据。
"""

import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None  # type: ignore

# DataFrame 返回列定义
_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]


class MT5DataFetcher:
    """通过 MetaTrader5 Python API 获取历史 OHLCV 数据。

    用法（上下文管理器）：
        with MT5DataFetcher() as fetcher:
            df = fetcher.fetch("XAUUSD", mt5.TIMEFRAME_H1, 2000)

    用法（手动）：
        fetcher = MT5DataFetcher()
        fetcher.connect()
        df = fetcher.fetch("XAUUSD", mt5.TIMEFRAME_H1, 2000)
        fetcher.shutdown()

    离线模式（仅读本地缓存，不连 MT5）：
        with MT5DataFetcher(offline=True) as fetcher:
            df = fetcher.fetch("XAUUSD", mt5.TIMEFRAME_H1, 2000)
    """

    def __init__(self, offline: bool = False) -> None:
        self.offline = offline
        self._mt5_initialized = False

    def connect(self) -> None:
        """连接到 MT5 终端。

        调用 `mt5.initialize()`，若连接失败则抛出 `ConnectionError`。
        离线模式下跳过连接，仅使用本地缓存。

        Raises:
            ConnectionError: MT5 终端未运行或连接失败（非离线模式）。
        """
        if self.offline:
            logger.info("[Fetcher] 离线模式：跳过 MT5 连接，仅使用本地缓存。")
            return

        if not _MT5_AVAILABLE:
            raise ConnectionError("MetaTrader5 package is not installed.")

        success = mt5.initialize()  # type: ignore[union-attr]
        if not success:
            error = mt5.last_error()  # type: ignore[union-attr]
            raise ConnectionError(f"MT5 connection failed: {error}")

        self._mt5_initialized = True
        logger.info("MT5 connection established.")

    def fetch(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        """获取指定品种的历史 OHLCV 数据（优先读本地缓存，增量更新）。

        Args:
            symbol:    MT5 品种标识符，例如 "XAUUSD"。
            timeframe: MT5 时间周期常量，例如 mt5.TIMEFRAME_H1（整数）。
            count:     要获取的 K 线数量。

        Returns:
            包含列 time, open, high, low, close, tick_volume 的 DataFrame。
            若品种不可用，返回空 DataFrame（列名相同）。
        """
        # ── 优先读本地缓存 ────────────────────────────────────────────
        # 使用本地缓存的全部历史数据，不再用 tail(count) 截断。
        # count 仅用于无本地缓存时从 MT5 全量下载的最大根数。
        try:
            from data_pipeline.kline_cache import KlineCache
            cache = KlineCache(timeframe=timeframe, bars_count=count)
            mt5_connected = (
                not self.offline
                and self._mt5_initialized
                and _MT5_AVAILABLE
                and mt5 is not None
            )
            df = cache.get(symbol, mt5_connected=mt5_connected)
            if df is not None and not df.empty:
                # 本地有数据，返回全部历史（不截断）
                return df.reset_index(drop=True)
        except Exception as exc:
            logger.debug(f"[Fetcher] Cache read failed for {symbol}: {exc}, falling back to MT5")

        # ── 缓存不足时从 MT5 直接拉 ──────────────────────────────────
        if self.offline or not _MT5_AVAILABLE or mt5 is None or not self._mt5_initialized:
            logger.warning(f"{'Offline mode' if self.offline else 'MT5 not available'}, "
                           f"returning empty DataFrame for {symbol}.")
            return pd.DataFrame(columns=_COLUMNS)

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)  # type: ignore[union-attr]

        if rates is None or len(rates) == 0:
            logger.warning(
                f"Symbol '{symbol}' returned no data (possibly unavailable). "
                f"MT5 error: {mt5.last_error()}"  # type: ignore[union-attr]
            )
            return pd.DataFrame(columns=_COLUMNS)

        df = pd.DataFrame(rates)[_COLUMNS]
        logger.debug(f"Fetched {len(df)} bars for {symbol} (timeframe={timeframe}) from MT5.")
        return df

    def shutdown(self) -> None:
        """断开与 MT5 终端的连接，释放资源。"""
        if not self.offline and self._mt5_initialized and _MT5_AVAILABLE and mt5 is not None:
            mt5.shutdown()  # type: ignore[union-attr]
            logger.info("MT5 connection closed.")

    # ── 上下文管理器支持 ──────────────────────────────────

    def __enter__(self) -> "MT5DataFetcher":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
