"""TradingView 数据源（tvdatafeed，匿名可用）。"""
from __future__ import annotations

import threading

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

_TF = {
    "1m": "in_1_minute",
    "5m": "in_5_minute",
    "15m": "in_15_minute",
    "30m": "in_30_minute",
    "1h": "in_1_hour",
    "4h": "in_4_hour",
    "1d": "in_daily",
    "1w": "in_weekly",
    "1M": "in_monthly",
}

# 自动探测交易所（symbol 未显式带 EXCHANGE: 前缀时）
_PROBE_EXCHANGES = ["", "OANDA", "FX_IDC", "TVC", "NASDAQ", "NYSE", "SSE", "SZSE", "BINANCE"]

_PRESETS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "NASDAQ:AAPL", "SSE:600519", "BINANCE:BTCUSDT"]


class TradingViewSource(DataSource):
    kind = "tradingview"
    label = "TradingView"

    def __init__(self) -> None:
        self._tv = None
        self._lock = threading.Lock()
        self._exchange_cache: dict[str, str] = {}

    def available(self) -> tuple[bool, str]:
        try:
            import tvDatafeed  # noqa: F401
        except ImportError:
            return (
                False,
                "未安装 tvdatafeed：pip install git+https://github.com/rongardF/tvdatafeed.git",
            )
        return (True, "匿名访问 · 交易所自动探测")

    def supported_timeframes(self) -> list[str]:
        return list(_TF.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def connect(self) -> None:
        if self._tv is not None:
            return
        try:
            from tvDatafeed import TvDatafeed
        except ImportError as exc:
            raise DataSourceUnavailable("未安装 tvdatafeed") from exc
        try:
            self._tv = TvDatafeed()
            try:
                setattr(self._tv, "_TvDatafeed__ws_timeout", 10.0)
            except Exception:
                pass
        except Exception as exc:
            raise DataSourceUnavailable(f"TradingView 连接失败: {exc}") from exc

    def _split_symbol(self, symbol: str) -> tuple[str, str | None]:
        s = symbol.strip()
        if ":" in s:
            ex, code = s.split(":", 1)
            return code.strip(), ex.strip().upper()
        return s, None

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        if timeframe not in _TF:
            raise DataSourceUnavailable(f"TradingView 不支持周期 {timeframe}")
        try:
            from tvDatafeed import Interval
        except ImportError as exc:
            raise DataSourceUnavailable("未安装 tvdatafeed") from exc

        code, exchange = self._split_symbol(symbol)
        interval = getattr(Interval, _TF[timeframe])
        want = n + 1 if drop_forming else n

        with self._lock:
            self.connect()
            df = None
            if exchange is not None:
                df = self._get_hist(code, exchange, interval, want)
            else:
                cached = self._exchange_cache.get(symbol)
                probe = [cached] + _PROBE_EXCHANGES if cached else _PROBE_EXCHANGES
                for ex in probe:
                    df = self._get_hist(code, ex, interval, want)
                    if df is not None and not df.empty:
                        self._exchange_cache[symbol] = ex
                        break

        if df is None or df.empty:
            raise DataSourceUnavailable(
                f"TradingView 无数据：{symbol}（可用 EXCHANGE:CODE 指定交易所）"
            )

        rows = list(df.itertuples(index=True))
        if drop_forming and len(rows) > 1:
            rows = rows[:-1]

        bars: list[Bar] = []
        for row in rows:
            dt = row.Index
            ts = int(dt.timestamp()) if hasattr(dt, "timestamp") else 0
            bars.append(
                Bar(
                    ts=ts,
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(getattr(row, "volume", 0.0) or 0.0),
                )
            )
        return bars

    def _get_hist(self, code, exchange, interval, n):
        try:
            return self._tv.get_hist(
                symbol=code, exchange=exchange, interval=interval, n_bars=n
            )
        except Exception:
            return None
        finally:
            # tvdatafeed 每次 get_hist 都新建 socket 且不关闭，主动关掉防泄漏
            ws = getattr(self._tv, "ws", None)
            if ws is not None:
                try:
                    ws.close()
                    self._tv.ws = None
                except Exception:
                    pass
