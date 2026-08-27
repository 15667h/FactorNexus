"""通达信数据源（pytdx，免费行情服务器，A 股 / 指数）。"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone, timedelta

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

# 通达信行情服务器（多个备选）
_SERVERS = [
    ("115.238.90.165", 7709),
    ("180.153.18.170", 7709),
    ("119.147.212.81", 7709),
    ("14.17.75.71", 7709),
    ("59.173.18.77", 7709),
]

# 项目周期 -> pytdx category
_CAT = {
    "1m": 8,
    "5m": 0,
    "15m": 1,
    "30m": 2,
    "1h": 3,
    "1d": 9,
    "1w": 5,
    "1M": 6,
}

_PRESETS = ["600519", "000001", "300750", "601318", "000858", "sh000001", "sz399006"]


def _parse_market(code: str) -> tuple[int, str]:
    """返回 (market, pure_code)。1=上海, 0=深圳。"""
    c = code.strip().upper()
    if c.startswith("SH"):
        return 1, c[2:]
    if c.startswith("SZ"):
        return 0, c[2:]
    if c[:1] in ("6", "5", "9") or c.startswith("11") or c.startswith("13"):
        return 1, c
    return 0, c


_CST = timezone(timedelta(hours=8))  # 通达信返回的 datetime 为北京时间（UTC+8）


def _is_index(market: int, code: str) -> bool:
    """判断是否为指数代码。

    上证指数系列：market=1 且 code 以 000 开头（如 000001 上证指数、000300 沪深300）。
    深证指数系列：market=0 且 code 以 399 开头（如 399001 深证成指、399006 创业板指）。
    注意：深圳市场 000xxx 是股票（如 000001 平安银行），只有 399xxx 才是深证指数，
    因此不能用纯 code 前缀判断，必须结合 market。
    """
    if market == 1 and code.startswith("000"):
        return True
    if market == 0 and code.startswith("399"):
        return True
    return False


def _parse_dt(s: str) -> int:
    """通达信 datetime（北京时间，如 '2026-07-15 15:00'）-> 收盘时刻的 Unix 秒。

    返回值语义为「该 bar 的收盘时刻（UTC 秒）」；drop_forming / _ensure_closed_bars
    据此判断是否已收盘（ts > now 即仍在形成中）。
    """
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(str(s), fmt).replace(tzinfo=_CST).timestamp())
        except ValueError:
            continue
    return 0


def _looks_corrupted(raw) -> bool:
    """检测返回数据是否乱码（datetime 年份异常）。

    典型场景：对指数误用 get_security_bars，第 2 条起 datetime 损坏（如 7772-67-85）；
    或纯数字指数代码未带 sh/sz 前缀（如 000300）导致市场误判后取到异常数据。
    """
    if not raw:
        return False
    bad = 0
    for r in raw:
        s = str(r.get("datetime", ""))
        y = int(s[:4]) if len(s) >= 4 and s[:4].isdigit() else -1
        if not (1990 <= y <= 2035):
            bad += 1
    return bad > 0 and bad >= len(raw) * 0.3


class TongdaxinSource(DataSource):
    kind = "tongdaxin"
    label = "通达信"

    def __init__(self) -> None:
        self._api = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        try:
            import pytdx  # noqa: F401
        except ImportError:
            return (False, "未安装 pytdx：pip install pytdx")
        return (True, "免费行情服务器 · A 股 / 指数")

    def supported_timeframes(self) -> list[str]:
        return list(_CAT.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def connect(self) -> None:
        if self._api is not None:
            return
        try:
            from pytdx.hq import TdxHq_API
        except ImportError as exc:
            raise DataSourceUnavailable("未安装 pytdx") from exc
        api = TdxHq_API()
        for host, port in _SERVERS:
            try:
                if api.connect(host, port):
                    self._api = api
                    return
            except Exception:
                continue
        raise DataSourceUnavailable("通达信所有行情服务器连接失败")

    def disconnect(self) -> None:
        if self._api is not None:
            try:
                self._api.disconnect()
            except Exception:
                pass
        self._api = None

    def _fetch_raw(self, cat: int, market: int, code: str, start: int,
                   count: int, is_index: bool):
        """指数走 get_index_bars，股票走 get_security_bars（均支持 start 翻页）。

        通达信协议规定指数必须用 get_index_bars；若对指数用 get_security_bars，
        返回数据从第 2 条起 datetime 会损坏（年份变成 7772、228200 等乱码）。
        """
        if is_index:
            return self._api.get_index_bars(cat, market, code, start, count)
        return self._api.get_security_bars(cat, market, code, start, count)

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True,
        adjust: str = "raw",   # 通达信为不复权数据，接收但忽略（接口统一）
    ) -> list[Bar]:
        """拉取 K 线（支持全历史翻页：单页上限 800，循环 start 向前翻）。

        注意：返回的是不复权数据（pytdx 限制），历史含除权跳空；volume 单位为
        「手」（1手=100股），与 OKX/MT5 的 volume 量纲不同，跨源不可比。
        """
        if timeframe not in _CAT:
            raise DataSourceUnavailable(f"通达信不支持周期 {timeframe}")
        market, code = _parse_market(symbol)
        cat = _CAT[timeframe]
        is_index = _is_index(market, code)

        with self._lock:
            self.connect()
            bars: list[Bar] = []
            start = 0
            need = min(max(n, 20), 100000)
            pages = 0
            while need > 0 and pages < 60:  # 60 页上限 ≈ 48000 根
                want = min(need + 2, 800)
                try:
                    raw = self._fetch_raw(cat, market, code, start, want, is_index)
                except Exception:
                    # 连接可能失效，重连一次（断联重试兜底）
                    self._api = None
                    self.connect()
                    raw = self._fetch_raw(cat, market, code, start, want, is_index)
                if not raw:
                    break
                if _looks_corrupted(raw):
                    # 数据乱码通常意味着接口选错（指数误用股票接口）或代码/市场不匹配。
                    raise DataSourceUnavailable(
                        f"通达信返回数据异常：{symbol}。若为指数请使用 sh/sz 前缀"
                        f"（如 sh000001、sz399006），股票代码请确认无误。"
                    )
                for r in raw:
                    bars.append(
                        Bar(
                            ts=_parse_dt(r.get("datetime", "")),
                            open=float(r["open"]),
                            high=float(r["high"]),
                            low=float(r["low"]),
                            close=float(r["close"]),
                            volume=float(r.get("vol", 0.0) or 0.0),  # 单位：手
                        )
                    )
                if len(raw) < want:
                    break  # 已到最早历史
                start += len(raw)
                need -= len(raw)
                pages += 1

        if not bars:
            raise DataSourceUnavailable(
                f"通达信无数据：{symbol}。请确认代码正确；指数需带 sh/sz 前缀（如 sh000001）。"
            )
        bars.sort(key=lambda b: b.ts)  # 保证升序
        # 剔除尚未收盘的 bar：通达信 datetime 为收盘时刻（北京时间），_parse_dt
        # 返回的 ts 即收盘时刻的 UTC 秒；ts > now 说明该 bar 仍在形成中。
        if drop_forming and bars:
            now = time.time()
            while bars and int(bars[-1].ts) > now:
                bars.pop()
        return bars[-n:]
