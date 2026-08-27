"""新浪财经数据源 — A股/指数历史K线 + 国内期货（主力连续）日线/分钟线。

接口（2026-08 实测锁定）：
  A股历史K线:
    https://money.finance.sina.com.cn/quotes_service/api/jsonp_v2.php/var%20_=/CN_MarketData.getKLineData?symbol=sh600519&scale=240&ma=no&datalen=1023
    返回 JSONP: var _=([{"day","open","high","low","close","volume"},...])
    scale: 5/15/30/60/240(日线)；datalen≤1023，需按日期分页；⚠️ 不复权（需 adjust.py）
    需 Referer: https://finance.sina.com.cn（2022 年起校验）

  期货日线（主力连续，2009 年至今全历史）:
    https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_=/InnerFuturesNewService.getDailyKLine?symbol=RB0
    返回 [{d,o,h,l,c,v,p,s},...]，p=持仓量

  期货分钟线（含夜盘）:
    https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_=/InnerFuturesNewService.getFewMinLine?symbol=RB0&type=5
    type: 5/15/30/60；返回 [{d:"2026-08-03 10:05:00",o,h,l,c,v,p},...]
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone, timedelta

import requests

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable
from web.data_sources.code_map import normalize_code, A_SHARE_PRESETS, FUTURES_PRESETS

_A_SHARE_KLINE = (
    "https://money.finance.sina.com.cn/quotes_service/api/jsonp_v2.php/"
    "var%20_=/CN_MarketData.getKLineData"
)
_FUT_DAILY = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
    "var%20_=/InnerFuturesNewService.getDailyKLine"
)
_FUT_MIN = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
    "var%20_=/InnerFuturesNewService.getFewMinLine"
)

# A股: 项目周期 -> scale；期货: type
_TF_TO_SCALE = {"5m": 5, "15m": 15, "30m": 30, "60m": 60, "1d": 240}
_TF_TO_FUT_TYPE = {"5m": 5, "15m": 15, "30m": 30, "60m": 60, "1d": 0}  # 0=日线走 getDailyKLine

_CST = timezone(timedelta(hours=8))
# 新浪 JSONP 实际格式: /*<script>...</script>*/\nvar _=([{...}]); （括号包裹数组）
_JSONP_RE = re.compile(r"var\s+_\s*=\s*(.*?);?\s*$", re.S)

_FUT_DAILY_PAGE = 2000  # 保留常量（兼容引用）；期货日线接口不支持分页，一次返回全部


class SinaSource(DataSource):
    kind = "sina"
    label = "新浪财经"

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            # 2022 年起新浪对 hq 类接口做 Referer 校验（历史K线接口同样带上）
            "Referer": "https://finance.sina.com.cn",
        })

    # ── DataSource 接口 ───────────────────────────────────────────────────

    def available(self) -> tuple[bool, str]:
        return (True, "A股/指数历史K线 · 国内期货主力连续（含持仓量）")

    def supported_timeframes(self) -> list[str]:
        return ["5m", "15m", "30m", "60m", "1d"]

    def preset_symbols(self) -> list[str]:
        return list(A_SHARE_PRESETS) + list(FUTURES_PRESETS)

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True,
        adjust: str = "raw",   # 新浪为不复权数据，接收但忽略（接口统一）
    ) -> list[Bar]:
        norm = normalize_code(symbol)
        if norm.is_futures:
            if timeframe not in _TF_TO_FUT_TYPE:
                raise DataSourceUnavailable(f"新浪期货源不支持周期 {timeframe}")
            if timeframe == "1d":
                bars = self._fetch_fut_daily(norm.sina_futures, n)
            else:
                bars = self._fetch_fut_min(norm.sina_futures, _TF_TO_FUT_TYPE[timeframe], n)
        else:
            if timeframe not in _TF_TO_SCALE:
                raise DataSourceUnavailable(f"新浪A股源不支持周期 {timeframe}")
            bars = self._fetch_a_share(norm.sina, _TF_TO_SCALE[timeframe], n)
        bars.sort(key=lambda b: b.ts)
        if drop_forming and bars:
            now = time.time()
            while bars and int(bars[-1].ts) > now:
                bars.pop()
        return bars[-n:]

    # ── A股历史 K 线（scale=240 为日线，无复权）────────────────────────────

    def _fetch_a_share(self, code: str, scale: int, n: int) -> list[Bar]:
        # datalen≤1023，按日期向前翻页
        bars: list[Bar] = []
        end_date = ""
        need = n
        for _ in range(30):
            params = {"symbol": code, "scale": scale, "ma": "no", "datalen": min(need, 1023)}
            if end_date:
                params["end_date"] = end_date
            payload = self._get_jsonp(_A_SHARE_KLINE, params)
            if not payload:
                break
            rows = payload if isinstance(payload, list) else []
            if not rows:
                break
            page = [self._parse_a_share_row(r) for r in rows]
            bars = page + bars
            if len(rows) < min(need, 1023):
                break
            need -= len(page)
            end_date = str(rows[0].get("day", ""))  # 最旧一天
            if not end_date:
                break
        if not bars:
            raise DataSourceUnavailable(f"新浪A股无K线数据：{code}")
        return bars

    @staticmethod
    def _parse_a_share_row(row: dict) -> Bar:
        day = str(row.get("day", ""))
        # 日线 "2026-08-26" 或分钟线 "2026-08-26 10:00:00"（带时间）
        if " " in day:
            ts = int(datetime.strptime(day, "%Y-%m-%d %H:%M:%S")
                     .replace(tzinfo=_CST).timestamp())
        else:
            ts = int(datetime.strptime(day, "%Y-%m-%d")
                     .replace(tzinfo=_CST).timestamp())
        return Bar(ts=ts,
                   open=float(row["open"]), high=float(row["high"]),
                   low=float(row["low"]), close=float(row["close"]),
                   volume=float(row.get("volume", 0.0)))  # 单位：股

    # ── 期货日线（含持仓量）──────────────────────────────────────────────

    def _fetch_fut_daily(self, code: str, n: int) -> list[Bar]:
        """新浪期货日线：接口一次返回全部历史（end 参数实测无效，勿分页）。

        由 fetch_bars 的 [-n:] 负责截断。
        """
        rows = self._get_jsonp(_FUT_DAILY, {"symbol": code}) or []
        if not rows:
            raise DataSourceUnavailable(f"新浪期货日线无数据：{code}")
        return [self._parse_fut_row(r) for r in rows]

    # ── 期货分钟线（含夜盘）──────────────────────────────────────────────

    def _fetch_fut_min(self, code: str, type_: int, n: int) -> list[Bar]:
        params = {"symbol": code, "type": type_}
        rows = self._get_jsonp(_FUT_MIN, params) or []
        if not rows:
            raise DataSourceUnavailable(f"新浪期货分钟线无数据：{code} type={type_}")
        bars = [self._parse_fut_row(r, minute=True) for r in rows]
        return bars

    @staticmethod
    def _parse_fut_row(row: dict, minute: bool = False) -> Bar:
        d = str(row.get("d", ""))
        if minute:
            ts = int(datetime.strptime(d, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_CST).timestamp())
        else:
            ts = int(datetime.strptime(d[:10], "%Y-%m-%d").replace(tzinfo=_CST).timestamp())
        return Bar(ts=ts,
                   open=float(row["o"]), high=float(row["h"]),
                   low=float(row["l"]), close=float(row["c"]),
                   volume=float(row.get("v", 0.0)),
                   extra={"oi": float(row["p"])} if row.get("p") not in (None, "", "0.000") else None)

    # ── JSONP 解析 ────────────────────────────────────────────────────────

    def _get_jsonp(self, url: str, params: dict):
        resp = self._session.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        text = resp.text.strip()
        m = _JSONP_RE.search(text)
        body = m.group(1).strip() if m else text
        # 剥掉外层括号: ([{...}]) -> [{...}]
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1]
        if body in ("null", "undefined"):
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise DataSourceUnavailable(f"新浪接口返回非 JSONP: {text[:120]}") from exc

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        self._session.close()
