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
        # M27 修复：分页边界若为包含式，最旧一天会在相邻两页各出现一次 →
        # 按 ts 去重（保留最后一条），避免重复 bar 进入因子计算。
        _dedup: dict[int, Bar] = {}
        for _b in bars:
            _dedup[int(_b.ts)] = _b
        bars = sorted(_dedup.values(), key=lambda b: b.ts)
        if drop_forming and bars:
            now = time.time()
            # M26：日线 ts 已统一为收盘时刻（15:00）→ ts>now 正确剔除形成中 bar；
            # 分钟 ts 为 bar 起始时刻 → 需加 bar 时长判断（旧实现两者都恒假）
            if timeframe == "1d":
                while bars and int(bars[-1].ts) > now:
                    bars.pop()
            else:
                _secs = {"5m": 300, "15m": 900, "30m": 1800,
                         "60m": 3600}.get(timeframe, 300)
                while bars and int(bars[-1].ts) + _secs > now:
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
            # M28：坏行返回 None，跳过（旧实现个别字段缺失/时间格式异常会
            # 抛 KeyError/ValueError 拖垮整次拉取）
            page = [b for b in (self._parse_a_share_row(r) for r in rows)
                    if b is not None]
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
    def _parse_a_share_row(row: dict) -> Bar | None:
        """解析新浪 A股 K 线行；字段缺失/时间格式异常返回 None（调用方跳过）。

        M26：日线 ts 统一为 bar 收盘时刻（A股日线 15:00 CST），与通达信一致，
        使 drop_forming（ts>now 剔除形成中 bar）对开盘时间源也生效——旧实现
        用当天 00:00 开盘时刻，盘中 ts 恒 < now，形成中/未收盘 bar 永不弹出。
        M28：字段用 .get + 数值校验，坏行不抛异常。
        """
        day = str(row.get("day", ""))
        try:
            if " " in day:
                try:
                    ts = int(datetime.strptime(day, "%Y-%m-%d %H:%M:%S")
                             .replace(tzinfo=_CST).timestamp())
                except ValueError:
                    # 分钟行可能缺秒（"2026-08-03 10:05"）
                    ts = int(datetime.strptime(day, "%Y-%m-%d %H:%M")
                             .replace(tzinfo=_CST).timestamp())
            else:
                dt_obj = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=_CST)
                ts = int(dt_obj.replace(hour=15, minute=0, second=0).timestamp())
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
        except (KeyError, ValueError, TypeError):
            return None
        return Bar(ts=ts, open=o, high=h, low=l, close=c,
                   volume=float(row.get("volume", 0.0)))  # 单位：股

    # ── 期货日线（含持仓量）──────────────────────────────────────────────

    def _fetch_fut_daily(self, code: str, n: int) -> list[Bar]:
        """新浪期货日线：接口一次返回全部历史（end 参数实测无效，勿分页）。

        由 fetch_bars 的 [-n:] 负责截断。
        """
        rows = self._get_jsonp(_FUT_DAILY, {"symbol": code}) or []
        if not rows:
            raise DataSourceUnavailable(f"新浪期货日线无数据：{code}")
        return [b for b in (self._parse_fut_row(r) for r in rows)
                if b is not None]

    # ── 期货分钟线（含夜盘）──────────────────────────────────────────────

    def _fetch_fut_min(self, code: str, type_: int, n: int) -> list[Bar]:
        params = {"symbol": code, "type": type_}
        rows = self._get_jsonp(_FUT_MIN, params) or []
        if not rows:
            raise DataSourceUnavailable(f"新浪期货分钟线无数据：{code} type={type_}")
        return [b for b in (self._parse_fut_row(r, minute=True) for r in rows)
                if b is not None]

    @staticmethod
    def _parse_fut_row(row: dict, minute: bool = False) -> Bar | None:
        """解析新浪期货行；字段缺失/格式异常返回 None（M28：坏行不拖垮整次拉取）。"""
        d = str(row.get("d", ""))
        try:
            if minute:
                try:
                    ts = int(datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
                             .replace(tzinfo=_CST).timestamp())
                except ValueError:
                    ts = int(datetime.strptime(d, "%Y-%m-%d %H:%M")
                             .replace(tzinfo=_CST).timestamp())
            else:
                ts = int(datetime.strptime(d[:10], "%Y-%m-%d")
                         .replace(tzinfo=_CST).timestamp())
            o = float(row["o"]); h = float(row["h"])
            l = float(row["l"]); c = float(row["c"])
        except (KeyError, ValueError, TypeError):
            return None
        return Bar(ts=ts, open=o, high=h, low=l, close=c,
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
