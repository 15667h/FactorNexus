"""腾讯财经数据源（A股 / 指数 / ETF）— 免费行情 + 历史K线（支持前/后复权）。

接口（2026-08 实测锁定）：
  实时行情:  https://qt.gtimg.cn/q=sh600519,sz000001
             返回 v_code="1~名称~代码~最新价~昨收~今开~...~"（~ 分隔文本，GBK 编码）
  历史K线:   https://ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,{start},{end},{count},{fq}
             返回 JSON；复权键 qfqday / hfqday / day
  ⚠️ K线行序为 [date, open, close, high, low, vol] —— 与常规 OHLC 不同，本类内部重排
  分钟K线:   https://ifzq.gtimg.cn/appstock/app/kline/mkline?param={code},m5,,{count}
             返回 JSON；键 m1/m5/m15/m30/m60；时间戳 YYYYMMDDHHMM
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import requests

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable
from web.data_sources.code_map import normalize_code, A_SHARE_PRESETS

_BASE_QUOTE = "https://qt.gtimg.cn/q="
_BASE_KLINE = "https://ifzq.gtimg.cn/appstock/app/fqkline/get"
_BASE_MKLINE = "https://ifzq.gtimg.cn/appstock/app/kline/mkline"

# 项目周期 -> 腾讯 kline 周期参数
_TF_TO_KLINE = {
    "5m": "m5", "15m": "m15", "30m": "m30", "60m": "m60",
    "1d": "day", "1w": "week", "1M": "month",
}
_TF_TO_MKLINE = {"1m": "m1", "5m": "m5", "15m": "m15", "30m": "m30", "60m": "m60"}

# 腾讯行情字段索引（split('~')）—— 实测锁定
F_PRICE = 3       # 最新价
F_PREV_CLOSE = 4  # 昨收
F_OPEN = 5        # 今开
F_TS = 30         # 时间戳 YYYYMMDDHHMMSS
F_CHG = 31        # 涨跌额
F_CHG_PCT = 32    # 涨跌幅 %
F_HIGH = 33       # 最高
F_LOW = 34        # 最低
F_VOL = 36        # 成交量（手）
F_AMOUNT = 37     # 成交额（万元）
F_TURNOVER = 38   # 换手率 %
F_MCAP = 44       # 总市值（亿）
F_FLOAT_MCAP = 45 # 流通市值（亿）
F_LIMIT_UP = 47   # 涨停价
F_LIMIT_DOWN = 48 # 跌停价

_CST = timezone(timedelta(hours=8))


class TencentSource(DataSource):
    kind = "tencent"
    label = "腾讯财经"

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    # ── DataSource 接口 ───────────────────────────────────────────────────

    def available(self) -> tuple[bool, str]:
        return (True, "免费行情 · A股/指数/ETF · 支持复权")

    def supported_timeframes(self) -> list[str]:
        return list(dict.fromkeys(list(_TF_TO_MKLINE) + list(_TF_TO_KLINE)))

    def preset_symbols(self) -> list[str]:
        return list(A_SHARE_PRESETS)

    # ── 历史 K 线 ────────────────────────────────────────────────────────

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True,
        adjust: str = "qfq",
    ) -> list[Bar]:
        """拉取 K 线。

        adjust: "qfq"(前复权, 默认) / "hfq"(后复权) / "raw"(不复权)。
        腾讯返回不复权数据键为 "day"，复权为 "qfqday"/"hfqday"。
        """
        norm = normalize_code(symbol)
        if norm.is_futures:
            raise DataSourceUnavailable("腾讯财经不支持国内期货，请用新浪期货源")
        code = norm.tencent

        if timeframe in _TF_TO_MKLINE:
            return self._fetch_minute(code, _TF_TO_MKLINE[timeframe], n, drop_forming)
        if timeframe in _TF_TO_KLINE:
            return self._fetch_daily(code, _TF_TO_KLINE[timeframe], n, drop_forming, adjust)
        raise DataSourceUnavailable(f"腾讯源不支持周期 {timeframe}")

    def _fetch_daily(
        self, code: str, klt: str, n: int, drop_forming: bool, adjust: str
    ) -> list[Bar]:
        fq = {"qfq": "qfq", "hfq": "hfq", "raw": ""}[adjust]
        # 腾讯单次最多 ~640 根，按需翻页（倒序拉取最近 n 根）
        bars: list[Bar] = []
        need = n
        end = ""
        for _ in range(40):  # 最多 40 页
            params = {"param": f"{code},{klt},{''},{end},{min(need, 640)},{fq}"}
            data = self._get_json(_BASE_KLINE, params)
            node = ((data.get("data") or {}).get(code) or {})
            rows = node.get("qfqday") or node.get("hfqday") or node.get("day") or []
            if not rows:
                break
            page_bars = [self._parse_daily_row(r) for r in rows]
            bars = page_bars + bars
            first_date = rows[0][0]
            if len(page_bars) < need:
                end = first_date  # 继续向前翻
                need -= len(page_bars)
            else:
                break
        if not bars:
            raise DataSourceUnavailable(f"腾讯无K线数据：{code}")
        bars.sort(key=lambda b: b.ts)
        # M27：分页边界可能包含式重复最旧一天 → 按 ts 去重（保留最后一条）
        _dedup: dict[int, Bar] = {}
        for _b in bars:
            _dedup[int(_b.ts)] = _b
        bars = sorted(_dedup.values(), key=lambda b: b.ts)
        if drop_forming and bars:
            now = time.time()
            while bars and int(bars[-1].ts) > now:
                bars.pop()
        return bars[-n:]

    @staticmethod
    def _parse_daily_row(row: list) -> Bar:
        # 腾讯行序: [date, open, close, high, low, volume] —— open/close 顺序与常规不同！
        date_s, o_s, c_s, h_s, l_s, v_s = (row + [""] * 6)[:6]
        # M26：统一 ts 为 bar 收盘时刻（A股日线 15:00 CST），与通达信一致，
        # 使 drop_forming（ts>now 剔除）对开盘时间源也生效——旧实现用当天
        # 00:00 开盘时刻，盘中 ts 恒 < now，形成中/未收盘 bar 永不弹出。
        dt_obj = datetime.strptime(str(date_s), "%Y-%m-%d").replace(tzinfo=_CST)
        ts = int(dt_obj.replace(hour=15, minute=0, second=0).timestamp())
        return Bar(ts=ts, open=float(o_s), high=float(h_s), low=float(l_s),
                   close=float(c_s), volume=float(v_s))

    def _fetch_minute(self, code: str, mkey: str, n: int, drop_forming: bool) -> list[Bar]:
        params = {"param": f"{code},{mkey},,{max(n, 20)}"}
        data = self._get_json(_BASE_MKLINE, params)
        node = ((data.get("data") or {}).get(code) or {})
        rows = node.get(mkey) or []
        if not rows:
            raise DataSourceUnavailable(f"腾讯分钟K线无数据：{code} {mkey}")
        bars: list[Bar] = []
        for r in rows:
            # 行序: [datetime(YYYYMMDDHHMM), open, close, high, low, volume, {}, extra]
            ts_s, o_s, c_s, h_s, l_s, v_s = (r + ["", "", "", "", ""])[:6]
            try:
                ts = int(datetime.strptime(str(ts_s), "%Y%m%d%H%M").replace(tzinfo=_CST).timestamp())
            except ValueError:
                continue
            bars.append(Bar(ts=ts, open=float(o_s), high=float(h_s),
                            low=float(l_s), close=float(c_s),
                            volume=float(v_s) if v_s else 0.0))
        bars.sort(key=lambda b: b.ts)
        if drop_forming and bars:
            now = time.time()
            # M26：分钟 ts 为 bar 起始时刻，形成中 bar 判断须加 bar 时长
            # （旧实现 ts>now 恒假，最后一根未收盘 bar 泄漏进数据）
            _secs = {"m1": 60, "m5": 300, "m15": 900, "m30": 1800,
                     "m60": 3600}.get(mkey, 3600)
            while bars and int(bars[-1].ts) + _secs > now:
                bars.pop()
        return bars[-n:]

    # ── 实时行情（含涨跌停价 / 市值，供 amkt.py 使用）────────────────────

    def quote(self, symbols: list[str]) -> dict[str, dict]:
        """批量实时行情。返回 {tencent_code: {price, limit_up, limit_down, ...}}。

        ⚠️ 腾讯接口返回 GBK 编码文本，必须按 GBK 解码，否则中文名称乱码。
        """
        codes = [normalize_code(s).tencent for s in symbols]
        out: dict[str, dict] = {}
        # M25 修复：腾讯返回纯代码（f[2]），市场前缀须用请求时确认的——
        # 旧实现 normalize_code(f[2]) 按首字符推断市场，上证指数 000001 会被
        # 推断成 sz000001，与请求的 sh000001 失配（键错误、指数当深市股票）。
        req_pure: dict[str, str] = {}
        for c in codes:
            req_pure.setdefault(normalize_code(c).code, c)
        for i in range(0, len(codes), 60):  # 每批 ≤60 只
            chunk = codes[i:i + 60]
            resp = self._session.get(_BASE_QUOTE + ",".join(chunk), timeout=self._timeout)
            resp.raise_for_status()
            text = resp.content.decode("gbk", errors="replace")
            for line in text.split(";"):
                line = line.strip()
                if not line.startswith("v_"):
                    continue
                body = line.split('="', 1)[-1].rstrip('";')
                f = body.split("~")
                if len(f) < 50 or not f[F_PRICE]:
                    continue
                # 腾讯返回纯代码（600519）→ 用请求时确认的前缀还原（M25）
                code = req_pure.get(f[2], normalize_code(f[2]).tencent)
                out[code] = {
                    "name": f[1],
                    "price": _f(f, F_PRICE),
                    "prev_close": _f(f, F_PREV_CLOSE),
                    "open": _f(f, F_OPEN),
                    "high": _f(f, F_HIGH),
                    "low": _f(f, F_LOW),
                    "volume": _f(f, F_VOL),          # 手
                    "amount": _f(f, F_AMOUNT),       # 万元
                    "turnover": _f(f, F_TURNOVER),   # %
                    "mcap": _f(f, F_MCAP),           # 亿元
                    "float_mcap": _f(f, F_FLOAT_MCAP),
                    "limit_up": _f(f, F_LIMIT_UP),
                    "limit_down": _f(f, F_LIMIT_DOWN),
                    "ts": f[F_TS],
                }
        return out

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _get_json(self, url: str, params: dict) -> dict:
        resp = self._session.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError as exc:
            raise DataSourceUnavailable(f"腾讯接口返回非 JSON: {resp.text[:120]}") from exc

    def connect(self) -> None:  # 无状态 HTTP，无需连接
        pass

    def disconnect(self) -> None:
        self._session.close()


def _f(fields: list[str], idx: int) -> float:
    try:
        return float(fields[idx])
    except (IndexError, TypeError, ValueError):
        return 0.0
