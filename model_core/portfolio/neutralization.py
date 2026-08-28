"""
model_core/portfolio/neutralization.py — 机构级因子中性化（P14）

对标华泰金工《基于量价的人工智能选股体系》：合成因子必须做
「行业、市值、20日收益、20日波动、20日换手」五因子中性化，
剥离系统性风格暴露，获得纯净的个股自身信号。

实现（机构标准）：
  1. 逐日横截面 OLS 回归：factor_t ~ 行业哑变量 + log(市值) + ret20
     + vol20 + turn20，取残差作为中性化因子。
  2. 行业数据：东方财富行业分类（A股全市场，缓存 store/meta/
     industry_map.json，refresh 可强制重拉；失败时降级为单行业
     = 退化为仅风格中性化，并如实报告）。
  3. 风格因子全部因果计算（t 只用 t 及以前数据，防前视）。

用法：
    from model_core.portfolio.neutralization import neutralize_panel, fetch_industry_map
    neutral = neutralize_panel(panel, klines, industry_map, mcap_proxy="close*volume")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── 行业数据（东方财富行业分类）────────────────────────────────────────────

_EASTMONEY_CLIST = "https://push2.eastmoney.com/api/qt/clist/get"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
       "Referer": "https://data.eastmoney.com/"}


def fetch_industry_map(store_dir: str | Path = "store",
                       refresh: bool = False) -> dict[str, str]:
    """获取全 A 股行业归属 {symbol(sh600519): 行业名}。

    东方财富行业板块 → 成分股（f12=板块代码/f14=板块名；成分 f12=股票代码
    f14=股票名）。缓存 store/meta/industry_map.json（24h 有效）。
    失败时返回 {}（调用方降级为单行业中性化）。
    """
    import requests

    root = Path(store_dir)
    cache = root / "meta" / "industry_map.json"
    if not refresh and cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    try:
        session = requests.Session()
        session.headers.update(_UA)
        # 1) 行业板块列表（东财接口偶发断连/风控 → 重试退避，与腾讯/新浪同策略）
        blocks = []
        for attempt in range(3):
            try:
                params = {"pn": 1, "pz": 200, "po": 1, "np": 1, "fltt": 2,
                          "invt": 2, "fid": "f3", "fs": "m:90 t:2",
                          "fields": "f12,f14"}
                r = session.get(_EASTMONEY_CLIST, params=params, timeout=15)
                r.raise_for_status()
                blocks = (r.json().get("data") or {}).get("diff") or []
                if blocks:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1.0 * (1.0 + attempt))
        industry_map: dict[str, str] = {}
        for b in blocks:
            bk, name = b.get("f12"), str(b.get("f14") or "未知")
            if not bk:
                continue
            # 2) 板块成分股
            p2 = {"pn": 1, "pz": 500, "po": 1, "np": 1, "fltt": 2,
                  "invt": 2, "fid": "f3", "fs": f"b:{bk}",
                  "fields": "f12,f14"}
            r2 = session.get(_EASTMONEY_CLIST, params=p2, timeout=15)
            r2.raise_for_status()
            members = (r2.json().get("data") or {}).get("diff") or []
            for m in members:
                code = str(m.get("f12") or "")
                if len(code) != 6:
                    continue
                sym = ("sh" if code[0] in "569" else "sz") + code
                industry_map[sym] = name
            time.sleep(0.15)  # 礼貌限速
        if industry_map:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(industry_map, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(cache)
            return industry_map
    except Exception:  # noqa: BLE001
        pass
    return {}


# ── 风格因子构建（全部因果）────────────────────────────────────────────────

def build_style_features(symbol: str, df: pd.DataFrame,
                         mcap_proxy: str = "close*volume") -> dict:
    """单标的风格特征（最新值）：市值代理、ret20、vol20、turn20。

    mcap_proxy: "close*volume"（默认，无股本数据时用成交额代理市值）
              | "amount"（用成交额）
    全部因果：只依赖 t 及以前数据，取序列末端值。
    """
    close = df["close"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)
    T = len(close)
    if T < 21:
        return {}
    eps = 1e-9
    if mcap_proxy == "amount" and "amount" in df.columns:
        amount = df["amount"].values.astype(np.float64)
    else:
        amount = close * vol
    ret = close[1:] / (close[:-1] + eps) - 1.0
    ret20 = float(close[-1] / (close[-21] + eps) - 1.0)
    vol20 = float(np.std(ret[-20:]))
    # 换手代理：成交额近20日均值 / 长期均值（量能活跃度）
    turn20 = float(np.mean(amount[-20:]) / (np.mean(amount[-60:]) + eps) - 1.0) \
        if T >= 60 else float(np.mean(amount[-20:]))
    mcap = float(np.log(np.mean(amount[-5:]) + eps))  # log 市值代理
    return {"mcap": mcap, "ret20": ret20, "vol20": vol20, "turn20": turn20}


# ── 逐日横截面中性化 ───────────────────────────────────────────────────────

def neutralize_panel(panel: pd.DataFrame, klines: dict[str, pd.DataFrame],
                     industry_map: dict[str, str] | None = None,
                     mcap_proxy: str = "close*volume") -> tuple[pd.DataFrame, dict]:
    """逐日横截面五因子中性化（全因果，防前视）。

    Args:
        panel: 因子面板，index=ts，columns=symbol，值为因子值（NaN 允许）。
        klines: {symbol: K线df}，用于算风格特征（与 panel 同源数据）。
        industry_map: {symbol: 行业}；缺省/空时退化为单行业（仅风格中性化）。
        mcap_proxy: 市值代理方式。

    Returns:
        (neutral_panel, report)
        report: {"n_days", "n_stocks", "industries", "degraded", "r2_mean"}
    """
    if panel.empty:
        return panel, {"n_days": 0, "n_stocks": 0, "industries": 0,
                       "degraded": True, "r2_mean": 0.0}
    symbols = [c for c in panel.columns]
    industry_map = industry_map or {}
    has_industry = bool(industry_map)

    # ── 预取每只股票 K 线并预计算「逐日」风格数组（因果滚动）──────────────
    # 历史 bug（H2 前视）：旧实现用 K 线末端时点快照（close[-1]/ret20[-1]/
    # vol20[-1]/turn20[-1]/mcap[-1]）作为面板所有历史日期的风格——早期日期
    # 被未来才可知的风格信息中性化，系统性高估中性化质量。改为按日因果。
    # 历史 bug（H11 崩溃）：旧实现用 panel 全部列 symbol 访问 ind_dummy，
    # 因子宇宙 ≠ K 线宇宙时 KeyError。改为只对「有 K 线且窗口足够」的标的
    # 参与横截面回归（usable 集合），输出面板保留原列。
    _eps = 1e-9
    styles_daily: dict[str, dict] = {}
    for s in symbols:
        df = klines.get(s)
        if df is None or df.empty:
            continue
        close = df["close"].values.astype(np.float64)
        vol = df["volume"].values.astype(np.float64)
        if mcap_proxy == "amount" and "amount" in df.columns:
            amount = df["amount"].values.astype(np.float64)
        else:
            amount = close * vol
        t_arr = df["ts"].values.astype(np.int64)
        T = len(close)
        if T < 21:
            continue
        ret = np.zeros(T)
        ret[1:] = close[1:] / (close[:-1] + _eps) - 1.0
        # 逐日因果风格（t 只用 t 及以前）——向量化：
        # ret20[t] = close[t]/close[t-20]-1
        ret20 = np.zeros(T)
        ret20[20:] = close[20:] / (close[:-20] + _eps) - 1.0
        # vol20[t] = std(ret[t-19..t])
        vol20 = np.full(T, np.nan)
        if T >= 21:
            vol20[20:] = np.array([
                float(np.std(ret[t - 19:t + 1])) for t in range(20, T)])
        # turn20[t] = mean(amount[t-19..t])/mean(amount[t-59..t])-1
        turn20 = np.zeros(T)
        for t in range(20, T):
            a_short = float(np.mean(amount[t - 19:t + 1]))
            a_long = float(np.mean(amount[t - 59:t + 1])) if t >= 60 \
                else a_short
            turn20[t] = a_short / (a_long + _eps) - 1.0
        # mcap[t] = log(mean(amount[t-4..t]))
        mcap = np.zeros(T)
        for t in range(4, T):
            mcap[t] = float(np.log(np.mean(amount[t - 4:t + 1]) + _eps))
        styles_daily[s] = {
            "ts": t_arr, "ret20": ret20, "vol20": vol20,
            "turn20": turn20, "mcap": mcap,
            "industry": industry_map.get(s, "Unknown"),
        }
    usable = [s for s in symbols if s in styles_daily]
    n_stocks = len(usable)
    if n_stocks < 10:
        return panel, {"n_days": 0, "n_stocks": n_stocks, "industries": 0,
                       "degraded": True, "r2_mean": 0.0}
    industries = sorted({styles_daily[s]["industry"] for s in usable})

    def _style_row(s: str, j: int) -> list[float] | None:
        """该股票截至索引 j 的风格行（j 为 t 前最后一根 ≤ t 的 bar）。"""
        sd = styles_daily[s]
        if j < 20 or not np.isfinite(sd["vol20"][j]):
            return None
        row = [1.0]
        if has_industry and len(industries) > 1:
            base = industries[0]
            ind = sd["industry"]
            row += [1.0 if ind == i and ind != base else 0.0
                    for i in industries[1:]]
        row += [sd["mcap"][j], sd["ret20"][j], sd["vol20"][j],
                sd["turn20"][j]]
        return row

    out = panel.copy()
    r2s: list[float] = []
    n_days = 0
    for ts, row in panel.iterrows():
        # 逐标的取截至 ts 的因果风格（searchsorted：≤ ts 的最后一根）。
        # 单位对齐：K 线 ts 是 epoch 秒；pd.Timestamp.value 是纳秒，
        # 必须 //1e9 转秒，否则所有日期都会被 searchsorted 推到最后一根。
        ts_i = int(ts.value // 10**9) if hasattr(ts, "value") else int(ts)
        rows_X: list[list[float]] = []
        rows_sym: list[str] = []
        for s in usable:
            sd = styles_daily[s]
            j = int(np.searchsorted(sd["ts"], ts_i, side="right")) - 1
            if j < 20:
                continue
            feat = _style_row(s, j)
            if feat is None:
                continue
            rows_X.append(feat)
            rows_sym.append(s)
        if len(rows_X) < 10:
            continue
        X = np.array(rows_X, dtype=np.float64)  # [N, F]
        y = row.reindex(rows_sym).values.astype(np.float64)
        valid = np.isfinite(y) & (np.abs(y) < 1e8)
        if valid.sum() < 10:
            continue
        Xv, yv = X[valid], y[valid]
        # 最小二乘残差（带截距的线性回归）
        try:
            beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
            resid = yv - Xv @ beta
        except np.linalg.LinAlgError:
            resid = yv - yv.mean()
        # 拟合优度：必须用未标准化残差（标准化后量纲被改写，R² 会失真为负）
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        if ss_tot > 1e-12:
            r2s.append(1.0 - float(np.sum(resid ** 2)) / ss_tot)
        # 残差标准化（截面 zscore，防量纲漂移）——仅用于输出面板
        sd = float(np.std(resid))
        if sd > 1e-9:
            resid = (resid - resid.mean()) / sd
        syms_use = [s for s, v in zip(rows_sym, valid) if v]
        syms_nan = [s for s, v in zip(rows_sym, valid) if not v]
        out.loc[ts, syms_use] = resid
        out.loc[ts, syms_nan] = np.nan
        n_days += 1
    return out, {
        "n_days": n_days, "n_stocks": n_stocks,
        "industries": len(industries) if has_industry else 0,
        "degraded": not has_industry or n_days == 0,
        "r2_mean": float(np.mean(r2s)) if r2s else 0.0,
    }
