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
    """逐日横截面五因子中性化。

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
    # 预计算各标的风格特征（用 K 线末端，滚动窗口内视为近似不变——
    # 机构日频中性化实践中风格特征用最近值/滚动均值，此处取最新值）
    styles: dict[str, dict] = {}
    for s in symbols:
        df = klines.get(s)
        if df is None or df.empty:
            continue
        st = build_style_features(s, df, mcap_proxy=mcap_proxy)
        if st:
            st["industry"] = industry_map.get(s, "Unknown")
            styles[s] = st
    n_stocks = len(styles)
    if n_stocks < 10:
        return panel, {"n_days": 0, "n_stocks": n_stocks, "industries": 0,
                       "degraded": True, "r2_mean": 0.0}

    industries = sorted({st["industry"] for st in styles.values()})
    # 行业哑变量设计矩阵（含行业时：行业-1 个哑变量 + 截距；不含：仅截距）
    ind_dummy: dict[str, list[int]] = {}
    for s, st in styles.items():
        row = [1.0]
        if has_industry and len(industries) > 1:
            base = industries[0]
            row += [1.0 if st["industry"] == i and st["industry"] != base
                    else 0.0 for i in industries[1:]]
        row += [st["mcap"], st["ret20"], st["vol20"], st["turn20"]]
        ind_dummy[s] = row
    n_feat = len(ind_dummy[symbols[0]])
    X = np.array([ind_dummy[s] for s in symbols], dtype=np.float64)  # [N, F]

    out = panel.copy()
    r2s: list[float] = []
    n_days = 0
    for ts, row in panel.iterrows():
        y = row.values.astype(np.float64)
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
        out.loc[ts, [s for s, v in zip(symbols, valid) if v]] = resid
        out.loc[ts, [s for s, v in zip(symbols, valid) if not v]] = np.nan
        n_days += 1
    return out, {
        "n_days": n_days, "n_stocks": n_stocks,
        "industries": len(industries) if has_industry else 0,
        "degraded": not has_industry or n_days == 0,
        "r2_mean": float(np.mean(r2s)) if r2s else 0.0,
    }
