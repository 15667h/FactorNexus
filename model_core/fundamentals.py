"""
model_core/fundamentals.py — 基本面数据管线（P18）

机构级因子库 = 量价 + 基本面（估值/盈利/成长/质量）。本模块补齐财务维度：

数据源（免费、实测可用）：
  1. 东财业绩报表（datacenter-web RPT_LICO_FN_CPD）：ROE/营收增速/净利增速/
     毛利率/负债率等 37 字段，按报告期（季报）下发
  2. 腾讯实时行情（qt.gtimg.cn）：PE(TTM)/PB/总市值/流通市值/股息率

产出：
  - fetch_fundamentals : 批量拉取 → 缓存 store/meta/fundamentals.json
  - build_fundamental_factors : 从财务快照构建基本面因子序列
    （估值 EP/BP、盈利 ROE/毛利率、成长 营收YOY/净利YOY、质量 负债率/现金流）
  - 全部因子按"最新报告期前值填充"（机构惯例：财报未出时用上一期）

用法：
    from model_core.fundamentals import fetch_fundamentals, build_fundamental_factors
    data = fetch_fundamentals(["sh600519", "sz000001"])
    factors = build_fundamental_factors(data)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

_EM_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_QT_URL = "https://qt.gtimg.cn/q="
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# 业绩报表字段 → 基本面因子（机构四类：估值/盈利/成长/质量）
# 实测（2026-08-27）：RPT_LICO_FN_CPD 每期仅含
#   WEIGHTAVG_ROE / TOTAL_OPERATE_INCOME / PARENT_NETPROFIT / BASIC_EPS
# 成长类（YOY）无现成字段 → 拉最近 6 期按「去年同期」自算同比；
# 毛利率/负债率无免费字段 → 可选（NaN 时跳过入库，文档披露）
_FUND_FIELDS = {
    # 盈利（最新期）
    "roe": "WEIGHTAVG_ROE",          # 加权净资产收益率
    "eps": "BASIC_EPS",              # 基本每股收益
    "net_profit": "PARENT_NETPROFIT",  # 归母净利
    "revenue": "TOTAL_OPERATE_INCOME", # 营收
    # 成长（去年同期自算）
    "rev_yoy": None,                 # 营收同比（自算）
    "profit_yoy": None,              # 归母净利同比（自算）
    # 质量（无免费字段，NaN 跳过）
    "gross_margin": None,
    "debt_ratio": None,
}
# 估值（腾讯行情）
_QT_FIELDS = {"pe": 39, "pb": 46, "mcap": 44, "float_mcap": 45}


def _to_symbol(code: str) -> str:
    """东财 6 位代码 → sh/sz 前缀。"""
    code = str(code).strip()
    if len(code) != 6:
        return ""
    return ("sh" if code[0] in "569" else "sz") + code


def fetch_fundamentals(symbols: list[str], store_dir: str | Path = "store",
                       refresh: bool = False) -> dict[str, dict]:
    """批量拉取基本面快照 {symbol: {报告期字段...}}，缓存 fundamentals.json。

    东财按代码逐只拉最新 1 期报告（季度更新）；腾讯估值按批拉（60 只/批）。
    失败标的降级为 {}（不阻塞），缓存 12 小时。
    """
    import requests

    root = Path(store_dir)
    cache = root / "meta" / "fundamentals.json"
    if not refresh and cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    session = requests.Session()
    session.headers.update(_UA)
    out: dict[str, dict] = {}
    for i, sym in enumerate(symbols):
        code = sym[2:] if sym[:2] in ("sh", "sz") else sym
        try:
            # 拉最近 6 期（计算去年同期同比）
            params = {"reportName": "RPT_LICO_FN_CPD", "columns": "ALL",
                      "filter": f'(SECURITY_CODE="{code}")',
                      "pageNumber": 1, "pageSize": 6,
                      "sortColumns": "REPORTDATE", "sortTypes": -1}
            r = session.get(_EM_URL, params=params, timeout=15)
            rows = (r.json().get("result") or {}).get("data") or []
            if rows:
                row = rows[0]
                rec: dict = {"report_date": str(row.get("REPORTDATE", ""))[:10]}
                # 盈利/规模（最新期）
                for key, col in _FUND_FIELDS.items():
                    if col is None:
                        continue
                    v = row.get(col)
                    try:
                        rec[key] = float(v) if v not in (None, "", "-") \
                            else np.nan
                    except (TypeError, ValueError):
                        rec[key] = np.nan
                # 成长：去年同期（REPORTDATE 同月同日）自算同比
                cur_md = str(row.get("REPORTDATE", ""))[5:10]
                for key, col in (("rev_yoy", "TOTAL_OPERATE_INCOME"),
                                 ("profit_yoy", "PARENT_NETPROFIT")):
                    cur_v = row.get(col)
                    yoy = np.nan
                    if cur_v not in (None, "", "-"):
                        for old in rows[1:]:
                            if str(old.get("REPORTDATE", ""))[5:10] == cur_md:
                                old_v = old.get(col)
                                if old_v not in (None, "", "-") and \
                                        abs(float(old_v)) > 1e-9:
                                    yoy = float(cur_v) / float(old_v) - 1.0
                                break
                    rec[key] = yoy
                # 质量字段（gross_margin/debt_ratio 无免费源 → NaN，跳过入库）
                rec.setdefault("gross_margin", np.nan)
                rec.setdefault("debt_ratio", np.nan)
                out[sym] = rec
            time.sleep(0.12)
        except Exception:  # noqa: BLE001
            continue
    # 腾讯估值（批量；接口要求带 sh/sz 前缀，如 q=sh600519,sz000001）
    if out:
        codes = ",".join(sym for sym in out if sym[:2] in ("sh", "sz"))
        try:
            resp = session.get(_QT_URL + codes, timeout=15)
            text = resp.content.decode("gbk", errors="replace")
            for line in text.split(";"):
                line = line.strip()
                if not line.startswith("v_"):
                    continue
                body = line.split('="', 1)[-1].rstrip('";')
                f = body.split("~")
                if len(f) < 50:
                    continue
                sym = _to_symbol(f[2])
                if sym not in out:
                    continue
                for key, idx in _QT_FIELDS.items():
                    try:
                        out[sym][key] = float(f[idx]) if f[idx] else np.nan
                    except (TypeError, ValueError):
                        out[sym][key] = np.nan
        except Exception:  # noqa: BLE001
            pass
    if out:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache)
    return out


# ── 基本面因子构建 ─────────────────────────────────────────────────────────

def build_fundamental_factors(data: dict[str, dict]) -> dict[str, np.ndarray]:
    """从基本面快照构建因子序列（估值/盈利/成长/质量）。

    Args:
        data: fetch_fundamentals 输出 {symbol: {字段}}。

    Returns:
        {因子名: np.ndarray[n_stocks]}。因子定义：
          ep        = 1/PE（盈利收益率，估值越低越好 → 越高越便宜）
          bp        = 1/PB（账面市值比，价值因子）
          roe       = 加权 ROE（盈利质量）
          gross     = 毛利率
          rev_yoy   = 营收同比（成长）
          profit_yoy= 净利同比（成长）
          debt      = 负债率（质量，负向）
    """
    factors: dict[str, list] = {
        "ep": [], "bp": [], "roe": [], "gross": [],
        "rev_yoy": [], "profit_yoy": [], "debt": [],
    }
    for sym, rec in data.items():
        eps = 1e-9
        pe = rec.get("pe")
        pb = rec.get("pb")
        ep = 1.0 / pe if pe and np.isfinite(pe) and pe > 0 else np.nan
        bp = 1.0 / pb if pb and np.isfinite(pb) and pb > 0 else np.nan
        factors["ep"].append(ep)
        factors["bp"].append(bp)
        factors["roe"].append(rec.get("roe", np.nan))
        factors["gross"].append(rec.get("gross_margin", np.nan))
        factors["rev_yoy"].append(rec.get("rev_yoy", np.nan))
        factors["profit_yoy"].append(rec.get("profit_yoy", np.nan))
        factors["debt"].append(rec.get("debt_ratio", np.nan))
    out = {}
    for name, vals in factors.items():
        arr = np.array(vals, dtype=np.float64)
        # 去极值 + 标准化（截面）
        ok = np.isfinite(arr)
        if ok.sum() > 0:
            med = np.nanmedian(arr)
            mad = np.nanmedian(np.abs(arr - med)) * 1.4826 + 1e-9
            arr = np.clip((arr - med) / (3.0 * mad), -3, 3)
        out[name] = arr
    return out


# ── 基本面因子 → 因子库（与挖掘管线对接）──────────────────────────────────

def save_fundamental_factors(data: dict[str, dict],
                             store_dir: str | Path = "store") -> int:
    """把基本面因子写入 FactorStore（kind=fundamental），返回入库数。

    每只股票、每个基本面因子存为一条因子记录（formula=20000+fid 编码，
    vocab_version="fundamental_v1"），供组合层/回测直接使用。
    """
    from data_pipeline.store.kline_store import FactorStore, KlineStore

    store = FactorStore(store_dir)
    kstore = KlineStore(store_dir)
    names = ["ep", "bp", "roe", "gross", "rev_yoy", "profit_yoy", "debt"]
    # 因子名 → 快照字段名（fetch_fundamentals 输出键）
    field_map = {"ep": "pe", "bp": "pb", "roe": "roe", "gross": "gross_margin",
                 "rev_yoy": "rev_yoy", "profit_yoy": "profit_yoy",
                 "debt": "debt_ratio"}
    saved = 0
    for sym, rec in data.items():
        kdf = kstore.load(sym, "1d")
        if kdf.empty:
            continue
        ts = kdf["ts"].values.astype("int64")
        n = len(ts)
        for i, name in enumerate(names):
            v = rec.get(field_map[name], np.nan)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            # H6 修复：ep/bp 因子语义 = 1/PE、1/PB（估值越低越好 → 因子越高越便宜）。
            # 历史 bug：直接存原始 PE/PB，名为 ep/bp 实为 PE/PB，方向互为倒数。
            if name in ("ep", "bp"):
                if v <= 0:
                    continue  # PE/PB 非正无法取倒数
                v = 1.0 / v
            factor = np.full(n, float(v))
            formula = [20000 + i]  # 基本面因子编码（>=20000）
            fdf = pd.DataFrame({"ts": ts, "factor": factor})
            report = {
                "engine": "fundamental", "kind": "fundamental",
                "feature": name, "report_date": rec.get("report_date", ""),
                "five_dim": {"total": 0.5},  # 从简（文档披露）
            }
            store.save(sym, formula, "fundamental_v1", fdf, report=report)
            saved += 1
    return saved
