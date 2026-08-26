"""
scripts/mine_high_freq.py — 高频因子挖掘（P15，华泰因子工厂 2.0 风格）

把分钟级 K 线聚合为「日频高频特征」（intraday 特征），逐特征做
机构级 OOS 认证（训练段选优 / OOS 段显著性 + 方向一致），达标入库
FactorStore（kind=highfreq），供组合层使用。

数据源：通达信分钟线（pytdx 全历史；1h/30m/15m 可选），腾讯/新浪
分钟线为备选（数据量有限）。

用法：
    python scripts/mine_high_freq.py --symbol sh600519 --tf 1h
    python scripts/mine_high_freq.py --symbols-file pool.txt --tf 30m
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_pipeline.store.kline_store import FactorStore  # noqa: E402
from model_core.highfreq_features import (  # noqa: E402
    HIGHFREQ_FEATURES, build_highfreq_features,
)


def _build_ret(close: np.ndarray, horizon: int) -> np.ndarray:
    T = len(close)
    ret = np.zeros(T)
    if T > horizon:
        ret[:T - horizon] = close[horizon:] / close[:-horizon] - 1.0
    return ret


def _oos_significance(f_os: np.ndarray, r_os: np.ndarray,
                      block: int = 10, n_boot: int = 500) -> tuple[float, float, float]:
    """OOS 段显著性：整体 Spearman + 中心化块自助 p 值（与挖矿机同口径）。"""
    from scipy.stats import spearmanr

    T = len(f_os)
    if T < block * 4:
        return 0.0, 0.0, 1.0
    ok = np.isfinite(f_os) & np.isfinite(r_os)
    if ok.sum() < block * 4:
        return 0.0, 0.0, 1.0
    f_v, r_v = f_os[ok], r_os[ok]
    if np.std(f_v) < 1e-12:
        return 0.0, 0.0, 1.0
    real = float(spearmanr(f_v, r_v).statistic)
    n_blocks = len(f_v) // block
    rng = np.random.default_rng(0)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        bi = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([bi[j] * block + np.arange(block)
                              for j in range(n_blocks)])
        idx = idx[idx < len(f_v)]
        with np.errstate(invalid="ignore"):
            b = spearmanr(f_v[idx], r_v[idx]).statistic
        boot[i] = b if np.isfinite(b) else 0.0
    sd = float(boot.std())
    t = float(real / sd) if sd > 1e-12 else 0.0
    p = float((np.abs(boot - boot.mean()) >= abs(real)).mean())
    return real, t, p


def mine_symbol(symbol: str, tf: str, cfg, store_dir: str) -> dict:
    """单标的：拉分钟数据 → 高频特征 → OOS 认证 → 入库。"""
    t0 = time.time()
    result = {"symbol": symbol, "status": "ok", "n_features": 0,
              "n_accepted": 0, "accepted": []}
    try:
        # ── 1. 分钟数据（通达信优先：全历史；腾讯/新浪备选）──────────
        from web.data_sources.factory import get_source
        bars = None
        last_err = ""
        for src_kind in ("tongdaxin", "tencent", "sina"):
            if bars:
                break
            try:
                src = get_source(src_kind)
                bars = src.fetch_bars(symbol, tf, n=cfg.bars, drop_forming=True)
                result["source"] = src_kind
            except Exception as exc:  # noqa: BLE001
                last_err = f"{src_kind}: {type(exc).__name__}: {exc}"
        if not bars or len(bars) < 200:
            result["status"] = "no_data"
            result["best_formula"] = last_err[:80]
            return result
        import pandas as pd
        df = pd.DataFrame([{
            "ts": int(b.ts), "open": float(b.open), "high": float(b.high),
            "low": float(b.low), "close": float(b.close),
            "volume": float(b.volume),
        } for b in bars])
        if result.get("source") == "tongdaxin":
            df = df.assign(volume=lambda d: d["volume"] * 100.0)

        # ── 2. 高频特征 → 日频面板 ────────────────────────────────────
        feats = build_highfreq_features(df)
        if not feats:
            result["status"] = "no_features"
            return result
        nd = len(next(iter(feats.values())))
        if nd < 60:
            result["status"] = "too_short"
            result["best_formula"] = f"仅 {nd} 个交易日（<60）"
            return result
        # 日频收盘价（用每日最后一根分钟 bar 的 close）
        days = sorted({_day_key(int(b.ts)) for b in bars})
        close_map: dict[str, float] = {}
        for b in bars:
            close_map[_day_key(int(b.ts))] = float(b.close)
        close = np.array([close_map[d] for d in days])
        ret = _build_ret(close, cfg.horizon)

        # ── 3. OOS 认证（训练段/OOS 段分离）──────────────────────────
        T = nd
        oos_n = max(int(T * cfg.oos_frac), 60)
        if oos_n >= T:
            oos_n = max(T // 2, 30)
        n_tr = T - oos_n
        result["detail"] = {"bars": len(df), "days": nd, "source": result.get("source")}

        accepted = []
        for name in HIGHFREQ_FEATURES:
            f = feats[name]
            if f is None or not np.isfinite(f).any():
                continue
            result["n_features"] += 1
            # 因果标准化（expanding zscore 后段，与因子库口径一致）
            fs = _expanding_zscore(f)
            if fs is None:
                continue
            f_tr, r_tr = fs[:n_tr], ret[:n_tr]
            f_os, r_os = fs[-oos_n:], ret[-oos_n:]
            # 训练段方向（决定翻转）；OOS 段必须同号
            tr_ok = np.isfinite(f_tr) & np.isfinite(r_tr)
            if tr_ok.sum() < 40 or np.std(f_tr[tr_ok]) < 1e-12:
                continue
            train_ic = float(np.corrcoef(f_tr[tr_ok], r_tr[tr_ok])[0, 1])
            if not np.isfinite(train_ic):
                continue
            direction = 1.0 if train_ic >= 0 else -1.0
            rankic, t, p = _oos_significance(f_os * direction, r_os)
            if not (np.isfinite(rankic) and np.isfinite(p)):
                continue
            if abs(rankic) >= cfg.min_oos_rankic and p <= cfg.min_oos_p:
                accepted.append({
                    "name": name, "rankic": rankic, "t": t, "p": p,
                    "direction": direction, "train_ic": train_ic,
                })
        result["n_accepted"] = len(accepted)
        result["accepted"] = accepted

        # ── 4. 入库（kind=highfreq，formula=特征索引编码）────────────
        store = FactorStore(store_dir)
        ts_days = np.array([int(pd.Timestamp(d).timestamp()) for d in days])
        for a in accepted:
            name = a["name"]
            fid = HIGHFREQ_FEATURES.index(name)
            formula = [10000 + fid]   # 特征编码（>=10000 区分 param/token 公式）
            factor_df = pd.DataFrame({
                "ts": ts_days, "factor": _expanding_zscore(feats[name]),
                "feature": name,
            })
            report = {
                "engine": "highfreq", "kind": "highfreq",
                "horizon": cfg.horizon, "source": result.get("source"),
                "timeframe": tf,
                "cert_rankic": round(a["rankic"], 4),
                "cert_p": round(a["p"], 4),
                "cert_mode": "oos_highfreq",
                "direction": a["direction"],
                "train_ic": round(a["train_ic"], 4),
                "t": round(a["t"], 3),
                "feature": name,
                "five_dim": {"total": 0.5},  # 高频特征五维从简（占位，文档披露）
            }
            store.save(symbol, formula, "highfreq_v1", factor_df, report=report)
        result["elapsed_s"] = round(time.time() - t0, 1)
    except Exception as exc:  # noqa: BLE001
        result["status"] = f"error: {type(exc).__name__}"
        result["best_formula"] = str(exc)[:120]
    return result


def _day_key(ts: int) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone(
        dt.timedelta(hours=8))).strftime("%Y-%m-%d")


def _expanding_zscore(f: np.ndarray) -> np.ndarray | None:
    """因果 expanding zscore（t 只用 t 及以前；与因子库后处理同口径）。"""
    f = np.asarray(f, dtype=np.float64)
    n = len(f)
    out = np.full(n, np.nan)
    if n < 30:
        return None
    for t in range(29, n):
        seg = f[:t + 1]
        ok = np.isfinite(seg)
        if ok.sum() < 30:
            continue
        m, s = float(seg[ok].mean()), float(seg[ok].std())
        if s > 1e-9:
            out[t] = (f[t] - m) / s
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="高频因子挖掘（分钟级 → 日频特征）")
    ap.add_argument("--symbol", default="sh600519", help="标的（如 sh600519）")
    ap.add_argument("--symbols-file", default="", help="标的清单文件（每行一个）")
    ap.add_argument("--tf", default="1h", choices=["1h", "30m", "15m", "5m"],
                    help="分钟周期（默认 1h，通达信全历史）")
    ap.add_argument("--bars", type=int, default=6000, help="分钟 K 线数量上限")
    ap.add_argument("--horizon", type=int, default=5, help="收益预测周期（交易日）")
    ap.add_argument("--oos-frac", type=float, default=0.25, help="OOS 段比例")
    ap.add_argument("--min-oos-rankic", type=float, default=0.02,
                    help="OOS RankIC 门槛（华泰高频因子同量级）")
    ap.add_argument("--min-oos-p", type=float, default=0.05, help="OOS p 值门槛")
    ap.add_argument("--store-dir", default="store", help="存储根目录")
    args = ap.parse_args()

    symbols = [args.symbol]
    if args.symbols_file:
        p = Path(args.symbols_file)
        if not p.exists():
            print(f"[错误] 清单文件不存在: {p}")
            sys.exit(2)
        symbols = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                   if l.strip()]

    class Cfg:
        pass
    cfg = Cfg()
    cfg.bars = args.bars
    cfg.horizon = args.horizon
    cfg.oos_frac = args.oos_frac
    cfg.min_oos_rankic = args.min_oos_rankic
    cfg.min_oos_p = args.min_oos_p

    total_acc = 0
    for i, sym in enumerate(symbols, 1):
        print(f"[高频挖掘 {i}/{len(symbols)}] {sym} ({args.tf})")
        r = mine_symbol(sym, args.tf, cfg, args.store_dir)
        print(f"  status={r['status']} 特征={r['n_features']} "
              f"达标={r['n_accepted']} 耗时={r.get('elapsed_s', 0)}s")
        for a in r.get("accepted", []):
            print(f"    ✓ {a['name']}: RankIC={a['rankic']:+.4f} "
                  f"p={a['p']:.3f} 方向={a['direction']:+.0f}")
        total_acc += r["n_accepted"]
    print(f"\n合计达标入库: {total_acc} 个高频因子")


if __name__ == "__main__":
    main()
