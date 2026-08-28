"""
model_core/eval/factor_monitor.py — 因子监控系统（P17）

机构级因子生命周期管理：入库因子持续跟踪 IC 衰减、方向稳定性、
拥挤度、失效预警。当因子实时段表现显著退化（IC 衰减/符号翻转/
与市场相关性漂移）时发出预警——这是"研究框架 → 生产系统"的标志组件。

能力：
  1. compute_decay_track : 计算因子 IC 衰减轨迹（分段 IC 序列）
  2. monitor_factor      : 单因子监控（实时段 vs 认证段对比）
  3. monitor_all         : 全库监控（批量 + 预警清单）
  4. 预警规则：实时段 |RankIC| 跌破认证段一半 / 方向翻转 /
     实时段 p 值不显著（块自助）

用法：
    from model_core.eval.factor_monitor import monitor_all
    report = monitor_all("store")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.store.kline_store import FactorStore, KlineStore


def compute_decay_track(factor: np.ndarray, ret: np.ndarray,
                        seg: int = 60) -> list[float]:
    """IC 衰减轨迹：把序列切成若干段，每段算 RankIC（观察衰减趋势）。

    Returns: [seg0_rankic, seg1_rankic, ...]（按时间升序）
    """
    from scipy.stats import spearmanr

    n = min(len(factor), len(ret))
    if n < seg * 2:
        return []
    f, r = factor[-n:], ret[-n:]
    out = []
    for s in range(0, n - seg + 1, seg):
        a, b = f[s:s + seg], r[s:s + seg]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 30 and np.std(a[ok]) > 1e-12:
            out.append(float(spearmanr(a[ok], b[ok]).statistic))
        else:
            out.append(0.0)
    return out


def _block_bootstrap_p(f: np.ndarray, r: np.ndarray, block: int = 20,
                       n_boot: int = 300) -> float:
    """中心化块自助 p 值（与认证口径一致）。"""
    from scipy.stats import spearmanr

    ok = np.isfinite(f) & np.isfinite(r)
    f_v, r_v = f[ok], r[ok]
    if len(f_v) < block * 4 or np.std(f_v) < 1e-12:
        return 1.0
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
    return float((np.abs(boot - boot.mean()) >= abs(real)).mean())


def monitor_factor(symbol: str, hash_: str, store_dir: str | Path = "store",
                   horizon: int = 5, recent: int = 120,
                   alert_ratio: float = 0.5,
                   alert_p: float = 0.05) -> dict:
    """单因子监控：实时段（最近 recent 根）vs 认证段对比。

    Returns:
        {symbol, hash, engine, kind, bars, cert_rankic, recent_rankic,
         recent_p, decay_track, direction_flip, status, alerts[]}
        status: "healthy" | "warning" | "stale"(数据不足) | "error"
    """
    store = FactorStore(store_dir)
    kstore = KlineStore(store_dir)
    fdf = store.load(symbol, hash_)
    kdf = kstore.load(symbol, "1d")
    if fdf is None or "factor" not in fdf.columns or fdf.empty:
        return {"symbol": symbol, "hash": hash_, "status": "error",
                "alerts": ["因子文件缺失"]}
    if kdf.empty:
        return {"symbol": symbol, "hash": hash_, "status": "error",
                "alerts": ["K线缺失"]}
    factor = fdf["factor"].values.astype(np.float64)
    close = kdf["close"].values.astype(np.float64)
    n = min(len(factor), len(close))
    if n < 60:
        return {"symbol": symbol, "hash": hash_, "status": "stale",
                "alerts": [f"数据不足 {n} 根"]}
    factor, close = factor[-n:], close[-n:]
    # M12 修复：期望收益尾部 horizon 根无未来数据，置 NaN 而非 0——
    # 旧实现置 0 会作为真实样本参与 spearmanr 与块自助，系统性拉低 |RankIC|
    # 并抬高 p 值（约 horizon/recent≈4% 的伪 0 样本）。
    ret = np.full(n, np.nan)
    if n > horizon:
        ret[:n - horizon] = close[horizon:] / close[:-horizon] - 1.0

    meta = store._load_index().get(f"{symbol}_{hash_}", {})
    report = meta.get("report") or {}
    rmeta = report.get("meta") or {}
    # cert 字段在 report.meta 内层（挖掘机入库口径）；兼容顶层/报告层
    cert_rankic = float(rmeta.get("cert_rankic",
                                  report.get("cert_rankic", 0.0)) or 0.0)
    direction = float(rmeta.get("direction",
                                report.get("direction", 1.0)) or 1.0)
    engine = rmeta.get("engine", report.get("engine", "?"))
    kind = rmeta.get("kind", report.get("kind", "?"))

    # 实时段（最近 recent 根）
    recent = min(recent, n)
    f_r, r_r = factor[-recent:], ret[-recent:]
    from scipy.stats import spearmanr
    ok = np.isfinite(f_r) & np.isfinite(r_r)
    if ok.sum() < 40 or np.std(f_r[ok]) < 1e-12:
        return {"symbol": symbol, "hash": hash_, "status": "stale",
                "alerts": [f"实时段有效样本不足 {ok.sum()}"]}
    recent_rankic = float(spearmanr(f_r[ok], r_r[ok]).statistic)
    recent_p = _block_bootstrap_p(f_r, r_r)

    # 方向一致性：实时段 RankIC 与入库 direction 同号才健康
    flip = recent_rankic * direction < 0
    # 衰减判定：实时段 |RankIC| < 认证段 |RankIC| × alert_ratio
    decay = abs(cert_rankic) > 1e-6 and \
        abs(recent_rankic) < abs(cert_rankic) * alert_ratio
    # 显著性判定：实时段不显著
    insignificant = recent_p > alert_p

    alerts = []
    if flip:
        alerts.append(f"方向翻转：入库 direction={direction:+.0f}，"
                      f"实时段 RankIC={recent_rankic:+.4f}")
    if decay:
        alerts.append(f"IC 衰减：认证段 |RankIC|={abs(cert_rankic):.4f}，"
                      f"实时段仅 {abs(recent_rankic):.4f}（<{alert_ratio:.0%}）")
    if insignificant:
        alerts.append(f"实时段不显著：块自助 p={recent_p:.3f}（>{alert_p}）")
    status = "healthy" if not alerts else "warning"

    return {
        "symbol": symbol, "hash": hash_, "engine": engine, "kind": kind,
        "bars": n,
        "cert_rankic": cert_rankic, "recent_rankic": recent_rankic,
        "recent_p": recent_p, "direction": direction,
        "decay_track": compute_decay_track(factor, ret),
        "direction_flip": flip, "status": status, "alerts": alerts,
    }


def monitor_all(store_dir: str | Path = "store", horizon: int = 5,
                recent: int = 120) -> dict:
    """全库监控：批量扫描所有入库因子，输出健康度统计 + 预警清单。

    Returns:
        {"total", "healthy", "warning", "stale", "error",
         "warnings": [单因子监控 dict...], "by_engine": {...}}
    """
    store = FactorStore(store_dir)
    factors = store.list_factors()
    results = []
    for f in factors:
        try:
            results.append(monitor_factor(f["symbol"], f["hash"], store_dir,
                                          horizon=horizon, recent=recent))
        except Exception as exc:  # noqa: BLE001
            results.append({"symbol": f["symbol"], "hash": f["hash"],
                            "status": "error",
                            "alerts": [f"{type(exc).__name__}: {exc}"]})
    warnings = [r for r in results if r["status"] == "warning"]
    by_engine: dict = {}
    for r in results:
        e = r.get("engine", "?")
        by_engine.setdefault(e, {"total": 0, "warning": 0})
        by_engine[e]["total"] += 1
        if r["status"] == "warning":
            by_engine[e]["warning"] += 1
    return {
        "total": len(results),
        "healthy": sum(1 for r in results if r["status"] == "healthy"),
        "warning": len(warnings),
        "stale": sum(1 for r in results if r["status"] == "stale"),
        "error": sum(1 for r in results if r["status"] == "error"),
        "warnings": warnings, "by_engine": by_engine,
    }


def save_monitor_report(report: dict, path: str | Path = "store/meta/"
                        "factor_monitor.json") -> None:
    """持久化监控快照（供趋势对比/预警记录）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    snap = {"ts": int(time.time()), "report": report}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=1,
                              default=str), encoding="utf-8")
    tmp.replace(p)
