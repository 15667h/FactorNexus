"""
scripts/migration_audit.py — P0 跨周期/跨品种迁移测试（改进方案 P0.3）

目标：检验信号是否"换池即死"（过拟合单一股票集合/周期）。

  迁移① 随机子池：全池随机抽 3 个互不重叠子池（各约 1/3），重估 RankIC——
        信号结构性有效要求各子池衰减 < 30%。
  迁移② 沪市分段：sh600 前段 vs sh601/603/605 后段（当前数据只有沪市）。
  迁移③ 周期迁移：尝试 1w/60m（当前仅有 1d 时如实报告数据不足）。

衰减判读：
  <30% 结构有效；30-50% 边界；>50% 该池过拟合（信号不可外推）。

用法：
    python scripts/migration_audit.py
    python scripts/migration_audit.py --report out/migration_audit.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model_core.strategy_factory.evaluate import cross_sectional_rankic  # noqa: E402


def _pool_rankic(score: pd.DataFrame, ret: pd.DataFrame,
                 cols: list[str]) -> dict:
    """子池 RankIC（复用策略工厂评估口径）。"""
    if len(cols) < 30:
        return {"n_stocks": len(cols), "rankic": np.nan, "n_days": 0}
    ics, days = cross_sectional_rankic(score[cols], ret[cols])
    arr = np.array(ics, dtype=np.float64)
    return {"n_stocks": len(cols), "n_days": len(arr),
            "rankic": float(arr.mean()) if len(arr) else np.nan}


def run_audit(store_dir: str, report: str, seed: int = 42) -> dict:
    print("═" * 62)
    print("  FactorNexus · P0 跨周期/跨品种迁移测试")
    print("═" * 62)
    from scripts.portfolio_pipeline import build_panels
    score, ret, _, _ = build_panels(store_dir, horizon=5)
    if score.empty or ret.empty:
        print("[错误] 面板为空")
        sys.exit(2)

    cols = list(score.columns)
    base = _pool_rankic(score, ret, cols)
    print(f"\n[全池基线] {base['n_stocks']} 只 → "
          f"RankIC {base['rankic']:+.4f}（{base['n_days']} 日）")

    # ── 迁移① 随机子池 ─────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    perm = rng.permutation(cols)
    n_pool = max(len(perm) // 3, 30)
    pools = [perm[i * n_pool:(i + 1) * n_pool] for i in range(3)]
    rows_a = []
    for i, p in enumerate(pools):
        r = _pool_rankic(score, ret, list(p))
        dec = float("nan") if not np.isfinite(base["rankic"]) or \
            not np.isfinite(r["rankic"]) else \
            (r["rankic"] - base["rankic"]) / abs(base["rankic"])
        rows_a.append({"pool": f"随机子池{i + 1}", **r, "decay": dec})

    # ── 迁移② 沪市分段 ─────────────────────────────────────────────
    sh600 = [c for c in cols if c.startswith("sh600")]
    sh_rest = [c for c in cols if c.startswith(("sh601", "sh603", "sh605"))]
    rows_b = []
    for name, sub in (("sh600 段", sh600), ("sh601/603/605 段", sh_rest)):
        if len(sub) < 30:
            rows_b.append({"pool": name, "n_stocks": len(sub),
                           "rankic": np.nan, "n_days": 0, "decay": np.nan,
                           "note": "样本不足 30"})
            continue
        r = _pool_rankic(score, ret, sub)
        dec = (r["rankic"] - base["rankic"]) / abs(base["rankic"])
        rows_b.append({"pool": name, **r, "decay": dec})

    # ── 迁移③ 周期迁移（数据探测）──────────────────────────────────
    from data_pipeline.store.kline_store import KlineStore
    kstore = KlineStore(store_dir)
    avail = {}
    for tf in ("1w", "60m", "1d"):
        # 抽样探测（全列遍历慢）——D11：移除死变量 n 与冗余 `if True`
        sample = cols[:20]
        n_ok = sum(1 for c in sample if not kstore.load(c, tf).empty)
        avail[tf] = {"sample_hit": n_ok, "sample_size": len(sample),
                     "enough": n_ok >= 15}
    rows_c = [{
        "pool": f"周期 {tf}", "n_stocks": a["sample_hit"],
        "rankic": np.nan, "n_days": 0, "decay": np.nan,
        "note": ("可迁移" if a["enough"] else
                 "数据不足（仅 1d；跨周期需先拉取 1w/60m）"),
    } for tf, a in avail.items()]

    # ── 汇总输出 ───────────────────────────────────────────────────
    print("\n[迁移矩阵]（衰减 = 子池 RankIC 相对全池的变化比例）")
    print(f"  {'池':<20} {'股票':>5} {'RankIC':>9} {'衰减':>9}  判读")
    all_rows = rows_a + rows_b + rows_c
    for r in all_rows:
        if not np.isfinite(r["rankic"]):
            print(f"  {r['pool']:<20} {r['n_stocks']:>5} "
                  f"{'—':>9} {'—':>9}  {r.get('note', '—')}")
            continue
        dec = r["decay"]
        verdict = ("结构有效" if abs(dec) < 0.30 else
                   "边界" if abs(dec) < 0.50 else "⚠️ 该池过拟合")
        print(f"  {r['pool']:<20} {r['n_stocks']:>5} "
              f"{r['rankic']:+.4f} {dec:+.0%}  {verdict}")

    # 判读统计
    valid = [r for r in all_rows if np.isfinite(r["rankic"])]
    n_ok = sum(1 for r in valid if abs(r["decay"]) < 0.30)
    passed = n_ok >= len(valid) * 0.6
    print(f"\n[结论] {'✅ 迁移稳健（' if passed else '⚠️ 迁移脆弱（'}"
          f"{n_ok}/{len(valid)} 个子池衰减 <30%）")

    result = {"baseline": base, "pools": all_rows, "passed": passed}
    if report:
        lines = [
            "# FactorNexus · P0 跨周期/跨品种迁移测试报告", "",
            f"- 全池基线：{base['n_stocks']} 只 → RankIC "
            f"{base['rankic']:+.4f}", "",
            "| 池 | 股票数 | RankIC | 衰减 | 判读 |",
            "|---|---|---|---|---|",
        ]
        for r in all_rows:
            if not np.isfinite(r["rankic"]):
                lines.append(f"| {r['pool']} | {r['n_stocks']} | — | — | "
                             f"{r.get('note', '—')} |")
                continue
            dec = r["decay"]
            v = ("结构有效" if abs(dec) < 0.30 else
                 "边界" if abs(dec) < 0.50 else "该池过拟合")
            lines.append(f"| {r['pool']} | {r['n_stocks']} | "
                         f"{r['rankic']:+.4f} | {dec:+.0%} | {v} |")
        verdict_final = "✅ 迁移稳健" if passed else "⚠️ 迁移脆弱"
        lines += ["", f"**结论**：{verdict_final}"
                  f"（{n_ok}/{len(valid)} 子池衰减 <30%）", ""]
        p = Path(report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"[报告] → {report}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="P0 跨周期/跨品种迁移测试")
    ap.add_argument("--store-dir", default="store")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    run_audit(args.store_dir, args.report)


if __name__ == "__main__":
    main()
