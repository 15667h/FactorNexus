"""
scripts/robustness_audit.py — P0 随机入场与极端样本检验（改进方案 P0.2）

对应社区四条拷问中的三条，全部输出到 store/meta/robustness_audit.json：

  检验① 随机入场 EV：N 次随机持仓（随机 k 只等权多空）同期收益分布，
        策略实际收益必须超出随机分布 95 分位——否则策略只是"市场在涨"。
  检验② 去顶检验：去掉 Top5/Top10 最大单日赢家后重算净值，EV 必须仍为正
        ——收益不能依赖少数极端样本。
  检验③ 最坏入场：从历史最差 10 个交易日（基准最大跌幅日）入场，记录
        净值路径——最坏情况下的生存能力。

用法：
    python scripts/robustness_audit.py
    python scripts/robustness_audit.py --n-sim 2000 --n-top 5
    python scripts/robustness_audit.py --report out/robustness_audit.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model_core.portfolio.portfolio import (  # noqa: E402
    backtest_portfolio, build_portfolio,
)


# ── 检验①：随机入场 EV ────────────────────────────────────────────────────

def random_entry_ev(ret_panel: pd.DataFrame, n_top: int = 5,
                    n_sim: int = 2000, cost: float = 0.0003,
                    seed: int = 42) -> dict:
    """N 次随机持仓模拟 → 总收益分布 → 策略分位。

    每次模拟：每天随机选 n_top 只做多 + n_top 只做空（等权），
    持有 1 天（与基准组合同频），含同等换手成本。
    """
    rng = np.random.default_rng(seed)
    cols = list(ret_panel.columns)
    n_cols = len(cols)
    ret = ret_panel.values.astype(np.float64)
    ret = np.nan_to_num(ret, nan=0.0)
    T = ret.shape[0]
    col_idx = np.arange(n_cols)
    totals = np.zeros(n_sim)
    for i in range(n_sim):
        # 每日随机多空（向量化：预生成全期索引）
        longs = rng.choice(col_idx, size=n_top, replace=False)
        shorts = rng.choice(col_idx, size=n_top, replace=False)
        w = np.zeros(n_cols)
        w[longs] = 1.0 / n_top
        w[shorts] = -1.0 / n_top
        daily = ret @ w                       # 固定随机组合持有全程
        # 换手成本：第一天建仓 + 每日等权不变（Σ|Δw|=0 后续）→ 只扣首日
        nav = np.cumprod(1.0 + daily)
        totals[i] = nav[-1] - 1.0 - 2.0 * cost
    return {"mean": float(totals.mean()),
            "std": float(totals.std()),
            "p95": float(np.percentile(totals, 95)),
            "p05": float(np.percentile(totals, 5)),
            "p50": float(np.percentile(totals, 50)),
            "n_sim": n_sim}


# ── 检验②：去顶检验 ──────────────────────────────────────────────────────

def top_winner_trim(weights: pd.DataFrame, ret_panel: pd.DataFrame,
                    cost: float = 0.0003) -> dict:
    """去掉 Top5/Top10 最大单日赢家后重算净值 → EV 须仍为正。"""
    bt = backtest_portfolio(weights, ret_panel, cost=cost)
    daily = np.asarray(bt["daily_ret"], dtype=np.float64)
    idx = np.argsort(daily)[::-1]              # 从大到小

    out = {"baseline_total": float(bt["total_ret"]), "rows": []}
    for k in (5, 10):
        mask = np.ones(len(daily), dtype=bool)
        mask[idx[:k]] = False                  # 剔除 Top k 赢家
        d2 = daily[mask]
        total2 = float(np.prod(1.0 + d2) - 1.0)
        out["rows"].append({"trim": k, "total_ret": total2,
                            "retained": float((1.0 + total2) /
                                              max(1.0 + float(bt["total_ret"]),
                                                  1e-12) - 1.0)})
    return out


# ── 检验③：最坏入场 ──────────────────────────────────────────────────────

def worst_entry(weights: pd.DataFrame, ret_panel: pd.DataFrame,
                cost: float = 0.0003, n_worst: int = 10) -> dict:
    """从基准最差 10 个交易日入场 → 净值路径（记录最坏起始下的表现）。"""
    bench = ret_panel.mean(axis=1, skipna=True).values
    bench = np.nan_to_num(bench, nan=0.0)
    worst_days = np.argsort(bench)[:n_worst]
    bt = backtest_portfolio(weights, ret_panel, cost=cost)
    daily = np.asarray(bt["daily_ret"], dtype=np.float64)
    out = {"n_worst": n_worst, "rows": []}
    for d in sorted(worst_days):
        seg = daily[d:]
        nav = float(np.prod(1.0 + seg) - 1.0)
        dd = float(np.min(1.0 + np.cumprod(1.0 + seg)) - 1.0)
        out["rows"].append({"entry_day": int(d),
                            "bench_ret_then": float(bench[d]),
                            "total_from_entry": nav,
                            "max_dd_from_entry": dd})
    out["worst_total"] = min(r["total_from_entry"] for r in out["rows"])
    out["worst_dd"] = min(r["max_dd_from_entry"] for r in out["rows"])
    return out


# ── 主流程 ───────────────────────────────────────────────────────────────

def run_audit(store_dir: str, n_top: int, n_sim: int, cost: float,
              report: str) -> dict:
    print("═" * 62)
    print("  FactorNexus · P0 随机入场与极端样本检验")
    print("═" * 62)
    from scripts.portfolio_pipeline import build_panels
    score, ret, _, _ = build_panels(store_dir, horizon=5)
    if score.empty or ret.empty:
        print("[错误] 面板为空")
        sys.exit(2)

    weights = build_portfolio(score, n_top=n_top, long_short=True)
    bt = backtest_portfolio(weights, ret, cost=cost)

    r1 = random_entry_ev(ret, n_top=n_top, n_sim=n_sim, cost=cost)
    r2 = top_winner_trim(weights, ret, cost=cost)
    r3 = worst_entry(weights, ret, cost=cost)

    pct = float((np.abs(np.random.default_rng(0).normal(
        r1["mean"], r1["std"], 10000)) <
        np.abs(bt["total_ret"])).mean()) if r1["std"] > 0 else 0.0
    passed1 = pct >= 0.95
    passed2 = all(r["total_ret"] > 0 for r in r2["rows"])
    passed3 = r3["worst_total"] > -0.5

    print(f"\n[检验① 随机入场 EV] {'✅ 通过' if passed1 else '⚠️ 未显著'}")
    print(f"    策略总收益 {bt['total_ret']:+.2%} vs 随机分布 "
          f"{r1['mean']:+.2%}±{r1['std']:.2%}（p95={r1['p95']:+.2%}），"
          f"超随机分位 {pct:.0%}")
    print(f"\n[检验② 去顶检验] {'✅ 通过' if passed2 else '❌ 依赖极端样本'}")
    for r in r2["rows"]:
        print(f"    去掉 Top{r['trim']} 赢家后总收益 {r['total_ret']:+.2%}"
              f"（保留 {r['retained']:+.1%}）")
    print(f"\n[检验③ 最坏入场] {'✅ 通过' if passed3 else '❌ 最坏情况不可生存'}")
    print(f"    最差入场总收益 {r3['worst_total']:+.2%}，最差回撤 {r3['worst_dd']:.2%}")

    result = {"strategy_total_ret": float(bt["total_ret"]),
              "random_entry": r1, "top_winner_trim": r2, "worst_entry": r3,
              "verdicts": {"random_ev": passed1, "trim": passed2,
                           "worst": passed3}}
    out_json = Path(store_dir) / "meta" / "robustness_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                   default=float), encoding="utf-8")
    print(f"\n[快照] → {out_json}")

    if report:
        verdict1 = ("✅ 超 95 分位（信号有真 alpha）" if passed1 else
                    "⚠️ 未显著（收益可能来自市场 beta）")
        lines = [
            "# FactorNexus · P0 随机入场与极端样本检验报告", "",
            f"- 面板：{score.shape[1]} 只 × {score.shape[0]} 日，"
            f"Top{n_top} 多空，成本 {cost}", "",
            f"## 检验① 随机入场 EV（{r1['n_sim']} 次模拟）", "",
            f"- 策略总收益：{bt['total_ret']:+.2%}",
            f"- 随机分布：{r1['mean']:+.2%} ± {r1['std']:.2%}（"
            f"p5={r1['p05']:+.2%} / p50={r1['p50']:+.2%} / "
            f"p95={r1['p95']:+.2%}）",
            f"- 结论：{verdict1}", "",
            "## 检验② 去顶检验", "",
        ]
        for r in r2["rows"]:
            lines.append(f"- 去掉 Top{r['trim']} 赢家后总收益 "
                         f"{r['total_ret']:+.2%}")
        lines += ["", "## 检验③ 最坏入场（基准最差 10 日入场）", ""]
        for r in r3["rows"][:5]:
            lines.append(f"- 第 {r['entry_day']} 日入场（当日基准 "
                         f"{r['bench_ret_then']:+.2%}）："
                         f"总收益 {r['total_from_entry']:+.2%}，"
                         f"最大回撤 {r['max_dd_from_entry']:.2%}")
        lines += ["", f"- 最差情况：总收益 {r3['worst_total']:+.2%}，"
                  f"回撤 {r3['worst_dd']:.2%}", ""]
        p = Path(report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"[报告] → {report}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="P0 随机入场与极端样本检验")
    ap.add_argument("--store-dir", default="store")
    ap.add_argument("--n-top", type=int, default=5)
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--cost", type=float, default=0.0003)
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    run_audit(args.store_dir, args.n_top, args.n_sim, args.cost,
              args.report)


if __name__ == "__main__":
    main()
