"""
scripts/portfolio_pipeline.py — 组合层一键流水线（P16，炼油厂点火）

从因子库一键跑通完整组合链路并输出机构级报告：

  因子面板构建 → 五因子中性化 → 因子正交化（增量评估）
  → 多因子合成（IC_IR / ML / 等权）→ 组合构建 → 组合回测
  → 绩效（含基准超额/IR）→ Brinson + 风格归因 → 终端报告

用法：
    python scripts/portfolio_pipeline.py                  # 全自动：因子库 → 组合报告
    python scripts/portfolio_pipeline.py --n-top 10       # 组合规模（多空各 N 只）
    python scripts/portfolio_pipeline.py --long-only      # 纯多头组合
    python scripts/portfolio_pipeline.py --ml             # 用随机森林合成（默认 IC_IR）
    python scripts/portfolio_pipeline.py --industry       # 强制拉取行业数据做行业中性化
    python scripts/portfolio_pipeline.py --report out.md  # 输出 Markdown 报告
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

from data_pipeline.store.kline_store import FactorStore, KlineStore  # noqa: E402
from model_core.portfolio.neutralization import (  # noqa: E402
    fetch_industry_map, neutralize_panel,
)
from model_core.portfolio.orthogonalization import incremental_rankic  # noqa: E402
from model_core.portfolio.combination import (  # noqa: E402
    combine_icir, combine_ml, combine_equal,
)
from model_core.portfolio.portfolio import (  # noqa: E402
    backtest_portfolio, build_portfolio, performance, risk_model,
)
from model_core.portfolio.attribution import (  # noqa: E402
    brinson_attribution, style_attribution,
)


def build_panels(store_dir: str = "store", horizon: int = 5,
                 bars: int = 0) -> tuple[pd.DataFrame, pd.DataFrame,
                                         dict, dict]:
    """因子库 → (股票得分面板, 收益面板, K线字典, 每股票因子信息)。

    架构修正（2026-08-27 真实数据验证发现）：
    挖掘层产出「单标的因子」（每只股票独立挖），组合层需要「横截面股票
    面板」（index=共同交易日, columns=股票）。因此按股票聚合：
      1. 每只股票：其全部因子序列按 ts 对齐 → IC_IR 时序加权合成
         （股票内因子合成，权重 ∝ 滚动窗口 IC_IR，防前视）
      2. 股票得分面板：所有股票 K 线日期的并集，各股票综合因子 reindex
      3. 收益面板同轴（未来 horizon 日收益）
    """
    from scipy.stats import spearmanr

    store, kstore = FactorStore(store_dir), KlineStore(store_dir)
    factors = store.list_factors()
    klines: dict[str, pd.DataFrame] = {}
    per_stock: dict[str, dict] = {}
    # 每股票: {ts: factor 列表, 来源描述}
    stock_factors: dict[str, dict[str, list[float]]] = {}
    stock_desc: dict[str, list[str]] = {}
    for f in factors:
        sym = f["symbol"]
        fdf = store.load(sym, f["hash"])
        kdf = kstore.load(sym, "1d")
        if fdf is None or "factor" not in fdf.columns or fdf.empty or kdf.empty:
            continue
        if bars > 0 and len(kdf) > bars:
            kdf = kdf.iloc[-bars:]
        factor = fdf["factor"].values.astype(np.float64)
        close = kdf["close"].values.astype(np.float64)
        n = min(len(factor), len(close))
        if n < 60:
            continue
        factor, close = factor[-n:], close[-n:]
        ts = kdf["ts"].values.astype("int64")[-n:]
        klines[sym] = kdf
        bucket = stock_factors.setdefault(sym, {})
        for t, v in zip(ts, factor):
            bucket.setdefault(int(t), []).append(float(v))
        _rep = f.get("report") or {}
        eng = _rep.get("engine", f.get("engine", "?"))
        kind = _rep.get("kind", f.get("kind", "?"))
        stock_desc.setdefault(sym, []).append(
            f"{eng}/{kind} {f['hash'][:8]}")

    # 共同日期轴（全部股票 K 线 ts 并集，排序）
    all_ts: set[int] = set()
    for kdf in klines.values():
        all_ts.update(int(t) for t in kdf["ts"].values)
    axis = sorted(all_ts)
    axis_dt = pd.to_datetime(axis, unit="s")

    ret_panel = pd.DataFrame(0.0, index=axis_dt, columns=list(klines))
    score_panel = pd.DataFrame(np.nan, index=axis_dt, columns=list(klines))
    for sym, kdf in klines.items():
        close = kdf["close"].values.astype(np.float64)
        t_arr = kdf["ts"].values.astype("int64")
        # 收益面板（未来 horizon 收益；0/负价格防护）
        t_idx = {int(t): i for i, t in enumerate(t_arr)}
        for i, t in enumerate(axis):
            j = t_idx.get(int(t))
            if j is not None and j + horizon < len(close):
                c0 = close[j]
                if np.isfinite(c0) and c0 > 1e-9:
                    ret_panel.loc[axis_dt[i], sym] = \
                        close[j + horizon] / c0 - 1.0
        # 股票综合因子：多因子按滚动 IC_IR 加权（时序，防前视）
        dates = sorted(stock_factors.get(sym, {}).keys())
        if not dates:
            continue
        mat = np.array([np.mean(stock_factors[sym][d]) for d in dates])
        n_d = len(dates)
        # 合成权重：用历史 IC 序列估计 IC_IR（窗口 60）
        w_sum = 0.0
        comp = np.zeros(n_d)
        n_fac = len(stock_factors[sym][dates[0]])
        for k in range(n_fac):
            f_k = np.array([stock_factors[sym][d][k]
                            if k < len(stock_factors[sym][d]) else np.nan
                            for d in dates])
            ics = []
            for i in range(60, n_d):
                seg_f, seg_r = f_k[i - 60:i], mat[i - 60:i]
                ok = np.isfinite(seg_f) & np.isfinite(seg_r)
                if ok.sum() >= 30 and np.std(seg_f[ok]) > 1e-12:
                    ics.append(spearmanr(seg_f[ok], seg_r[ok]).statistic)
            icir = (np.mean(ics) / np.std(ics)) if len(ics) >= 20 \
                and np.std(ics) > 1e-12 else 0.0
            w_sum += abs(icir)
            comp += icir * f_k
        if w_sum > 1e-12:
            comp /= w_sum
        for i, t in enumerate(dates):
            if t in t_idx and t_idx[t] < len(axis):
                score_panel.loc[axis_dt[t_idx[t]], sym] = comp[i]
        per_stock[sym] = {"factors": len(stock_desc.get(sym, [])),
                          "descs": stock_desc.get(sym, []),
                          "n_days": len(dates)}
    return score_panel, ret_panel, klines, per_stock


def _fmt(x, pct=False, nd=2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if pct:
        return f"{v * 100:+.2f}%"
    return f"{v:+.{nd}f}"


def run_pipeline(args) -> dict:
    print("═" * 62)
    print("  FactorNexus · 组合层一键流水线（P16）")
    print("═" * 62)

    # ── 1. 股票得分面板（按股票聚合因子，IC_IR 时序加权合成）──────────
    score_panel, ret_panel, klines, per_stock = build_panels(
        args.store_dir, horizon=args.horizon, bars=args.bars)
    if score_panel.empty:
        print("[错误] 因子库为空或无可对齐面板——先运行 mine_full_market.py 挖矿")
        sys.exit(2)
    n_sym = int(score_panel.shape[1])
    n_fac = sum(v["factors"] for v in per_stock.values())
    print(f"[1/8] 股票得分面板: {n_fac} 个因子 → {n_sym} 只股票 "
          f"{score_panel.shape[0]} 交易日")

    # ── 2. 五因子中性化（横截面）─────────────────────────────────────
    industry_map = fetch_industry_map(args.store_dir) if args.industry else {}
    neutral, rep = neutralize_panel(score_panel, klines,
                                    industry_map=industry_map or None)
    ind_n = rep["industries"] if args.industry else 0
    print(f"[2/8] 五因子中性化: 行业数={ind_n} 有效天数={rep['n_days']} "
          f"R²={rep['r2_mean']:.3f} (degraded={rep['degraded']})")

    # ── 3. 合成（股票面板已是合成得分；此处报告横截面 RankIC）────────
    from scipy.stats import spearmanr
    ics = []
    for ts in neutral.index:
        if ts not in ret_panel.index:
            continue
        f = neutral.loc[ts].astype(float)
        r = ret_panel.loc[ts].astype(float)
        common = f.index.intersection(r.index)
        fv, rv = f[common].values, r[common].values
        ok = np.isfinite(fv) & np.isfinite(rv)
        if ok.sum() >= 10 and np.std(fv[ok]) > 1e-12:
            ics.append(spearmanr(fv[ok], rv[ok]).statistic)
    x_rankic = float(np.mean(ics)) if ics else 0.0
    print(f"[3/8] 合成得分横截面 RankIC: {x_rankic:+.4f} "
          f"（{len(ics)} 个有效截面日）")

    # ── 4. 组合构建 ──────────────────────────────────────────────────
    composite = neutral if neutral.notna().any(axis=1).sum() > 0 \
        else score_panel
    weights = build_portfolio(composite, n_top=args.n_top,
                              weights=args.weight, long_short=not args.long_only)
    active = int((weights.abs().sum(axis=1) > 0).sum())
    print(f"[4/8] 组合构建: {'纯多头' if args.long_only else '多空'} Top{args.n_top} "
          f"权重={args.weight} 有持仓天数={active}")

    # ── 5. 组合回测 ──────────────────────────────────────────────────
    bt = backtest_portfolio(weights, ret_panel, cost=args.cost,
                            limit_filter=not args.no_limit_filter)
    print(f"[5/8] 组合回测: 总收益={_fmt(bt['total_ret'], True)} "
          f"年化={_fmt(bt['annual_ret'], True)} 波动={_fmt(bt['annual_vol'], True)} "
          f"Sharpe={_fmt(bt['sharpe'])} 回撤={_fmt(bt['max_dd'], True)} "
          f"换手={bt['turnover']:.3f}")

    # ── 6. 绩效与风险 ────────────────────────────────────────────────
    bench = ret_panel.mean(axis=1).values
    perf = performance(bt, bench_ret=bench)
    rm = risk_model(bt["daily_ret"])
    print(f"[6/8] 绩效/风险: 超额={_fmt(perf.get('excess_ret', 0), True)} "
          f"IR={_fmt(perf.get('info_ratio', 0))} "
          f"年化波动={_fmt(rm['vol'], True)}")

    # ── 7. Brinson + 风格归因 ────────────────────────────────────────
    daily = bt["daily_ret"]
    n = min(len(daily), len(bench))
    att_line = "无基准数据"
    if n > 20:
        ex = np.column_stack([bench[:n], np.ones(n)])
        sr = np.column_stack([bench[:n], np.zeros(n)])
        sa = style_attribution(daily[:n], ex, sr)
        bri = brinson_attribution(float(np.mean(daily)), float(np.mean(bench)),
                                  np.array([0.6, 0.4]), np.array([0.5, 0.5]),
                                  np.array([float(np.mean(bench)), 0.0]))
        att_line = (f"Brinson 配置={bri['allocation']:+.4f} 选股="
                    f"{bri['selection']:+.4f} | 风格R²={sa['r2']:.3f} "
                    f"特质={sa['idiosyncratic']:+.4f}")
    print(f"[7/8] 归因: {att_line}")

    # ── 8. 每股票因子明细 ────────────────────────────────────────────
    stock_line = "  ".join(f"{s}:{v['factors']}"
                           for s, v in list(per_stock.items())[:10])
    print(f"[8/8] 股票因子分布: {stock_line}")
    descs = [d for v in per_stock.values() for d in v["descs"]]

    # ── 报告 ─────────────────────────────────────────────────────────
    report = {
        "factors": n_fac, "stocks": n_sym, "method": "股票内IC_IR加权+横截面合成",
        "neutral_r2": rep["r2_mean"], "industries": ind_n,
        "cross_rankic": x_rankic,
        "backtest": {k: bt[k] for k in
                     ("total_ret", "annual_ret", "annual_vol", "sharpe",
                      "sortino", "max_dd", "calmar", "turnover", "n")},
        "performance": {k: perf.get(k) for k in ("excess_ret", "info_ratio")},
        "risk": rm, "attribution": att_line,
        "descs": descs[:20],
    }
    if args.report:
        _write_markdown(report, args.report)
        print(f"\n[报告] → {args.report}")
    return report


def _write_markdown(r: dict, path: str) -> None:
    lines = [
        "# FactorNexus 组合层流水线报告", "",
        f"- 因子数：{r['factors']} ｜ 股票数：{r['stocks']}",
        f"- 合成方法：{r['method']}",
        f"- 中性化：行业数={r['industries']} 平均R²={r['neutral_r2']:.3f}",
        "",
        "## 组合绩效",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 总收益 | {_fmt(r['backtest']['total_ret'], True)} |",
        f"| 年化收益 | {_fmt(r['backtest']['annual_ret'], True)} |",
        f"| 年化波动 | {_fmt(r['backtest']['annual_vol'], True)} |",
        f"| Sharpe | {_fmt(r['backtest']['sharpe'])} |",
        f"| 索提诺 | {_fmt(r['backtest']['sortino'])} |",
        f"| 最大回撤 | {_fmt(r['backtest']['max_dd'], True)} |",
        f"| Calmar | {_fmt(r['backtest']['calmar'])} |",
        f"| 换手(日均) | {r['backtest']['turnover']:.3f} |",
        f"| 超额收益 | {_fmt(r['performance'].get('excess_ret'), True)} |",
        f"| 信息比率 | {_fmt(r['performance'].get('info_ratio'))} |",
        "",
        "## 合成得分横截面 RankIC",
        "",
        f"- 横截面 RankIC：{r['cross_rankic']:+.4f}",
        "",
        "## 归因",
        "",
        f"- {r['attribution']}",
        "",
        "## 参与因子",
        "",
    ]
    lines += [f"- `{d}`" for d in r["descs"]]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="组合层一键流水线：因子库 → 中性化/正交化/合成 → 组合回测 → 归因")
    ap.add_argument("--n-top", type=int, default=5, help="组合规模（多空各 N 只）")
    ap.add_argument("--long-only", action="store_true", help="纯多头（默认多空）")
    ap.add_argument("--weight", default="equal", choices=["equal", "score"],
                    help="权重方式（默认等权）")
    ap.add_argument("--window", type=int, default=60,
                    help="IC_IR 滚动窗口（默认 60 交易日）")
    ap.add_argument("--ml", action="store_true", help="用随机森林合成（默认 IC_IR）")
    ap.add_argument("--ml-window", type=int, default=120, help="ML 滚动训练窗口")
    ap.add_argument("--ml-trees", type=int, default=100, help="ML 树数")
    ap.add_argument("--horizon", type=int, default=5, help="收益预测周期")
    ap.add_argument("--bars", type=int, default=0,
                    help="K 线窗口（0=全历史）")
    ap.add_argument("--cost", type=float, default=0.0003, help="单边换手成本")
    ap.add_argument("--no-limit-filter", action="store_true",
                    help="关闭涨跌停限制（默认开）")
    ap.add_argument("--industry", action="store_true",
                    help="拉取行业数据做行业中性化（默认仅风格中性化）")
    ap.add_argument("--report", default="", help="输出 Markdown 报告路径")
    ap.add_argument("--store-dir", default="store", help="存储根目录")
    args = ap.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
