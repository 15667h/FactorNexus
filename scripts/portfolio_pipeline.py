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
    combine_icir, combine_equal,
)
from model_core.portfolio.portfolio import (  # noqa: E402
    backtest_portfolio, build_portfolio, performance, risk_model,
)
from model_core.portfolio.attribution import (  # noqa: E402
    style_attribution,
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
        # 数据清洗（2026-08-27 修复）：K 线库可能有未来时间戳/重复/0 价格
        # （历史混库遗留），直接用会污染面板（曾出现 14208 交易日 ≈ 58 年，
        # 远超 A 股历史——脏 ts 使并集轴膨胀）
        from data_pipeline.quality import clean_series
        kdf, _ = clean_series(kdf)
        if kdf.empty or len(kdf) < 60:
            continue
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

    # 共同日期轴 = **因子实际覆盖的交易日**（2026-08-27 修复）：
    # 之前用全部 K 线 ts 并集（90 只老股票 1992 上市 → 14208 天），而因子只
    # 覆盖近期 4400 天 → 组合 9800 天空仓（收益 0）却对比全轴基准（正收益）
    # → 超额恒 -100%、风格R²=-1164 的假象。轴应取因子存在的日期，
    # 收益面板沿轴对齐（无因子日不参与组合与基准对比）。
    factor_ts: set[int] = set()
    for sym, bucket in stock_factors.items():
        factor_ts.update(int(t) for t in bucket.keys())
    if not factor_ts:
        return (pd.DataFrame(), pd.DataFrame(), klines, per_stock)
    axis = sorted(factor_ts)
    axis_dt = pd.to_datetime(axis, unit="s")

    # 收益面板（未来 **1 日** 收益，2026-08-27 修复）：
    # 组合回测 backtest_portfolio 是每日 mark-to-market（ret1d），若传 horizon 日
    # 收益会被当作 1 日收益每日兑现 → 复利爆炸（曾出现 max_dd 95%、
    # 年化+2.79% 与总收益-15% 矛盾的假象）。
    # 用 NaN 初始化而非 0.0 —— 无数据日应保持 NaN（基准 nanmean 会跳过），
    # 若填 0 会把基准均值稀释到接近 0，导致超额/IR/风格归因全部失真。
    ret_panel = pd.DataFrame(np.nan, index=axis_dt, columns=list(klines))
    score_panel = pd.DataFrame(np.nan, index=axis_dt, columns=list(klines))
    for sym, kdf in klines.items():
        close = kdf["close"].values.astype(np.float64)
        t_arr = kdf["ts"].values.astype("int64")
        t_idx = {int(t): i for i, t in enumerate(t_arr)}
        for i, t in enumerate(axis):
            j = t_idx.get(int(t))
            if j is not None and j + 1 < len(close):
                c0 = close[j]
                if np.isfinite(c0) and c0 > 1e-9:
                    ret_panel.loc[axis_dt[i], sym] = \
                        close[j + 1] / c0 - 1.0
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
        # H4 修复：用「全市场并集轴」的 ts→行下标映射写回，而非股票内部
        # K 线位置 t_idx[t]。历史 bug：不同股票因子覆盖长度不同时
        # （新股 1200 根 vs 老股 2000 根），t_idx[t]（小）≠ axis 位置（大），
        # 得分被静默错位到更早日期，与 ret_panel 失配。
        axis_index = {int(t): i for i, t in enumerate(axis)}
        for i, t in enumerate(dates):
            pos = axis_index.get(int(t))
            if pos is not None:
                score_panel.loc[axis_dt[pos], sym] = comp[i]
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
        # 双侧常数防护：fv 或 rv 任一常数 → spearmanr 未定义（返回 nan）
        if ok.sum() >= 10 and np.std(fv[ok]) > 1e-12 \
                and np.std(rv[ok]) > 1e-12:
            ic = spearmanr(fv[ok], rv[ok]).statistic
            if np.isfinite(ic):
                ics.append(ic)
    x_rankic = float(np.mean(ics)) if ics else 0.0
    print(f"[3/8] 合成得分横截面 RankIC: {x_rankic:+.4f} "
          f"（{len(ics)} 个有效截面日）")

    # ── 4. 组合构建（P19：顶层风险预算优化器可选）────────────────────
    composite = neutral if neutral.notna().any(axis=1).sum() > 0 \
        else score_panel
    opt = getattr(args, "optimizer", "equal")
    if opt in ("markowitz", "risk_parity", "black_litterman"):
        from model_core.portfolio.optimizer import optimize_portfolio_panel
        weights = optimize_portfolio_panel(
            composite, ret_panel, method=opt, n_top=args.n_top,
            window=args.opt_window, rebalance=args.rebalance,
            risk_aversion=args.risk_aversion,
            long_short=not args.long_only)
        print(f"[4/8] 组合构建: {'纯多头' if args.long_only else '多空'} "
              f"Top{args.n_top} 优化器={opt} "
              f"(窗口{args.opt_window}日/持有{args.rebalance}日/"
              f"风险厌恶{args.risk_aversion})", flush=True)
    else:
        weights = build_portfolio(composite, n_top=args.n_top,
                                  weights=args.weight,
                                  long_short=not args.long_only)
        print(f"[4/8] 组合构建: {'纯多头' if args.long_only else '多空'} "
              f"Top{args.n_top} 权重={args.weight}", flush=True)
    active = int((weights.abs().sum(axis=1) > 0).sum())
    print(f"      有持仓天数={active}", flush=True)

    # ── 5. 组合回测 ──────────────────────────────────────────────────
    # M2 修复：涨跌停阈值分板块——主板 10%，科创板(688)/创业板(30) 20%。
    bt = backtest_portfolio(
        weights, ret_panel, cost=args.cost,
        limit_filter=not args.no_limit_filter,
        limit_pct={"sh688": 0.199, "sz30": 0.199, "__default__": 0.099})
    print(f"[5/8] 组合回测: 总收益={_fmt(bt['total_ret'], True)} "
          f"年化={_fmt(bt['annual_ret'], True)} 波动={_fmt(bt['annual_vol'], True)} "
          f"Sharpe={_fmt(bt['sharpe'])} 回撤={_fmt(bt['max_dd'], True)} "
          f"换手={bt['turnover']:.3f}")

    # ── 6. 绩效与风险 ────────────────────────────────────────────────
    # 基准 = 当日有数据的股票等权均值（nanmean 跳过无数据日；若用 0 填充的
    # mean 会把基准稀释到接近 0，超额/IR 失真——2026-08-27 修复）
    bench = ret_panel.mean(axis=1, skipna=True).values
    bench = np.nan_to_num(bench, nan=0.0)
    perf = performance(bt, bench_ret=bench)
    rm = risk_model(bt["daily_ret"])
    print(f"[6/8] 绩效/风险: 超额={_fmt(perf.get('excess_ret', 0), True)} "
          f"IR={_fmt(perf.get('info_ratio', 0))} "
          f"年化波动={_fmt(rm['vol'], True)}")

    # ── 7. 风格归因（基准 beta + 特质分解）────────────────────────────
    daily = bt["daily_ret"]
    n = min(len(daily), len(bench))
    att_line = "无基准数据"
    if n > 20:
        ex = np.column_stack([bench[:n], np.ones(n)])
        sr = np.column_stack([bench[:n], np.zeros(n)])
        sa = style_attribution(daily[:n], ex, sr)
        # D3 修复：旧实现用硬编码 [0.6,0.4]/[0.5,0.5] 权重与合成行业收益调
        # Brinson——纯伪数字，误导归因结论。无真实行业持仓数据时不做 Brinson，
        # 仅报告风格归因（组合收益 ~ 基准收益 + 截距 的回归分解）。
        _beta = sa.get("style_contrib", {}).get("style_0", 0.0)
        att_line = (f"风格R²={sa['r2']:.3f} 基准beta≈{_beta:+.3f} "
                    f"特质={sa['idiosyncratic']:+.4f} "
                    f"(Brinson 需行业持仓数据，跳过)")
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
    ap.add_argument("--optimizer", default="equal",
                    choices=["equal", "score", "markowitz", "risk_parity",
                             "black_litterman"],
                    help="顶层风险预算优化器（P19）：equal/score=排序选股；"
                         "markowitz/risk_parity/black_litterman=风险模型优化")
    ap.add_argument("--opt-window", type=int, default=60,
                    help="优化器协方差滚动窗口（交易日，默认 60）")
    ap.add_argument("--rebalance", type=int, default=5,
                    help="优化器持有期（交易日，默认 5=匹配 horizon）")
    ap.add_argument("--risk-aversion", type=float, default=2.0,
                    help="风险厌恶系数（markowitz/BL，默认 2.0）")
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
