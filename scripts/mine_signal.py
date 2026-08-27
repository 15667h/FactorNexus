"""
scripts/mine_signal.py — 策略工厂一键流程（M1→M6 全链路）

因子库 → 特征数据集 → walk-forward 多模型（LGBM/MLP/S4）→ 集成 →
信号评估 → 组合层联动（组合构建/回测/绩效/风险/归因）

用法：
    python scripts/mine_signal.py                        # 默认 LGBM + 集成
    python scripts/mine_signal.py --models lgbm,mlp,s4,ensemble   # 模型池对比
    python scripts/mine_signal.py --nn-epochs 15         # 神经网络训练轮数
    python scripts/mine_signal.py --stacking             # 叠加 stacking 两层
    python scripts/mine_signal.py --portfolio --n-top 5  # 接入组合层
    python scripts/mine_signal.py --report out.md        # 输出 Markdown 报告
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

from model_core.strategy_factory import (  # noqa: E402
    build_dataset, walk_forward_fit_predict,
    make_lgbm_regressor, make_mlp_regressor, make_s4_regressor,
    make_bagging_factory, make_ensemble_factory, stacking_fit_predict,
    evaluate_signal, quantile_analysis,
)
from model_core.portfolio.portfolio import (  # noqa: E402
    build_portfolio, backtest_portfolio, performance, risk_model,
)


def _fmt(v, nd=4, pct=False) -> str:
    try:
        if pct:
            return f"{float(v):+.{nd}%}"
        return f"{float(v):+.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _model_factories(models: list[str], trees: int, lr: float,
                     nn_epochs: int, device: str) -> dict[str, callable]:
    """模型名 → 工厂映射（M3/M4 模型池）。"""
    f: dict[str, callable] = {}
    for m in models:
        if m == "lgbm":
            f["lgbm"] = lambda: make_lgbm_regressor(
                n_estimators=trees, learning_rate=lr)
        elif m == "mlp":
            f["mlp"] = lambda: make_mlp_regressor(
                epochs=nn_epochs, device=device)
        elif m == "s4":
            f["s4"] = lambda: make_s4_regressor(
                epochs=nn_epochs, device=device)
        elif m == "gbdt":
            from model_core.strategy_factory import make_gbdt_regressor
            f["gbdt"] = lambda: make_gbdt_regressor(
                n_estimators=min(trees, 200), learning_rate=lr)
        else:
            raise ValueError(f"未知模型 {m}（可选: lgbm,mlp,s4,gbdt,ensemble）")
    return f


def _run_walk_forward(ds, factories: dict, args) -> dict[str, object]:
    """M2 滚动训练：每个模型独立 walk-forward → OOS 预测。"""
    window = None if args.window <= 0 else args.window
    results: dict[str, object] = {}
    for name, factory in factories.items():
        print(f"\n  ── 模型 [{name}] walk-forward 滚动训练 ──")
        res = walk_forward_fit_predict(
            ds, model_factory=factory, step=args.step, window=window,
            gap=args.gap)
        results[name] = res
        print(f"     折数={res.n_folds} OOS 天数={res.oos_days} "
              f"覆盖率={res.coverage:.1%}")
    return results


def _evaluate_model(name: str, res, ret_panel: pd.DataFrame) -> dict:
    """M5 评估：单模型 → RankIC/IC_IR/分层/换手/分段方向。"""
    ev = evaluate_signal(res.pred, ret_panel)
    qa = quantile_analysis(res.pred, ret_panel)
    return {
        "name": name, "rankic": ev["rankic"], "icir": ev["icir"],
        "ic_std": ev["ic_std"], "n_days": ev["n_days"],
        "coverage": ev["coverage"], "turnover": ev["turnover"],
        "half_agree": ev.get("half_agree", False),
        "monotonicity": qa["monotonicity"], "long_short": qa["long_short"],
        "group_returns": qa["group_returns"],
    }


def _print_model_row(m: dict) -> None:
    print(f"    {m['name']:<10} RankIC={_fmt(m['rankic'])}  "
          f"IC_IR={_fmt(m['icir'])} (std={_fmt(m['ic_std'])})  "
          f"日数={m['n_days']} 覆盖={m['coverage']:.0%}  "
          f"换手={m['turnover']:.3f} 分段一致={m['half_agree']}  "
          f"单调性={_fmt(m['monotonicity'])} 多空={_fmt(m['long_short'], 5)}")


def _hold_weights(w: pd.DataFrame, hold: int) -> pd.DataFrame:
    """权重面板持有平滑：每 hold 天按当日信号定仓，区间内持有不变。

    匹配预测周期（信号预测 H 日收益 → H 日调仓），显著降低噪声换手
    （弱信号每日重排的 TopN 几乎随机换血，换手成本会吞噬全部收益）。
    """
    if hold <= 1:
        return w
    idx = list(w.index)
    out = pd.DataFrame(0.0, index=w.index, columns=w.columns)
    for i in range(0, len(idx), hold):
        seg = idx[i:i + hold]
        if not seg:
            break
        out.loc[seg] = w.loc[seg[0]].values
    return out


def _run_portfolio(pred: pd.DataFrame, ret_panel: pd.DataFrame,
                   args) -> dict:
    """M6 组合层接入：信号面板 → 组合构建（排序选股/风险预算优化器）
    → 回测 → 绩效/风险。"""
    print("  ── 组合层联动（M6：信号 → 组合）──")
    opt = getattr(args, "optimizer", "equal")
    if opt in ("markowitz", "risk_parity", "black_litterman"):
        from model_core.portfolio.optimizer import optimize_portfolio_panel
        w = optimize_portfolio_panel(
            pred, ret_panel, method=opt, n_top=args.n_top,
            window=getattr(args, "opt_window", 60),
            rebalance=args.rebalance, risk_aversion=args.risk_aversion,
            long_short=not args.long_only)
        print(f"     组合: {'纯多头' if args.long_only else '多空'} "
              f"Top{args.n_top} 优化器={opt} "
              f"(窗口{getattr(args, 'opt_window', 60)}日/持有{args.rebalance}日)")
    else:
        w_raw = build_portfolio(pred, n_top=args.n_top,
                                long_short=not args.long_only)
        w = _hold_weights(w_raw, args.rebalance)
        print(f"     组合: {'纯多头' if args.long_only else '多空'} "
              f"Top{args.n_top} 权重=等权 持有期={args.rebalance}日")
    active = int((w.abs().sum(axis=1) > 0).sum())
    print(f"     有持仓天数={active}")
    bt = backtest_portfolio(w, ret_panel, cost=args.cost)
    # 基准对齐回测 ts（ret_panel 全轴可能比权重轴长）
    ts = sorted(set(w.index) & set(ret_panel.index))
    bench = ret_panel.reindex(index=ts).mean(axis=1, skipna=True).values
    bench = np.nan_to_num(bench, nan=0.0)
    perf = performance(bt, bench_ret=bench)
    rm = risk_model(bt["daily_ret"])
    cost_loss = float(bt["turnover"] * args.cost * len(ts))
    print(f"     回测: 总收益={_fmt(bt['total_ret'], 2, True)} "
          f"年化={_fmt(bt['annual_ret'], 2, True)} "
          f"Sharpe={_fmt(bt['sharpe'])} 回撤={_fmt(bt['max_dd'], 2, True)} "
          f"换手={bt['turnover']:.3f}")
    print(f"     成本: 单边{args.cost:.4f} → 全期换手成本损耗≈{cost_loss:.1%}")
    print(f"     绩效: 超额={_fmt(perf.get('excess_ret'), 2, True)} "
          f"IR={_fmt(perf.get('info_ratio'))} 年化波动={_fmt(rm['vol'], 2, True)}")
    return {"weights": w, "backtest": bt, "performance": perf, "risk": rm}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="策略工厂：因子库 → 多模型预测 → 集成 → 评估 → 组合层")
    ap.add_argument("--store-dir", default="store", help="存储根目录")
    ap.add_argument("--horizon", type=int, default=5, help="预测周期（交易日）")
    ap.add_argument("--bars", type=int, default=0, help="K线窗口（0=全历史）")
    ap.add_argument("--step", type=int, default=60, help="walk-forward 每折长度")
    ap.add_argument("--window", type=int, default=240, help="训练窗口（0=expanding）")
    ap.add_argument("--gap", type=int, default=5, help="训练/预测间隔（防泄漏）")
    ap.add_argument("--trees", type=int, default=300, help="LightGBM 树数")
    ap.add_argument("--lr", type=float, default=0.05, help="学习率")
    ap.add_argument("--models", default="lgbm,ensemble",
                    help="模型池（逗号分隔: lgbm,mlp,s4,gbdt,ensemble；"
                         "ensemble=选定模型排名平均集成）")
    ap.add_argument("--nn-epochs", type=int, default=12, help="MLP/S4 训练轮数")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="NN 设备")
    ap.add_argument("--ensemble-method", default="rank_avg",
                    choices=["rank_avg", "bagging"],
                    help="集成方式（rank_avg=截面排名平均, bagging=多seed平均）")
    ap.add_argument("--stacking", action="store_true",
                    help="叠加时间分段 stacking（第一层+meta 两层）")
    ap.add_argument("--portfolio", action="store_true", help="接入组合层")
    ap.add_argument("--n-top", type=int, default=5, help="组合规模（多空各 N 只）")
    ap.add_argument("--long-only", action="store_true", help="纯多头（默认多空）")
    ap.add_argument("--rebalance", type=int, default=0,
                    help="调仓持有期（交易日；0=每日调仓，建议=horizon 匹配"
                         "预测周期降换手）")
    ap.add_argument("--optimizer", default="equal",
                    choices=["equal", "markowitz", "risk_parity",
                             "black_litterman"],
                    help="顶层风险预算优化器（P19，--portfolio 时生效）")
    ap.add_argument("--opt-window", type=int, default=60,
                    help="优化器协方差滚动窗口（交易日，默认 60）")
    ap.add_argument("--risk-aversion", type=float, default=2.0,
                    help="风险厌恶系数（markowitz/BL，默认 2.0）")
    ap.add_argument("--cost", type=float, default=0.0003, help="单边换手成本")
    ap.add_argument("--report", default="", help="输出评估报告路径")
    args = ap.parse_args()

    print("═" * 62)
    print("  FactorNexus · 策略工厂（P24 M1→M6 全链路）")
    print("═" * 62)

    # ── M1 数据层 ────────────────────────────────────────────────────
    ds = build_dataset(args.store_dir, horizon=args.horizon, bars=args.bars)
    print(f"[M1 数据] 样本={ds.n_samples} 特征={len(ds.feature_names)} "
          f"股票={ds.meta['n_symbols']} 因子={ds.meta['n_factors']} "
          f"horizon={ds.meta['horizon']}")
    if ds.n_samples < 200:
        print("[错误] 样本过少——先运行 mine_full_market.py 积累因子库")
        sys.exit(2)
    ts_dt = pd.to_datetime(ds.ts, unit="s")
    print(f"          时间范围 [{ts_dt.min():%Y-%m-%d} .. {ts_dt.max():%Y-%m-%d}]")

    # ── 模型池解析（M3/M4）───────────────────────────────────────────
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    base_models = [m for m in models if m != "ensemble"]
    if not base_models:
        base_models = ["lgbm"]
    factories = _model_factories(base_models, args.trees, args.lr,
                                 args.nn_epochs, args.device)

    # ── M2 walk-forward：逐模型 OOS 预测 ─────────────────────────────
    results = _run_walk_forward(ds, factories, args)
    empty = [k for k, v in results.items()
             if v.pred is None or v.pred.empty]
    if len(results) == len(empty):
        print("[错误] 全部模型无 OOS 预测——样本/参数不足以滚动训练")
        sys.exit(2)

    # ── M5 集成 ──────────────────────────────────────────────────────
    print("\n[M5 集成]")
    ens_pred = None
    if "ensemble" in models:
        if args.ensemble_method == "bagging" and len(base_models) == 1:
            name = base_models[0]
            print(f"      bagging：{name} × 3 seed 平均")
            # seed-aware 工厂（bagging 的多样性来自不同 seed）
            if name == "lgbm":
                seed_f = lambda seed=42: make_lgbm_regressor(  # noqa: E731
                    n_estimators=args.trees, learning_rate=args.lr,
                    seed=seed)
            elif name == "gbdt":
                from model_core.strategy_factory import make_gbdt_regressor
                seed_f = lambda seed=42: make_gbdt_regressor(  # noqa: E731
                    n_estimators=min(args.trees, 200), learning_rate=args.lr,
                    seed=seed)
            elif name == "mlp":
                seed_f = lambda seed=42: make_mlp_regressor(  # noqa: E731
                    epochs=args.nn_epochs, device=args.device, seed=seed)
            elif name == "s4":
                seed_f = lambda seed=42: make_s4_regressor(  # noqa: E731
                    epochs=args.nn_epochs, device=args.device, seed=seed)
            else:
                seed_f = factories[name]
            res = walk_forward_fit_predict(
                ds, model_factory=make_bagging_factory(
                    seed_f, n_models=3),
                step=args.step,
                window=None if args.window <= 0 else args.window,
                gap=args.gap)
            results["ensemble"] = res
            ens_pred = res.pred
        else:
            # 排名平均集成：直接对已有模型的 OOS 预测做截面排名平均
            preds = [results[k].pred for k in base_models
                     if k in results and results[k].pred is not None
                     and not results[k].pred.empty]
            if len(preds) <= 1:
                # 单模型 → 自动升级为多 seed bagging（否则集成=原模型无增益）
                print(f"      rank_avg 仅 1 个模型 → 自动升级 "
                      f"{base_models[0]} × 3 seed bagging")
                name0 = base_models[0]
                if name0 == "lgbm":
                    seed_f = lambda seed=42: make_lgbm_regressor(  # noqa: E731
                        n_estimators=args.trees, learning_rate=args.lr,
                        seed=seed)
                elif name0 == "gbdt":
                    from model_core.strategy_factory import make_gbdt_regressor
                    seed_f = lambda seed=42: make_gbdt_regressor(  # noqa: E731
                        n_estimators=min(args.trees, 200),
                        learning_rate=args.lr, seed=seed)
                elif name0 == "mlp":
                    seed_f = lambda seed=42: make_mlp_regressor(  # noqa: E731
                        epochs=args.nn_epochs, device=args.device, seed=seed)
                elif name0 == "s4":
                    seed_f = lambda seed=42: make_s4_regressor(  # noqa: E731
                        epochs=args.nn_epochs, device=args.device, seed=seed)
                else:
                    seed_f = factories[name0]
                res = walk_forward_fit_predict(
                    ds, model_factory=make_bagging_factory(
                        seed_f, n_models=3),
                    step=args.step,
                    window=None if args.window <= 0 else args.window,
                    gap=args.gap)
                results["ensemble"] = res
                ens_pred = res.pred
            else:
                from model_core.strategy_factory import rank_average
                ens_pred = rank_average(preds)
                results["ensemble"] = type("R", (), {"pred": ens_pred})()
                print(f"      rank_avg：{len(preds)} 个模型 OOS 预测截面排名平均"
                      f" → 集成信号（{ens_pred.shape[0]} 日 × "
                      f"{ens_pred.shape[1]} 股）")
    if args.stacking and base_models:
        print("      stacking：时间分段两层（第一层 base → meta 岭回归）")
        from sklearn.linear_model import Ridge
        meta_factory = lambda: Ridge(alpha=1.0)
        st = stacking_fit_predict(ds, [factories[k] for k in base_models],
                                  meta_factory, split_frac=0.7, gap=args.gap)
        results["stacking"] = st
        print(f"      → stacking OOS：{st.pred.shape[0]} 日 × "
              f"{st.pred.shape[1]} 股（{st.n_folds} 折）")

    # ── 评估对比（M5）───────────────────────────────────────────────
    from scripts.portfolio_pipeline import build_panels
    _, ret_panel, _, _ = build_panels(args.store_dir, horizon=args.horizon,
                                      bars=args.bars)
    print("\n[M5 评估对比]（横截面 RankIC / IC_IR / 覆盖 / 换手 / 分段一致）")
    rows: list[dict] = []
    for name, res in results.items():
        if res.pred is None or res.pred.empty:
            continue
        m = _evaluate_model(name, res, ret_panel)
        rows.append(m)
        _print_model_row(m)
    if not rows:
        print("[错误] 无任何可评估预测")
        sys.exit(2)

    # 最佳模型（按 IC_IR）
    best = max(rows, key=lambda r: abs(r["icir"]))
    print(f"\n  最佳: {best['name']}（IC_IR={_fmt(best['icir'])}）")

    # ── M6 组合层联动 ───────────────────────────────────────────────
    combo = {}
    if args.portfolio:
        sel = ens_pred if ens_pred is not None and not ens_pred.empty \
            else results[best["name"]].pred
        combo = _run_portfolio(sel, ret_panel, args)

    # ── 报告 ─────────────────────────────────────────────────────────
    if args.report:
        lines = [
            "# FactorNexus 策略工厂评估报告（M1→M6）", "",
            f"- 样本 {ds.n_samples} / 特征 {len(ds.feature_names)} / "
            f"股票 {ds.meta['n_symbols']} / 因子 {ds.meta['n_factors']}",
            f"- 时间范围 [{ts_dt.min():%Y-%m-%d} .. {ts_dt.max():%Y-%m-%d}]",
            "",
            "## 模型对比", "",
            "| 模型 | RankIC | IC_IR | IC_std | 有效日 | 覆盖 | 换手 | "
            "分段一致 | 单调性 | 多空 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for m in rows:
            lines.append(
                f"| {m['name']} | {_fmt(m['rankic'])} | {_fmt(m['icir'])} | "
                f"{_fmt(m['ic_std'])} | {m['n_days']} | {m['coverage']:.0%} | "
                f"{m['turnover']:.3f} | {m['half_agree']} | "
                f"{_fmt(m['monotonicity'])} | {_fmt(m['long_short'], 5)} |")
        if combo:
            bt, perf, rm = (combo["backtest"], combo["performance"],
                            combo["risk"])
            lines += [
                "", "## 组合层（M6）", "",
                f"- 组合: {'纯多头' if args.long_only else '多空'} "
                f"Top{args.n_top} 成本={args.cost}",
                f"- 总收益 {bt['total_ret']:+.2%} / 年化 {bt['annual_ret']:+.2%} "
                f"/ Sharpe {bt['sharpe']:+.2f} / 回撤 {bt['max_dd']:.1%}",
                f"- 超额 {perf.get('excess_ret', 0):+.2%} / "
                f"IR {perf.get('info_ratio', 0):+.2f} / "
                f"年化波动 {rm['vol']:.2%}",
            ]
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[报告] → {args.report}")


if __name__ == "__main__":
    main()
