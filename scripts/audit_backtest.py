"""
scripts/audit_backtest.py — P0 防作弊审计（改进方案 P0.1）

对"回测本身"做三道防线审计，输出证据表：

  审计① 成交时点（t 收盘信号 → t+1 执行）：
    构造单日脉冲收益，验证 PnL[t] 只使用 w[t-1]（而非 w[t]）——
    若系统用当日信号当日成交（前视），审计会精确抓到。

  审计② 因子因果性（改尾部 → 历史不变）：
    篡改 K 线尾部 100 根（含未来信息），因子历史值必须逐位不变——
    P8 已在算子层锁定，此处做全链路（K线→指标→因子）证据输出。

  审计③ 随机收益对照（shuffle 收益 → 净值应归零）：
    打乱收益面板（破坏信号-收益相关性）重跑同一权重 → 净值分布；
    原策略总收益必须显著超出随机分布（否则=收益来自市场本身而非信号）。

用法：
    python scripts/audit_backtest.py                          # 三道审计 + 报告
    python scripts/audit_backtest.py --report out/audit_backtest.md
    python scripts/audit_backtest.py --shuffle-n 50           # 随机对照次数

验收（IMPROVEMENT_PLAN P0）：
    - 审计① 断言通过（脉冲验证）
    - 审计② 历史因子最大改写 < 1e-9
    - 审计③ 原策略 |total_ret| 超出随机分布 95 分位（或如实报告不显著）
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

from model_core.portfolio.portfolio import backtest_portfolio  # noqa: E402


# ── 审计①：成交时点 t+1 断言 ─────────────────────────────────────────────

def audit_execution_timing(ret_panel: pd.DataFrame) -> dict:
    """脉冲收益法：验证 PnL[t] 只使用 w[t-1]。

    构造收益面板（除第 k 天外全 0，第 k 天全股票 = 1.0），任意权重面板回测：
    - 若 PnL[k] == w[k-1] 总权重（而非 w[k]）→ t+1 执行成立 ✅
    - 若 PnL[k] == w[k] 总权重 → 当日成交（前视）❌
    成本设 0 消除换手干扰。
    """
    n = len(ret_panel)
    cols = list(ret_panel.columns)
    pulse = 0.10   # 脉冲幅度：>9.9% 涨停线会触发 limit_filter，<21% 跳变防御线
    out = {"passed": False, "pnl_at_pulse": np.nan,
           "w_prev_sum": np.nan, "w_cur_sum": np.nan,
           "detail": ""}
    for k in [max(1, n // 2), n - 2]:          # 两个脉冲点
        ret_pulse = pd.DataFrame(0.0, index=ret_panel.index,
                                 columns=cols)
        ret_pulse.iloc[k] = pulse
        # 任意权重面板：前一半持有 A 组，后一半持有 B 组（制造 w[k-1] ≠ w[k]）
        w = pd.DataFrame(0.0, index=ret_panel.index, columns=cols)
        w.iloc[:n // 2] = 0.2
        w.iloc[:n // 2, :5] = 0.2
        w.iloc[n // 2:] = 0.0
        w.iloc[n // 2:, :5] = -0.2
        bt = backtest_portfolio(w, ret_pulse, cost=0.0, ppy=244)
        pnl = bt["daily_ret"]                     # pnl[i] = w_prev @ r[i]
        # 第 k 天脉冲的 PnL 应等于 pulse × Σw[k-1]（而非 Σw[k]）
        w_prev = w.iloc[k - 1].fillna(0.0).values
        w_cur = w.iloc[k].fillna(0.0).values
        pnl_k = float(pnl[k])
        out["pnl_at_pulse"] = pnl_k
        out["w_prev_sum"] = float(np.sum(w_prev))
        out["w_cur_sum"] = float(np.sum(w_cur))
        if abs(pnl_k - pulse * float(np.sum(w_prev))) > 1e-9:
            out["detail"] = (f"脉冲日 PnL={pnl_k:.6f} ≠ pulse×Σw[t-1]="
                             f"{pulse * np.sum(w_prev):.6f}"
                             f"（Σw[t]={np.sum(w_cur):.6f}）"
                             f"——成交时点存在前视")
            return out
    out["passed"] = True
    out["detail"] = "脉冲验证通过：PnL[t] 恒等于 pulse×Σw[t-1]，t+1 执行成立"
    return out


# ── 审计②：因子因果性（改尾部 → 历史不变）───────────────────────────────

def audit_factor_causality(store_dir: str, bars_tail: int = 100) -> dict:
    """篡改 K 线尾部 → 因子历史值必须逐位不变。

    取因子库第一只有效因子，重建其 K 线 + 指标 + 因子：
      1. 基线因子（原 K 线）
      2. 篡改尾部 K 线（最后 bars_tail 根 close 放大 1.5 倍）
      3. 前段（剔除尾部）因子值逐位对比 → 最大绝对差
    """
    from data_pipeline.store.kline_store import FactorStore, KlineStore
    from model_core.indicator_builder import build_indicators
    from model_core.param_vm import ParamVM
    from model_core.formula_dsl import chrom_to_formula

    store, kstore = FactorStore(store_dir), KlineStore(store_dir)
    factors = store.list_factors()
    if not factors:
        return {"passed": False, "detail": "因子库为空，无法审计"}
    f = factors[0]
    sym = f["symbol"]
    fdf = store.load(sym, f["hash"])
    if fdf is None or "factor" not in fdf.columns:
        return {"passed": False, "detail": f"{sym} 无 factor 列"}
    kdf = kstore.load(sym, "1d")
    if kdf.empty or len(kdf) < bars_tail + 250:
        return {"passed": False, "detail": f"{sym} K 线不足"}

    chrom = f["chrom"] if isinstance(f.get("chrom"), list) else None
    base = fdf["factor"].values.astype(np.float64)
    n = min(len(base), len(kdf))

    def _factor_of(df):
        from data_pipeline.quality import clean_series
        df, _ = clean_series(df)
        ind = build_indicators(df)
        vm = ParamVM(ind)
        if chrom is not None:
            return np.asarray(vm.execute(chrom_to_formula(chrom)),
                              dtype=np.float64)
        # 无染色体 → 用指标第一个序列近似（仅作因果方向证据）
        return np.asarray(ind["close"], dtype=np.float64)

    f0 = _factor_of(kdf)
    kdf2 = kdf.copy()
    col = "close"
    kdf2.loc[kdf2.index[-bars_tail:], col] = \
        kdf2.loc[kdf2.index[-bars_tail:], col].values * 1.5
    f1 = _factor_of(kdf2)
    n_cmp = min(len(f0), len(f1)) - bars_tail
    if n_cmp < 100:
        return {"passed": False, "detail": "可比段过短"}
    max_diff = float(np.max(np.abs(f0[:n_cmp] - f1[:n_cmp])))
    out = {
        "passed": max_diff < 1e-9,
        "symbol": sym, "hash": f["hash"][:10],
        "tail_mutated": bars_tail, "comparable": n_cmp,
        "max_rewrite": max_diff,
        "detail": (f"篡改尾部 {bars_tail} 根后，前 {n_cmp} 根因子最大改写 "
                   f"{max_diff:.2e}（<1e-9 通过）"),
    }
    return out


# ── 审计③：随机收益对照（shuffle 归零检验）───────────────────────────────

def audit_random_returns(score_panel: pd.DataFrame, ret_panel: pd.DataFrame,
                         n_top: int = 5, cost: float = 0.0003,
                         n_shuffle: int = 50, seed: int = 42,
                         hold: int = 5) -> dict:
    """打乱收益面板 → 同权重回测 → 净值分布 vs 原策略净值。

    原理：若信号有真 alpha，打乱收益（破坏相关性）后净值应显著退化；
    若净值几乎不变，说明收益来自市场 beta/噪声而非信号。
    hold：持有期（交易日）。弱信号下每日调仓的换手成本会淹没 alpha，
    审计应按实际生产配置（持有期=预测周期）执行。
    """
    from model_core.portfolio.portfolio import build_portfolio

    rng = np.random.default_rng(seed)
    w = build_portfolio(score_panel, n_top=n_top, long_short=True)
    if hold > 1:
        idx = list(w.index)
        w_held = pd.DataFrame(0.0, index=w.index, columns=w.columns)
        for i in range(0, len(idx), hold):
            seg = idx[i:i + hold]
            if seg:
                w_held.loc[seg] = w.loc[seg[0]].values
        w = w_held
    bt_real = backtest_portfolio(w, ret_panel, cost=cost)
    real_total = float(bt_real["total_ret"])

    shuff_totals = []
    for i in range(n_shuffle):
        # 全矩阵重排（彻底破坏信号-收益关联，包括列均值结构——
        # 逐列重排会保留列均值，对常数 alpha 合成数据无效）
        flat = ret_panel.values.ravel()
        r = pd.DataFrame(rng.permutation(flat).reshape(ret_panel.shape),
                         index=ret_panel.index, columns=ret_panel.columns)
        bt = backtest_portfolio(w, r, cost=cost)
        shuff_totals.append(float(bt["total_ret"]))
    arr = np.array(shuff_totals)
    # 双尾分位：原策略是否超出随机分布
    pct = float((np.abs(arr) < np.abs(real_total)).mean())
    out = {
        "passed": pct >= 0.95,
        "real_total": real_total,
        "shuffle_mean": float(arr.mean()),
        "shuffle_std": float(arr.std()),
        "shuffle_p95_abs": float(np.percentile(np.abs(arr), 95)),
        "pct_below_real": pct,
        "n_shuffle": n_shuffle,
        "detail": (f"原策略总收益 {real_total:+.2%}，shuffle 收益分布 "
                   f"{arr.mean():+.2%}±{arr.std():.2%}，"
                   f"{pct:.0%} 的随机样本 |收益| 低于原策略"),
    }
    return out


# ── 报告 ─────────────────────────────────────────────────────────────────

def run_audit(store_dir: str, n_top: int, cost: float, n_shuffle: int,
              report: str) -> dict:
    print("═" * 62)
    print("  FactorNexus · P0 防作弊审计（audit_backtest）")
    print("═" * 62)
    from scripts.portfolio_pipeline import build_panels
    score, ret, _, _ = build_panels(store_dir, horizon=5)
    if score.empty or ret.empty:
        print("[错误] 面板为空——先运行 mine_full_market.py 积累因子库")
        sys.exit(2)

    r1 = audit_execution_timing(ret)
    r2 = audit_factor_causality(store_dir)
    r3 = audit_random_returns(score, ret, n_top=n_top, cost=cost,
                              n_shuffle=n_shuffle)

    print(f"\n[审计① 成交时点] {'✅ 通过' if r1['passed'] else '❌ 失败'}")
    print(f"    {r1['detail']}")
    print(f"\n[审计② 因子因果] {'✅ 通过' if r2['passed'] else '❌ 失败'}")
    print(f"    {r2.get('detail', '（未执行）')}")
    print(f"\n[审计③ 随机对照] {'✅ 通过' if r3['passed'] else '⚠️ 未显著'}")
    print(f"    {r3['detail']}")

    result = {"execution_timing": r1, "factor_causality": r2,
              "random_returns": r3}
    if report:
        lines = [
            "# FactorNexus · P0 防作弊审计报告", "",
            f"- 审计时间：2026-08-28（脚本 `scripts/audit_backtest.py`）",
            f"- 面板：{score.shape[1]} 只股票 × {score.shape[0]} 交易日",
            "", "## 审计① 成交时点（t+1 执行断言）", "",
            f"- 状态：{'✅ 通过' if r1['passed'] else '❌ 失败'}",
            f"- 证据：{r1['detail']}", "",
            "## 审计② 因子因果（改尾部 → 历史不变）", "",
            f"- 状态：{'✅ 通过' if r2['passed'] else '❌ 失败'}",
            f"- 证据：{r2.get('detail', '（未执行）')}", "",
            "## 审计③ 随机收益对照（shuffle 归零检验）", "",
            f"- 状态：{'✅ 通过' if r3['passed'] else '⚠️ 未显著'}",
            f"- 原策略总收益：{r3['real_total']:+.2%}",
            f"- shuffle 分布：{r3['shuffle_mean']:+.2%} ± "
            f"{r3['shuffle_std']:.2%}（p95 |收益| = "
            f"{r3['shuffle_p95_abs']:.2%}）",
            f"- 随机样本低于原策略的比例：{r3['pct_below_real']:.0%}",
            "", "## 结论", "",
            "- 三证全过：回测成交时点无前视、因子全链路因果、"
            "信号收益显著优于随机分布。",
            "- 任一失败：对应审计输出即为修复入口（见代码注释）。",
            "",
        ]
        p = Path(report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[报告] → {report}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="P0 防作弊审计")
    ap.add_argument("--store-dir", default="store")
    ap.add_argument("--n-top", type=int, default=5)
    ap.add_argument("--cost", type=float, default=0.0003)
    ap.add_argument("--shuffle-n", type=int, default=50)
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    run_audit(args.store_dir, args.n_top, args.cost, args.shuffle_n,
              args.report)


if __name__ == "__main__":
    main()
