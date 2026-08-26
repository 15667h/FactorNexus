"""
model_core/portfolio/portfolio.py — 组合构建 + 组合回测 + 简化风险模型（P14）

对齐 Qlib Portfolio Generator / Order Executor / Analyser 与 Barra 风格：
  1. build_portfolio   : 合成因子横截面排序 → 权重（等权/因子加权，多空可选）
  2. backtest_portfolio: 组合回测（t+1 执行、换手成本、跳变防御、涨跌停限制）
  3. risk_model        : 简化风险模型（组合波动/协方差/风格暴露/Beta）
  4. performance       : 组合绩效指标（收益/波动/Sharpe/IR/最大回撤/超额）

用法：
    from model_core.portfolio.portfolio import build_portfolio, backtest_portfolio
    weights = build_portfolio(score_panel, n_top=30)
    perf = backtest_portfolio(weights, ret1d_panel, cost=0.0003)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── 组合构建 ───────────────────────────────────────────────────────────────

def build_portfolio(score_panel: pd.DataFrame, n_top: int = 30,
                    weights: str = "equal", long_short: bool = True,
                    threshold: float | None = None) -> pd.DataFrame:
    """横截面排序选股 → 权重矩阵（index=ts, columns=symbol）。

    Args:
        score_panel: 合成因子得分面板（越大越好）。
        n_top: 多头（与空头）数量。
        weights: "equal"（等权）| "score"（按得分线性加权）。
        long_short: True=多空（Top做多 + Bottom做空）；False=纯多头。
        threshold: 可选，|score| 低于阈值的股票不入组合。

    Returns:
        权重矩阵（每日 Σ|w| = 2 for 多空 / =1 for 纯多；NaN=空仓）。
    """
    out = pd.DataFrame(0.0, index=score_panel.index, columns=score_panel.columns)
    for ts, row in score_panel.iterrows():
        vals = row.dropna()
        if vals.empty or len(vals) < n_top * 2:
            continue
        if threshold is not None:
            vals = vals[vals.abs() >= threshold]
        ranked = vals.sort_values(ascending=False)
        longs = ranked.head(n_top)
        shorts = ranked.tail(n_top) if long_short else []
        w = pd.Series(0.0, index=vals.index)
        if weights == "score":
            wl = longs - longs.mean() if len(longs) > 1 else \
                pd.Series(1.0 / len(longs), index=longs.index)
            wl = wl / (wl.abs().sum() + 1e-12)
            w[longs.index] = wl
            if len(shorts):
                ws = shorts - shorts.mean() if len(shorts) > 1 else \
                    pd.Series(1.0 / len(shorts), index=shorts.index)
                ws = ws / (ws.abs().sum() + 1e-12)
                w[shorts.index] = -ws
        else:
            w[longs.index] = 1.0 / len(longs)
            if len(shorts):
                w[shorts.index] = -1.0 / len(shorts)
        out.loc[ts] = w
    return out


# ── 组合回测 ───────────────────────────────────────────────────────────────

def backtest_portfolio(weights: pd.DataFrame, ret1d_panel: pd.DataFrame,
                       cost: float = 0.0003, limit_filter: bool = True,
                       ppy: int = 244) -> dict:
    """组合回测（Qlib Order Executor 简化版）。

    规则：
      - t 收盘信号 → t+1 执行（避免前视）
      - 每日 mark-to-market，调仓日扣换手成本（双边 |Δw|·cost）
      - 涨跌停不可成交（±9.9% 近似；跳变日收益置 0 防混库）

    Returns: {nav, daily_ret, total_ret, annual_ret, annual_vol, sharpe,
              sortino, max_dd, calmar, turnover, n}
    """
    symbols = list(weights.columns)
    ts = sorted(set(weights.index) & set(ret1d_panel.index))
    if len(ts) < 20:
        return {"nav": np.array([1.0]), "daily_ret": np.array([0.0]),
                "total_ret": 0.0, "annual_ret": 0.0, "annual_vol": 0.0,
                "sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0,
                "calmar": 0.0, "turnover": 0.0, "n": 0}
    w_prev = np.zeros(len(symbols))
    pnl = np.zeros(len(ts))
    turnover_sum = 0.0
    nav = np.ones(len(ts) + 1)
    ret1d = ret1d_panel.reindex(index=ts, columns=symbols).fillna(0.0).values
    for i, t in enumerate(ts):
        w_cur = weights.loc[t].reindex(symbols).fillna(0.0).values
        if limit_filter:
            # 涨停不可买入（正权重日收益>=9.9% 视为涨停，买入受限→保持原仓位）
            r = ret1d[i]
            w_cur = w_cur.copy()
            up = r >= 0.099
            down = r <= -0.099
            w_cur[up & (w_cur > w_prev)] = w_prev[up & (w_cur > w_prev)]
            w_cur[down & (w_cur < w_prev)] = w_prev[down & (w_cur < w_prev)]
        # 跳变防御：|1日收益|>21% 置 0（混库/复权瑕疵，收益不可信）
        r_safe = np.where(np.abs(ret1d[i]) > 0.21, 0.0, ret1d[i])
        pnl[i] = float(w_prev @ r_safe)
        turnover = float(np.abs(w_cur - w_prev).sum())
        pnl[i] -= turnover * cost
        turnover_sum += turnover
        nav[i + 1] = nav[i] * (1.0 + pnl[i])
        w_prev = w_cur
    daily = pnl[1:]
    mean_d, std_d = float(daily.mean()), float(daily.std())
    total_ret = float(nav[-1] - 1.0)
    annual_ret = float((1.0 + mean_d) ** ppy - 1.0) if mean_d > -1 else -1.0
    annual_vol = float(std_d * np.sqrt(ppy))
    sharpe = float(mean_d / std_d * np.sqrt(ppy)) if std_d > 1e-12 else 0.0
    downside = daily[daily < 0]
    sortino = float(mean_d / (downside.std() + 1e-12) * np.sqrt(ppy)) \
        if downside.size > 1 else 0.0
    peak = np.maximum.accumulate(nav)
    max_dd = float(((peak - nav) / peak).max()) if peak[-1] > 0 else 0.0
    calmar = float(annual_ret / (max_dd + 1e-12)) if max_dd > 1e-9 else 0.0
    return {"nav": nav, "daily_ret": pnl, "total_ret": total_ret,
            "annual_ret": annual_ret, "annual_vol": annual_vol,
            "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd,
            "calmar": calmar, "turnover": turnover_sum / max(len(ts), 1),
            "n": len(ts)}


# ── 简化风险模型（Barra 风格）─────────────────────────────────────────────

def risk_model(daily_ret: np.ndarray, style_exposure: np.ndarray | None = None,
               style_returns: np.ndarray | None = None) -> dict:
    """简化风险模型：组合波动/协方差 + 风格暴露风险分解。

    Args:
        daily_ret: 组合日收益序列 [T]。
        style_exposure: 组合对各风格的暴露 [T, K]（可选）。
        style_returns: 风格日收益 [T, K]（可选）。

    Returns:
        {vol, var, sharpe_style: {风格: 风险贡献占比}}（含风格时）
        或 {vol, var}（无风格数据）。
    """
    r = np.asarray(daily_ret, dtype=np.float64)
    out: dict = {"vol": float(np.std(r) * np.sqrt(244)),
                 "var": float(np.var(r))}
    if style_exposure is not None and style_returns is not None:
        ex = np.asarray(style_exposure, dtype=np.float64)
        sr = np.asarray(style_returns, dtype=np.float64)
        n = min(len(r), len(ex), len(sr))
        if n >= 20:
            # 风格暴露波动 → 风险贡献（简化：暴露×风格收益标准差）
            contrib = np.std(sr[:n], axis=0) * np.mean(np.abs(ex[:n]), axis=0)
            total = float(contrib.sum())
            out["style_risk"] = {
                f"style_{i}": float(c / total) if total > 1e-12 else 0.0
                for i, c in enumerate(contrib)
            }
    return out


def performance(perf: dict, bench_ret: np.ndarray | None = None) -> dict:
    """组合绩效报告（含超额/IR，若给基准）。"""
    out = dict(perf)
    if bench_ret is not None and len(bench_ret) == len(perf["daily_ret"]):
        daily = np.asarray(perf["daily_ret"], dtype=np.float64)
        b = np.asarray(bench_ret, dtype=np.float64)
        ex = daily - b
        mean_ex, std_ex = float(ex.mean()), float(ex.std())
        out["excess_ret"] = float(np.prod(1.0 + ex) - 1.0)
        out["info_ratio"] = float(mean_ex / std_ex * np.sqrt(244)) \
            if std_ex > 1e-12 else 0.0
    return out
