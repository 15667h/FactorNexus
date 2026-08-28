"""
model_core/portfolio/optimizer.py — 组合优化器（P19）

机构级权重优化，替代等权/因子加权选股：
  1. markowitz     : 均值-方差有效前沿（给定目标收益或风险厌恶系数）
  2. risk_parity   : 风险平价（等风险贡献，迭代求解）
  3. black_litterman: BL 模型（市场均衡先验 + 主观观点 → 后验期望收益 → 组合）
  4. efficient_frontier: 有效前沿采样（最小方差 → 最大收益）

全部基于 numpy/scipy（无额外依赖），输入为收益矩阵 [T, N]。

用法：
    from model_core.portfolio.optimizer import markowitz, risk_parity, \
        black_litterman
    w = markowitz(ret_matrix, risk_aversion=2.0)
    w = risk_parity(cov_matrix)
    w = black_litterman(cov, market_w, views={0: 0.05}, view_conf=0.5)
"""
from __future__ import annotations

import numpy as np


def _cov_returns(returns: np.ndarray) -> np.ndarray:
    """收益矩阵 [T,N] → 协方差（样本，NaN 安全）。"""
    r = np.asarray(returns, dtype=np.float64)
    ok = np.isfinite(r).all(axis=1)
    if ok.sum() < 30:
        r = np.nan_to_num(r, nan=0.0)
    else:
        r = r[ok]
    return np.cov(r, rowvar=False)


def _mean_returns(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    ok = np.isfinite(r).all(axis=1)
    if ok.sum() >= 30:
        r = r[ok]
    return np.nan_to_num(r.mean(axis=0), nan=0.0)


def _optimize_weights(mu: np.ndarray, cov: np.ndarray,
                      risk_aversion: float, target_ret: float | None,
                      long_only: bool) -> np.ndarray:
    """凸二次规划：min 0.5 w'Σw - λ·w'μ，约束 Σw=1（long_only 时 w≥0）。"""
    from scipy.optimize import minimize

    n = len(mu)
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n

    def obj(w):
        return 0.5 * w @ cov @ w - risk_aversion * w @ mu

    def eq(w):
        return np.sum(w) - 1.0

    cons = [{"type": "eq", "fun": eq}]
    if target_ret is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target_ret})
    x0 = np.full(n, 1.0 / n)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.asarray(res.x, dtype=np.float64)
    # M10 修复：纯多头（全正）按 Σ|w| 归一化等价于 Σw=1；长空（含负权重）
    # 时 Σ|w|≠1，按 Σ|w| 归一化会破坏 SLSQP 强制的 Σw=1 约束（净暴露不受控、
    # 不保证多空中性）。长空直接返回 res.x（已满足 Σw=1）。
    if long_only:
        s = float(np.abs(w).sum())
        return w / s if s > 1e-12 else x0
    net = float(w.sum())
    return w / net if abs(net) > 1e-12 else x0


def markowitz(returns: np.ndarray, risk_aversion: float = 2.0,
              target_ret: float | None = None,
              long_only: bool = True) -> np.ndarray:
    """均值-方差组合权重。

    risk_aversion 越大 → 越保守（低波动）；target_ret 给定 → 有效前沿上
    指定收益的最小方差组合。
    """
    mu = _mean_returns(returns)
    cov = _cov_returns(returns)
    return _optimize_weights(mu, cov, risk_aversion, target_ret, long_only)


def risk_parity(returns: np.ndarray | np.ndarray,
                max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """风险平价：每资产风险贡献相等（Spinu 2013 凸公式，全局收敛）。

    风险平价解 = argmin 0.5·w'Σw − Σᵢ ln(wᵢ)（Spinu 证明等价，
    凸优化无局部最优）。约束 Σw=1, w>0（下限 1e-6）。
    """
    from scipy.optimize import minimize

    # M9 修复：docstring 允许直接传协方差矩阵，但旧实现 ndim==2 时必然把
    # 协方差矩阵当收益矩阵再 np.cov 一次（二次协方差化，结果远离真实协方差）。
    # 对称方阵视为协方差；否则当作收益矩阵 [T,N] 估计协方差。
    _arr = np.asarray(returns, dtype=np.float64)
    if (_arr.ndim == 2 and _arr.shape[0] == _arr.shape[1]
            and np.allclose(_arr, _arr.T)):
        cov = _arr
    else:
        cov = _cov_returns(_arr)
    n = cov.shape[0]

    def obj(w):
        return 0.5 * w @ cov @ w - np.sum(np.log(w))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.full(n, 1.0 / n)
    res = minimize(obj, x0, method="SLSQP",
                   bounds=[(1e-6, 1.0)] * n, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = np.asarray(res.x, dtype=np.float64)
    w = w / w.sum()
    return w


def black_litterman(returns: np.ndarray, market_weights: np.ndarray,
                    views: dict[int, float] | None = None,
                    view_conf: float = 0.5, tau: float = 0.05,
                    risk_aversion: float = 2.5,
                    long_only: bool = True) -> np.ndarray:
    """Black-Litterman：市场均衡先验 + 观点 → 后验期望收益 → 组合权重。

    Args:
        returns: 收益矩阵 [T,N]（估计协方差）。
        market_weights: 市值权重（先验锚点）。
        views: {资产索引: 期望收益观点}；None = 纯均衡（即市场组合）。
        view_conf: 观点置信度（0-1，越大越信观点）。
        tau: 先验不确定性标度（BL 惯例 0.01-0.1）。
        risk_aversion: 风险厌恶（BL 后验 → 权重）。

    Returns:
        后验组合权重 [N]。
    """
    n = len(market_weights)
    cov = _cov_returns(returns)
    mu_eq = risk_aversion * cov @ np.asarray(market_weights, dtype=np.float64)
    views = views or {}
    if views:
        P = np.zeros((len(views), n))
        q = np.zeros(len(views))
        for i, (idx, val) in enumerate(views.items()):
            P[i, idx] = 1.0
            q[i] = val
        omega = np.diag(np.full(len(views),
                                (1.0 - view_conf) / max(view_conf, 1e-9)
                                * float(np.trace(cov)) / n))
        # 后验均值：μ_post = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 μ_eq + P'Ω^-1 q]
        t_inv = np.linalg.inv(tau * cov)
        A = t_inv + P.T @ np.linalg.inv(omega) @ P
        b = t_inv @ mu_eq + P.T @ np.linalg.inv(omega) @ q
        mu_post = np.linalg.solve(A, b)
    else:
        mu_post = mu_eq
    return _optimize_weights(mu_post, cov, risk_aversion, None, long_only)


def efficient_frontier(returns: np.ndarray, n_points: int = 12,
                       long_only: bool = True) -> tuple[list[float],
                                                        list[float],
                                                        list[np.ndarray]]:
    """有效前沿采样：最小方差 → 最大收益。

    Returns: (收益列表, 波动列表, 权重列表)
    """
    mu = _mean_returns(returns)
    cov = _cov_returns(returns)
    # 最小方差组合
    w_min = _optimize_weights(mu, cov, 0.0, None, long_only)
    rets, vols, ws = [], [], []
    r_min = float(w_min @ mu)
    # 单资产最大收益（long_only 上界）
    r_max = float(mu.max())
    for i in range(n_points):
        target = r_min + (r_max - r_min) * i / max(n_points - 1, 1)
        try:
            w = _optimize_weights(mu, cov, 0.0, target, long_only)
        except Exception:  # noqa: BLE001
            continue
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        rets.append(r)
        vols.append(v)
        ws.append(w)
    return rets, vols, ws


# ── 顶层风险预算：面板级优化（P19 接入组合层）───────────────────────────────

def optimize_portfolio_panel(score_panel, ret_panel,
                             method: str = "markowitz",
                             n_top: int = 5, window: int = 60,
                             rebalance: int = 5,
                             risk_aversion: float = 2.0,
                             long_short: bool = True,
                             view_conf: float = 0.5,
                             verbose: bool = False) -> "pd.DataFrame":
    """顶层风险预算：信号面板 + 滚动协方差 → 优化器权重面板。

    机构范式：候选池（信号 Top/Bottom）→ 风险模型（滚动收益协方差，
    只用 t 及以前，防前视）→ 优化器分配权重 → 持有 rebalance 日。

    Args:
        score_panel: 信号/得分面板（index=交易日, columns=股票，越大越好）。
        ret_panel:   1 日收益面板（同轴同列）。
        method:      "markowitz" | "risk_parity" | "black_litterman"。
        n_top:       多头（与空头）候选数。
        window:      协方差估计滚动窗口（交易日）。
        rebalance:   持有期（交易日；每 rebalance 日优化一次，区间内持有）。
        risk_aversion: 风险厌恶（markowitz / BL）。
        long_short:  True=多空（Bottom 做空）；False=纯多头。
        view_conf:   BL 观点置信度。

    Returns:
        权重面板（同轴；多空 Σ|w|=2，纯多头 Σw=1；NaN=空仓）。
    """
    import pandas as pd

    out = pd.DataFrame(np.nan, index=score_panel.index,
                       columns=score_panel.columns)
    idx = list(score_panel.index)
    n_days = len(idx)
    ret_aligned = ret_panel.reindex(index=idx, columns=score_panel.columns)
    for i in range(0, n_days, max(1, rebalance)):
        t = idx[i]
        row = score_panel.loc[t].astype(float)
        vals = row.dropna()
        need = n_top * 2 if long_short else n_top
        if len(vals) < max(need, 2):
            continue
        ranked = vals.sort_values(ascending=False)
        longs = ranked.head(n_top).index.tolist()
        shorts = ranked.tail(n_top).index.tolist() if long_short else []
        cands = longs + shorts
        # 风险模型：只用 t 及以前的收益（防前视）
        lo = max(0, i - window)
        rmat = ret_aligned.iloc[lo:i + 1, :].reindex(
            columns=cands).values.astype(np.float64)
        if rmat.shape[0] < 30 or np.isfinite(rmat).sum() < 30:
            continue
        try:
            w = _optimize_candidates(rmat, longs, shorts, method,
                                     risk_aversion, view_conf)
        except Exception:  # noqa: BLE001
            continue
        seg = idx[i:i + max(1, rebalance)]
        if not seg:
            break
        # 块赋值（链式赋值陷阱：逐元素 out.loc[seg[j]][c] 不写回 DataFrame）
        row_vals = np.zeros(len(score_panel.columns))
        for k, c in enumerate(cands):
            row_vals[score_panel.columns.get_loc(c)] = float(w[k])
        out.loc[seg] = row_vals
    return out


def _optimize_candidates(rmat: np.ndarray, longs: list, shorts: list,
                         method: str, risk_aversion: float,
                         view_conf: float) -> np.ndarray:
    """候选收益矩阵 → 优化权重向量（多空时 Σ|w|=2，纯多头 Σw=1）。"""
    n = len(longs) + len(shorts)
    if method == "markowitz":
        w = markowitz(rmat, risk_aversion=risk_aversion,
                      long_only=not shorts)
    elif method == "risk_parity":
        if shorts:
            wl = risk_parity(rmat[:, :len(longs)])
            ws = risk_parity(-rmat[:, len(longs):])   # 空头池取负收益同向 RP
            w = np.concatenate([wl, -ws])
        else:
            w = risk_parity(rmat)
    elif method == "black_litterman":
        # 市场先验 = 候选等权；观点 = 多空方向（Top 正 / Bottom 负，幅度 1%）
        mkt = np.full(n, 1.0 / n)
        views: dict[int, float] = {}
        for k in range(n):
            views[k] = 0.01 if k < len(longs) else -0.01
        w = black_litterman(rmat, mkt, views=views, view_conf=view_conf,
                            risk_aversion=risk_aversion,
                            long_only=not shorts)
    else:
        raise ValueError(f"未知优化器 {method}（可选: markowitz, "
                         f"risk_parity, black_litterman）")
    w = np.asarray(w, dtype=np.float64)
    if shorts:                                   # 多空：Σ|w| = 2
        s = float(np.abs(w).sum())
        return w / s * 2.0 if s > 1e-12 else w
    s = float(w.sum())                           # 纯多头：Σw = 1
    return w / s if s > 1e-12 else w
