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
    s = float(np.abs(w).sum())
    return w / s if s > 1e-12 else x0


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

    cov = _cov_returns(returns) if returns.ndim == 2 else \
        np.asarray(returns, dtype=np.float64)
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
