"""
model_core/portfolio/barra_risk.py — Barra 风格风险模型（P20）

机构级风险模型（对标 CNE5/CNE6 简化版）：
  1. 风格因子库：市值(log)、动量(20日)、波动率(20日)、换手/量能、Beta(对市场)
  2. ledoit_wolf_shrinkage : 协方差矩阵收缩估计（Ledoit & Wolf 2004 解析解）
  3. barra_risk_model      : 组合风险分解
      总风险² = 风格风险² + 特质风险²
      风格风险 = w' (B Σ_f B') w；特质风险 = w' diag(σ_ε²) w
  4. style_exposures       : 计算组合对风格因子的暴露

用法：
    from model_core.portfolio.barra_risk import (
        ledoit_wolf_shrinkage, style_exposures, barra_risk_model)
"""
from __future__ import annotations

import numpy as np


# ── Ledoit-Wolf 收缩协方差（解析解）──────────────────────────────────────

def ledoit_wolf_shrinkage(X: np.ndarray) -> np.ndarray:
    """Ledoit & Wolf (2004) 收缩协方差估计。

    Σ_shrunk = δ·F + (1-δ)·S
      S = 样本协方差，F = 目标（对角化：保留方差、去协方差）
      δ = 收缩强度（解析最优）

    Args:
        X: 收益矩阵 [T, N]（每列一个资产）。

    Returns:
        收缩后协方差 [N, N]（正定，可逆）。
    """
    X = np.asarray(X, dtype=np.float64)
    T, N = X.shape
    if T < 2 or N < 2:
        return np.cov(X, rowvar=False) if N > 1 else np.array([[1.0]])
    # 去均值
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / T                     # 样本协方差
    # 目标 F = diag(S)
    F = np.diag(np.diag(S))
    # δ* = Σ_ij Var(s_ij - f_ij) / Σ_ij (s_ij - f_ij)²
    num = 0.0
    den = 0.0
    for i in range(N):
        for j in range(N):
            sij = S[i, j]
            fij = F[i, j]
            # s_ij 的抽样方差（四阶矩）
            x_i = Xc[:, i]
            x_j = Xc[:, j]
            a = x_i * x_j - sij
            var_sij = float(np.mean(a ** 2)) - float(np.mean(a)) ** 2
            num += var_sij
            den += (sij - fij) ** 2
    delta = float(np.clip(num / (den + 1e-12), 0.0, 1.0))
    return delta * F + (1.0 - delta) * S


# ── 风格因子库（横截面）───────────────────────────────────────────────────

def build_style_factors(returns: np.ndarray, volumes: np.ndarray | None = None,
                        market_ret: np.ndarray | None = None) -> np.ndarray:
    """构建风格因子暴露矩阵 [T, N, K]。

    风格（Barra 简化五因子）：
      SIZE  = log(近20日均成交额)（市值代理）
      MOM   = 近20日累计收益（动量）
      VOL   = 近20日收益标准差（波动率，负向）
      LIQ   = 近5日/20日均量比（流动性/换手）
      BETA  = 对市场收益的滚动 Beta（无市场时 = 1 常数）

    Returns:
        exposures [T, N, K]（每期截面 zscore 标准化）。
    """
    r = np.asarray(returns, dtype=np.float64)
    T, N = r.shape
    K = 5
    ex = np.zeros((T, N, K))
    eps = 1e-9
    for t in range(T):
        lo = max(0, t - 19)
        seg = r[lo:t + 1]
        # SIZE：累计成交额代理（收益绝对值累积）
        size = np.log(np.sum(np.abs(seg), axis=0) + eps)
        mom = np.sum(seg, axis=0)
        vol = np.std(seg, axis=0)
        if volumes is not None:
            v = np.asarray(volumes, dtype=np.float64)
            v5 = np.mean(v[max(0, t - 4):t + 1], axis=0)
            v20 = np.mean(v[lo:t + 1], axis=0)
            liq = v5 / (v20 + eps)
        else:
            liq = np.abs(r[t]) / (vol + eps)
        beta = np.ones(N)
        if market_ret is not None:
            m = np.asarray(market_ret, dtype=np.float64)[lo:t + 1]
            if len(m) >= 10 and np.std(m) > 1e-9:
                for i in range(N):
                    if np.std(seg[:, i]) > 1e-9:
                        beta[i] = float(np.cov(seg[:, i], m)[0, 1]
                                        / np.var(m))
        mat = np.column_stack([size, mom, -vol, liq, beta])   # VOL 负向
        for k in range(K):
            col = mat[:, k]
            sd = np.std(col)
            if sd > 1e-9:
                ex[t, :, k] = (col - np.mean(col)) / sd
    return ex


def style_exposures(weights: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """组合对风格因子的时间序列暴露 [T, K] = w_t' X_t。"""
    w = np.asarray(weights, dtype=np.float64)      # [T, N]
    T, N = w.shape
    K = exposures.shape[2]
    out = np.zeros((T, K))
    for t in range(T):
        out[t] = w[t] @ exposures[t]
    return out


# ── Barra 风险分解 ────────────────────────────────────────────────────────

def barra_risk_model(returns: np.ndarray, weights: np.ndarray,
                     exposures: np.ndarray | None = None) -> dict:
    """Barra 风格风险分解（简化 CNE 结构）。

    模型：r_t = X_t f_t + ε_t
      总风险² = w'(B Σ_f B' + Σ_ε)w
      Σ_f = 风格因子收益协方差（Ledoit-Wolf 收缩）
      Σ_ε = 特质风险（残差方差对角阵）

    Args:
        returns: 收益矩阵 [T, N]。
        weights: 权重矩阵 [T, N]（时间序列组合权重）。
        exposures: 风格暴露 [T, N, K]（None → 自动构建）。

    Returns:
        {total_vol, style_vol, idio_vol, r2, style_corr}
    """
    r = np.asarray(returns, dtype=np.float64)
    T, N = r.shape
    w = np.asarray(weights, dtype=np.float64)
    if exposures is None:
        exposures = build_style_factors(r)
    K = exposures.shape[2]

    # 1) 风格因子收益：F_t = X_t^+ r_t（横截面回归系数）
    F = np.zeros((T, K))
    resid = np.zeros((T, N))
    computed = np.zeros(T, dtype=bool)   # 真正完成回归的日（M11）
    for t in range(T):
        X = exposures[t]                               # [N, K]
        ok = np.isfinite(X).all(axis=1) & np.isfinite(r[t])
        if ok.sum() < K + 2:
            continue
        Xv, rv = X[ok], r[t][ok]
        try:
            beta, *_ = np.linalg.lstsq(Xv, rv, rcond=None)
            F[t] = beta
            resid[t][ok] = rv - Xv @ beta
            computed[t] = True
        except np.linalg.LinAlgError:
            pass
    # 2) 收缩协方差（风格）——M11 修复：旧实现用 isfinite 过滤，回归被跳过日
    #    F=0 是有限值会被误纳入，稀释/扭曲协方差；改用 computed 掩码。
    F_ok = F[computed]
    if len(F_ok) < 20:
        return {"total_vol": 0.0, "style_vol": 0.0, "idio_vol": 0.0,
                "r2": 0.0, "style_corr": np.zeros((K, K))}
    cov_f = ledoit_wolf_shrinkage(F_ok)
    # 3) 特质风险（残差方差）——同样只用真正回归的日
    var_eps = np.var(resid[computed], axis=0) + 1e-12
    # 4) 组合平均权重
    w_mean = np.nanmean(w, axis=0)
    w_mean = w_mean / (np.abs(w_mean).sum() + 1e-12)
    X_mean = np.nanmean(exposures, axis=0)             # [N, K]
    # 组合风格风险：w'(B Σ_f B')w
    B = X_mean
    style_var = float(w_mean @ (B @ cov_f @ B.T) @ w_mean)
    idio_var = float(w_mean @ (np.diag(var_eps)) @ w_mean)
    total_var = style_var + idio_var
    return {
        "total_vol": float(np.sqrt(max(total_var, 0.0))),
        "style_vol": float(np.sqrt(max(style_var, 0.0))),
        "idio_vol": float(np.sqrt(max(idio_var, 0.0))),
        "r2": float(style_var / max(total_var, 1e-12)),
        "style_corr": cov_f,
    }
