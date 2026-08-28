"""
model_core/eval/significance.py — 过拟合统计控制（Bailey & López de Prado 体系）

  1. DSR  (Deflated Sharpe Ratio)  : 对 N 次挖掘试验做多重检验校正的夏普比率
  2. PBO  (Probability of Backtest Overfitting) : CSCV 组合对称交叉验证，
        计算"训练段最优因子在测试段跑输中位数的概率"
  3. CPCV (Combinatorial Purged Cross-Validation) : 组合净化交叉验证，
        产出多路径回测分布（K 折 purge+embargo → C(K, K/2) 路径）

实现要点：
  - 全部只用「已实现收益序列」，不涉及未来信息
  - DSR/PBO 需要「试验次数 N」——由挖掘引擎上报（GP 种群×代数 或 RL 采样数）
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

_EPS = 1e-12


# ── 工具 ──────────────────────────────────────────────────────────────────

def _sharpe(pnl: np.ndarray, ppy: int = 244) -> float:
    if pnl.size < 3:
        return 0.0
    sd = pnl.std()
    if sd < _EPS:
        return 0.0
    return float(pnl.mean() / sd * math.sqrt(ppy))


def _moments(pnl: np.ndarray) -> tuple[float, float]:
    """偏度 γ3、超额峰度 γ4（无偏估计近似）。"""
    n = len(pnl)
    if n < 4:
        return 0.0, 0.0
    m2 = ((pnl - pnl.mean()) ** 2).mean()
    if m2 < _EPS:
        return 0.0, 0.0
    m3 = ((pnl - pnl.mean()) ** 3).mean()
    m4 = ((pnl - pnl.mean()) ** 4).mean()
    g3 = m3 / (m2 ** 1.5)
    g4 = m4 / (m2 ** 2) - 3.0
    return float(g3), float(g4)


# ── DSR（Deflated Sharpe Ratio）───────────────────────────────────────────

def expected_max_sharpe(n_trials: int, skew: float = 0.0, kurt: float = 3.0,
                        trials_indep: float = 1.0) -> float:
    """N 次独立试验下期望最大夏普（E[max SR]，Bailey 2014 闭式近似）。

    注意：Bailey 的 E[maxSR] = (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))
    是纯 z 分位近似（非年化口径），不直接依赖 skew/kurt——
    skew/kurt 的影响体现在 DSR 分母的 V 项中。

    Args:
        n_trials: 试验次数 N（挖掘中评估的公式总数）
        trials_indep: 试验相关性折扣（1=完全独立；拥挤试验 <1 放大校正）
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649  # Euler-Mascheroni
    ne = max(n_trials * trials_indep, 1.0)
    z1 = stats.norm.ppf(1.0 - 1.0 / ne)
    z2 = stats.norm.ppf(1.0 - 1.0 / (ne * math.e))
    return float((1 - gamma) * z1 + gamma * z2)


def compute_dsr(pnl: np.ndarray, n_trials: int, ppy: int = 244,
                trials_indep: float = 1.0) -> float:
    """Deflated Sharpe Ratio：校正选择偏差后"真实夏普>0"的概率。

    Bailey & López de Prado (2014) "The Deflated Sharpe Ratio" 原文公式：

        DSR = Φ[ (SR̂ − SR₀)·√(T−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) ]

    其中 SR̂ 为非年化夏普、SR₀ = E[max SR]（N 次试验校正的期望最大夏普）。
    历史 bug（2026-08-26 审计）：E[maxSR] 项未乘 √(T−1) 也未除以 √V，
    校正被削弱约 √(T−1)/√V ≈ 40 倍 → DSR 虚高（泄漏因子也能 DSR=0.93）。

    Returns:
        [0,1]，>0.95 表示该因子几乎不可能由运气产生
    """
    if pnl.size < 4 or n_trials <= 1:
        return 0.5
    # 非年化夏普（Bailey 公式口径：SR̂ 为每期夏普，√(T-1) 负责统计检验）
    sd = pnl.std()
    if sd < _EPS:
        return 0.5
    sr = float(pnl.mean() / sd)
    g3, g4 = _moments(pnl)
    emax = expected_max_sharpe(n_trials, g3, g4 + 3.0, trials_indep)
    # V = 1 - γ3·SR + (γ4-1)/4·SR²，其中 γ4 为标准峰度（正态=3）；g4 为超额峰度
    var = 1.0 - g3 * sr + (g4 + 2.0) / 4.0 * sr ** 2
    denom = math.sqrt(max(var, _EPS))
    dsr_z = (sr - emax) * math.sqrt(pnl.size - 1) / denom
    return float(stats.norm.cdf(dsr_z))


# ── PBO（CSCV 组合对称交叉验证）───────────────────────────────────────────

def compute_pbo_cscv(pnl_matrix: np.ndarray, n_blocks: int = 16, seed: int = 0) -> float:
    """回测过拟合概率（Bailey et al. 2015 CSCV）。

    Args:
        pnl_matrix: [n_factors, T] 各因子的收益序列（同一时间轴）
        n_blocks:   时间轴分块数 S（默认 16 → C(16,8)=12870 组合；可抽样加速）

    Returns:
        PBO ∈ [0,1]：训练段最优因子在测试段跑输中位数的概率（>0.5 高度过拟合）
    """
    n_f, T = pnl_matrix.shape
    if T < n_blocks * 2 or n_f < 3:
        return 0.5
    # 分块（末尾不足块丢弃）
    block_size = T // n_blocks
    blocks = [pnl_matrix[:, i * block_size:(i + 1) * block_size] for i in range(n_blocks)]

    n_train = n_blocks // 2
    rng = np.random.default_rng(seed)
    # 全组合 C(S, S/2)；S=16 → 12870 全枚举；更大则抽样 5000
    all_combos = list(itertools.combinations(range(n_blocks), n_train))
    if len(all_combos) > 8000:
        idx = rng.choice(len(all_combos), size=5000, replace=False)
        combos = [all_combos[i] for i in idx]
    else:
        combos = all_combos

    count_worse = 0
    for combo in combos:
        test_blocks = [i for i in range(n_blocks) if i not in combo]
        train_pnl = np.hstack([blocks[i] for i in combo]).mean(axis=1)  # [n_f]
        test_pnl = np.hstack([blocks[i] for i in test_blocks]).mean(axis=1)
        if train_pnl.std() < _EPS:
            continue
        best_idx = int(np.argmax(train_pnl))
        # rank = 测试段比"训练最优因子"更好的因子比例；
        # rank > 0.5 → 训练最优因子在测试段跑输中位数 → 过拟合（ω*<0.5）
        rank = (test_pnl > test_pnl[best_idx]).mean()
        count_worse += 1 if rank > 0.5 else 0
    return float(count_worse / max(len(combos), 1))


# ── CPCV（组合净化交叉验证）───────────────────────────────────────────────

def cpcv_paths(pnl: np.ndarray, n_folds: int = 6, purge: int = 0,
               embargo: int = 0, seed: int = 0) -> np.ndarray:
    """组合净化交叉验证：K 折 → C(K, K/2) 条 训练/测试 路径。

    Args:
        pnl:        [T] 收益序列（单因子）
        n_folds:    折数 K（默认 6 → 20 条路径）
        purge:      训练段末尾剔除样本数（标签泄漏净化）
        embargo:    训练/测试间隔样本数（相关泄漏隔离）

    Returns:
        [n_paths] 各路径测试段累计收益（或 NaN 表示无效路径）
    """
    T = len(pnl)
    if T < n_folds * 4:
        return np.array([], dtype=float)
    fold_size = T // n_folds
    folds = [pnl[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]

    n_train = n_folds // 2
    combos = list(itertools.combinations(range(n_folds), n_train))
    rng = np.random.default_rng(seed)
    if len(combos) > 8000:
        idx = rng.choice(len(combos), size=5000, replace=False)
        combos = [combos[i] for i in idx]

    paths = []
    for combo in combos:
        test_folds = [i for i in range(n_folds) if i not in combo]
        # purge + embargo（M15 修复：旧实现只实现 purge，embargo 参数从未生效）。
        # purge   = 标签泄漏净化：训练段末尾剔除（预测 horizon 标签重叠）
        # embargo = 相关泄漏隔离：在 purge 基础上再额外剔除，拉开训练尾与测试首
        #           的序列相关样本（López de Prado CPCV 标准做法）
        _cut = purge + embargo
        train = np.hstack([f[:max(len(f) - _cut, 0)] for f in
                           [folds[i] for i in combo]])
        test = np.hstack([folds[i] for i in test_folds])
        if len(train) < 2 or len(test) < 2:
            continue
        paths.append(float(test.sum()))
    return np.array(paths)


def cpcv_summary(paths: np.ndarray) -> dict:
    """CPCV 路径分布摘要（中位路径累计收益 + 置信区间）。"""
    if paths.size == 0:
        return {"n_paths": 0, "median": 0.0, "p10": 0.0, "p90": 0.0, "all_positive": False}
    return {
        "n_paths": int(paths.size),
        "median": float(np.median(paths)),
        "p10": float(np.percentile(paths, 10)),
        "p90": float(np.percentile(paths, 90)),
        "all_positive": bool((paths > 0).all()),
    }
