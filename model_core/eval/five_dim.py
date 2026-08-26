"""
model_core/eval/five_dim.py — AlphaEval 五维因子评价（对齐 arXiv 2508.13174）

五维：
  1. 预测力 PPS   : β·IC + (1-β)·RankIC（单标的用滚动时序 IC/RankIC，β=0.5）
  2. 时间稳定性   : 月频 IC 的 IR（mean/std）、正 IC 月占比、IC 半衰期 τ（指数拟合）
  3. 鲁棒性       : 随机剔除 20% 样本 5 次的 IC 波动 + 参数抖动敏感性（越小越好）
  4. 金融逻辑     : 十分组收益单调性（Spearman 相关，越大越好）
  5. 多样性       : 与因子库既有因子的最大 |corr|（越小越好；由调用方传入库）

输出 FiveDimScore（0-1 归一化分项 + 综合分），供入库/雷达图展示。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9
_BARS_PER_YEAR = 244


# ── 基础统计 ──────────────────────────────────────────────────────────────

def _seq_ic(factor: np.ndarray, ret: np.ndarray, window: int = 20) -> np.ndarray:
    """滚动时序 IC 序列（因果窗口相关）。"""
    n = min(len(factor), len(ret))
    f, r = factor[:n], ret[:n]
    ics = []
    for t in range(window, n):
        x, y = f[t - window:t], r[t - window:t]
        xm, ym = x - x.mean(), y - y.mean()
        sx, sy = (xm ** 2).mean() ** 0.5, (ym ** 2).mean() ** 0.5
        if sx < _EPS or sy < _EPS:
            continue
        ics.append((xm * ym).mean() / (sx * sy))
    return np.array(ics)


def _seq_rankic(factor: np.ndarray, ret: np.ndarray, window: int = 20) -> np.ndarray:
    """滚动 RankIC（Spearman）序列。"""
    n = min(len(factor), len(ret))
    f, r = factor[:n], ret[:n]

    def _rank(x: np.ndarray) -> np.ndarray:
        order = x.argsort()
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(x))
        return ranks / max(len(x) - 1, 1)

    ics = []
    for t in range(window, n):
        x, y = _rank(f[t - window:t]), _rank(r[t - window:t])
        xm, ym = x - x.mean(), y - y.mean()
        sx, sy = (xm ** 2).mean() ** 0.5, (ym ** 2).mean() ** 0.5
        if sx < _EPS or sy < _EPS:
            continue
        ics.append((xm * ym).mean() / (sx * sy))
    return np.array(ics)


def _ic_half_life(ics: np.ndarray, n_period: int = 22) -> float:
    """IC 半衰期：指数拟合 IC(t) ≈ IC0 · 2^(-t/τ)，返回 τ（期数）。

    把滚动 IC 序列按 n_period 分块取均值，拟合 ln(IC_block) 对时间。
    """
    if len(ics) < 2 * n_period:
        return float("inf")
    n_blocks = len(ics) // n_period
    blocks = np.array([ics[i * n_period:(i + 1) * n_period].mean() for i in range(n_blocks)])
    blocks = np.abs(blocks)
    if (blocks <= _EPS).all():
        return float("inf")
    t = np.arange(n_blocks, dtype=float)
    y = np.log(np.maximum(blocks, _EPS))
    denom = ((t - t.mean()) ** 2).sum()
    if denom < _EPS:
        return float("inf")
    slope = ((t - t.mean()) * (y - y.mean())).sum() / denom
    if slope >= 0:  # IC 不衰减（或增强）
        return float("inf")
    tau = -np.log(2.0) / slope
    return float(np.clip(tau, 1.0, 1e6))


# ── 五维评分 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FiveDimScore:
    pps: float       # 预测力 [0,1]
    stability: float # 时间稳定性 [0,1]
    robustness: float  # 鲁棒性 [0,1]（1=最稳）
    logic: float     # 金融逻辑 [0,1]
    diversity: float # 多样性 [0,1]（1=最新颖）
    total: float     # 综合 = 加权平均（预测力×0.3 + 稳定性×0.25 + 鲁棒×0.2 + 逻辑×0.15 + 多样×0.1）

    def as_dict(self) -> dict:
        return {"pps": self.pps, "stability": self.stability,
                "robustness": self.robustness, "logic": self.logic,
                "diversity": self.diversity, "total": self.total}


def five_dim_evaluate(
    factor: np.ndarray,
    ret: np.ndarray,
    library_factors: list[np.ndarray] | None = None,
    beta: float = 0.5,
    n_perturb: int = 5,
) -> FiveDimScore:
    """五维评分。

    Args:
        factor: 因子序列 [T]
        ret:    未来收益 [T]（与 factor 对齐 t+1）
        library_factors: 因子库已有因子列表（用于多样性维度；None → diversity=1）
        beta:   PPS 中 IC/RankIC 权重
        n_perturb: 鲁棒性测试扰动次数
    """
    n = min(len(factor), len(ret))
    if n < 60:
        return FiveDimScore(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    f, r = factor[:n].astype(float), ret[:n].astype(float)

    # 1. 预测力 PPS
    ic = _seq_ic(f, r)
    ric = _seq_rankic(f, r)
    ic_mean = ic.mean() if ic.size else 0.0
    ric_mean = ric.mean() if ric.size else 0.0
    pps_raw = beta * abs(ic_mean) + (1 - beta) * abs(ric_mean)
    # 归一化标尺：A股日频时序滚动 IC 的高分位 ~0.08（截面 IC 高分位 ~0.05）
    pps = float(np.clip(pps_raw / 0.08, 0.0, 1.0))

    # 2. 时间稳定性：月频 IR + 正IC占比 + 半衰期
    if ic.size >= 44:
        monthly = np.array([ic[i:i + 22].mean() for i in range(0, len(ic) - 21, 22)])
    else:
        monthly = ic
    ir = monthly.mean() / (monthly.std() + _EPS) if monthly.size else 0.0
    pos_ratio = (ic > 0).mean() if ic.size else 0.0
    tau = _ic_half_life(ic)
    stability = 0.5 * float(np.clip(abs(ir) / 2.0, 0.0, 1.0)) \
        + 0.3 * float(pos_ratio) \
        + 0.2 * float(np.clip(2.0 / (1.0 + tau / 22.0), 0.0, 1.0))  # 半衰期≥1年→低分
    stability = float(np.clip(stability, 0.0, 1.0))

    # 3. 鲁棒性：剔除 20% 样本扰动
    rng = np.random.default_rng(0)
    ics_pert = []
    for _ in range(n_perturb):
        keep = rng.choice(n, size=int(n * 0.8), replace=False)
        ics_pert.append(_seq_ic(f[keep], r[keep]).mean())
    pert_std = np.std(ics_pert) if ics_pert else 0.0
    pert_mean = abs(np.mean(ics_pert)) + _EPS
    robustness = float(np.clip(1.0 - pert_std / (pert_mean + _EPS) * 3.0, 0.0, 1.0))

    # 4. 金融逻辑：十分组收益单调性
    dec = np.quantile(f, np.linspace(0, 1, 11))
    group_ret = []
    for i in range(10):
        m = (f >= dec[i]) & (f < dec[i + 1]) if i < 9 else (f >= dec[9])
        if m.sum() >= 3:
            group_ret.append(r[m].mean())
    if len(group_ret) >= 5:
        order = np.argsort(np.argsort(group_ret))
        mono, _ = _spearman(order, np.arange(len(group_ret)))
        logic = float(np.clip(abs(mono), 0.0, 1.0))
    else:
        logic = 0.0

    # 5. 多样性（与因子库最大 |corr| 越小越好）
    if library_factors:
        best = 0.0
        for lf in library_factors:
            m = min(len(f), len(lf))
            if m < 20:
                continue
            x, y = f[:m], lf[:m]
            xm, ym = x - x.mean(), y - y.mean()
            sx, sy = (xm ** 2).mean() ** 0.5, (ym ** 2).mean() ** 0.5
            if sx < _EPS or sy < _EPS:
                continue
            best = max(best, abs((xm * ym).mean() / (sx * sy)))
        diversity = float(np.clip(1.0 - best, 0.0, 1.0))
    else:
        diversity = 1.0

    total = (0.30 * pps + 0.25 * stability + 0.20 * robustness
             + 0.15 * logic + 0.10 * diversity)
    return FiveDimScore(pps, stability, robustness, logic, diversity, float(total))


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman 相关（小数组直接算秩）。"""
    def _rank(a: np.ndarray) -> np.ndarray:
        order = a.argsort()
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(a))
        return ranks
    rx, ry = _rank(x).astype(float), _rank(y).astype(float)
    rxm, rym = rx - rx.mean(), ry - ry.mean()
    denom = (rxm ** 2).sum() ** 0.5 * (rym ** 2).sum() ** 0.5
    if denom < _EPS:
        return 0.0, 1.0
    r = (rxm * rym).sum() / denom
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1 - r * r, _EPS))
    return float(r), float(t)
