"""
model_core/eval/report.py — FactorReport：因子完整评价报告（入库载荷）

字段：
  - 公式（chrom / describe）
  - 五维评分（FiveDimScore）
  - 经典统计（IC / RankIC / ICIR / Sharpe / MaxDD / 换手）
  - 显著性（DSR / PBO / CPCV 摘要）
  - 元数据（挖掘时间/引擎/区间）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np

from model_core.eval.five_dim import FiveDimScore, _seq_ic, _seq_rankic
from model_core.eval.significance import (
    compute_dsr, compute_pbo_cscv, cpcv_paths, cpcv_summary, _sharpe,
)


@dataclass
class FactorReport:
    chrom: list[int]
    describe: str
    five_dim: FiveDimScore
    ic: float = 0.0
    rankic: float = 0.0
    icir: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    turnover: float = 0.0
    dsr: float = 0.5
    pbo: float = 0.5
    cpcv: dict = field(default_factory=dict)
    n_trials: int = 0
    mined_at: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["five_dim"] = self.five_dim.as_dict()
        return d


def _max_drawdown(pnl: np.ndarray) -> float:
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max()
    return float(dd)


def build_factor_report(
    factor: np.ndarray,
    ret: np.ndarray,
    chrom: list[int],
    describe: str,
    n_trials: int,
    library_factors: list[np.ndarray] | None = None,
    ppy: int = 244,
    **meta,
) -> FactorReport:
    """为一条因子生成完整评价报告。

    Args:
        factor: 因子序列 [T]（已后处理）
        ret:    未来收益 [T]
        chrom:  染色体
        describe: 公式描述
        n_trials: 挖掘试验次数（DSR 多重检验校正）
        library_factors: 因子库已有因子（多样性/相关性用）
    """
    from model_core.eval.five_dim import five_dim_evaluate

    n = min(len(factor), len(ret))
    f, r = factor[:n].astype(float), ret[:n].astype(float)

    ic_arr = _seq_ic(f, r)
    ric_arr = _seq_rankic(f, r)
    ic = float(ic_arr.mean()) if ic_arr.size else 0.0
    rankic = float(ric_arr.mean()) if ric_arr.size else 0.0
    icir = float(ic / (ic_arr.std() + 1e-9)) if ic_arr.size else 0.0

    # 机构 E3：IC 衰减曲线（lag 1..10 的 Pearson IC——因子预测力随持有期衰减）
    ic_decay: list[float] = []
    if n > 20:
        for lag in range(1, 11):
            if lag >= n:
                break
            x, y = f[:-lag], r[lag:]
            xm, ym = x - x.mean(), y - y.mean()
            sd = xm.std() * ym.std()
            ic_decay.append(float((xm * ym).mean() / sd) if sd > 1e-9 else 0.0)

    # 机构 E2：分层回测（十分组 Q1-Q10 收益单调性 + 多空收益）
    group_ret: list[float] = []
    if n >= 50:
        q = np.quantile(f, np.linspace(0, 1, 11))
        for i in range(10):
            m = (f >= q[i]) & (f < q[i + 1]) if i < 9 else (f >= q[9])
            if m.sum() >= 3:
                group_ret.append(float(r[m].mean()))
        while len(group_ret) < 10:
            group_ret.append(0.0)

    # 仓位 = tanh(因子)（对齐信号口径），PnL = pos*ret
    pos = np.tanh(f)
    pnl = pos * r
    # 换手 = |Δpos| 均值
    turnover = float(np.abs(np.diff(pos)).mean()) if n > 1 else 0.0

    five_dim = five_dim_evaluate(f, r, library_factors)

    # 显著性（PBO 需要"因子×时间"矩阵，单因子时用自身分块模拟）
    pbo = compute_pbo_cscv(pnl.reshape(1, -1), n_blocks=8) if n >= 32 else 0.5
    dsr = compute_dsr(pnl, n_trials=max(n_trials, 2), ppy=ppy)
    cpcv = cpcv_summary(cpcv_paths(pnl, n_folds=6))

    meta_out = dict(meta)
    meta_out["ic_decay"] = ic_decay
    meta_out["group_returns"] = group_ret
    if len(group_ret) == 10:
        from scipy.stats import spearmanr
        mono = float(spearmanr(np.arange(10), group_ret).statistic)
        meta_out["group_monotonicity"] = mono
        meta_out["long_short_ret"] = group_ret[-1] - group_ret[0]

    return FactorReport(
        chrom=chrom,
        describe=describe,
        five_dim=five_dim,
        ic=ic, rankic=rankic, icir=icir,
        sharpe=_sharpe(pnl, ppy),
        max_dd=_max_drawdown(pnl),
        turnover=turnover,
        dsr=dsr, pbo=pbo, cpcv=cpcv,
        n_trials=n_trials,
        meta=meta_out,
    )
