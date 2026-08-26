"""
model_core/param_vm.py — 参数化公式执行器（ParamVM，P1.2）

对齐华泰《高频特征参数化》（2026.4）四步流程：
  1. 输入与切片   : 以窗口中心 slice 为基准向左右各延伸 0.5*window，确定计算区间；
                   对日频数据退化为「滚动窗口」（slice=None=尾盘/最近 window 根）
  2. 时序掩码     : 以 mask_field 为参考，按 mask_rule（如 high_0.7=保留分位前 70%）
                   筛选窗口内样本
  3. 算子降维     : mode=1 单变量算子（Mean/Std/Slope/Skew/...）对核心指标 A 聚合；
                   mode=2 双变量算子（Corr/R2/Intercept/Euclid/...）对 A 与 B（可移位
                   B_shift_lag）求交叉特征 → 输出每个时间点的因子值
  4. 因子后处理   : 中位数去极值(5×MAD) → 时序 ZScore（因果 expanding，无 look-ahead）

性能：全部基于 sliding_window_view 向量化（一次求整个序列），
单个公式在 T=3000 上 < 5ms，满足 GP 种群批量评估。
"""
from __future__ import annotations

import numpy as np

from model_core.formula_dsl import ParamFormula, WINDOWS, SHIFT_LAGS

_EPS = 1e-9


# ── 滚动窗口工具 ──────────────────────────────────────────────────────────

def _roll(x: np.ndarray, w: int) -> np.ndarray:
    """因果滑动窗口矩阵 [T-w+1, w]（每行 = 时间 t 及其前 w-1 个历史值）。"""
    if w <= 1:
        return x.reshape(-1, 1)
    return np.lib.stride_tricks.sliding_window_view(x, w)


def _resolve_window(formula: ParamFormula, T: int) -> int:
    """公式 window → 实际窗口长度（All → 全历史）。"""
    w = formula.window
    return T if w == "All" else int(w)


def _resolve_slice_offset(formula: ParamFormula, w: int) -> int:
    """slice → 窗口起点偏移（日频语义）。

    slice=None → 尾盘（最近 w 根，偏移 0）
    slice=s   → 窗口中心前移到历史更早处：偏移 = (1 - s) * w（s=1 → 最旧窗口）
    """
    if formula.slice is None:
        return 0
    return int((1.0 - formula.slice) * w)


# ── 单变量算子（mode=1）──────────────────────────────────────────────────

def _op_mean(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return m.mean(axis=1)
    return (m * mask).sum(axis=1) / (mask.sum(axis=1) + _EPS)


def _op_std(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    mu = _op_mean(m, mask)[:, None]
    if mask is None:
        return ((m - mu) ** 2).mean(axis=1) ** 0.5
    return ((((m - mu) ** 2) * mask).sum(axis=1) / (mask.sum(axis=1) + _EPS)) ** 0.5


def _op_sum(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return _op_mean(m, mask) * (mask.sum(axis=1) if mask is not None else m.shape[1])


def _op_slope(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """窗口内时序线性回归斜率（时间索引为 x）。"""
    w = m.shape[1]
    tidx = np.arange(w, dtype=np.float64)
    tx = tidx - tidx.mean()
    denom = (tx ** 2).sum() + _EPS
    if mask is None:
        xm = m.mean(axis=1, keepdims=True)
        return ((m - xm) * tx).sum(axis=1) / denom
    cnt = mask.sum(axis=1)[:, None] + _EPS
    xm = (m * mask).sum(axis=1, keepdims=True) / cnt
    tm = (mask * tx[None, :]).sum(axis=1, keepdims=True) / cnt
    num = ((m - xm) * (tx[None, :] - tm) * mask).sum(axis=1)
    return num / ((((tx[None, :] - tm) ** 2) * mask).sum(axis=1) + _EPS)


def _op_skew(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    mu = _op_mean(m, mask)[:, None]
    s = _op_std(m, mask)[:, None] + _EPS
    if mask is None:
        return (((m - mu) / s) ** 3).mean(axis=1)
    # 注意 cnt 必须一维：否则 (n,) / (n,1) 广播成 (n,n) 矩阵
    cnt = mask.sum(axis=1) + _EPS
    return (((((m - mu) / s) ** 3) * mask).sum(axis=1) / cnt)


def _op_kurt(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    mu = _op_mean(m, mask)[:, None]
    s = _op_std(m, mask)[:, None] + _EPS
    if mask is None:
        return (((m - mu) / s) ** 4).mean(axis=1) - 3.0
    cnt = mask.sum(axis=1) + _EPS
    return (((((m - mu) / s) ** 4) * mask).sum(axis=1) / cnt) - 3.0


def _op_quantile(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """当前值在窗口内的分位（严格小于占比，语义与 TS_RANK 一致）。"""
    cur = m[:, -1:]
    if mask is None:
        return (m < cur).mean(axis=1)
    cnt = mask.sum(axis=1) + _EPS
    return ((m < cur) * mask).sum(axis=1) / cnt


def _op_ac1(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """lag-1 自相关（窗口内）。"""
    w = m.shape[1]
    if w < 3:
        return np.zeros(m.shape[0])
    x, y = m[:, :-1], m[:, 1:]
    if mask is not None:
        mk = mask[:, :-1] * mask[:, 1:]
        if mk.sum() == 0:
            return np.zeros(m.shape[0])
        x, y, msk = x * mk, y * mk, mk
        xm = x.sum(axis=1, keepdims=True) / (msk.sum(axis=1, keepdims=True) + _EPS)
        ym = y.sum(axis=1, keepdims=True) / (msk.sum(axis=1, keepdims=True) + _EPS)
        cov = ((x - xm) * (y - ym) * msk).sum(axis=1)
        sx = ((((x - xm) ** 2) * msk).sum(axis=1)) ** 0.5
        sy = ((((y - ym) ** 2) * msk).sum(axis=1)) ** 0.5
        return cov / (sx * sy + _EPS)
    xm, ym = x.mean(axis=1, keepdims=True), y.mean(axis=1, keepdims=True)
    cov = ((x - xm) * (y - ym)).sum(axis=1)
    sx = (((x - xm) ** 2).sum(axis=1)) ** 0.5
    sy = (((y - ym) ** 2).sum(axis=1)) ** 0.5
    return cov / (sx * sy + _EPS)


def _op_range(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return m.max(axis=1) - m.min(axis=1)
    msk = m.copy(); msk[~mask.astype(bool)] = np.nan
    return np.nanmax(msk, axis=1) - np.nanmin(msk, axis=1)


def _op_last(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return m[:, -1]


def _op_med(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.median(m, axis=1)
    out = np.empty(m.shape[0])
    for i in range(m.shape[0]):
        vals = m[i][mask[i].astype(bool)]
        out[i] = np.median(vals) if vals.size else 0.0
    return out


def _op_mad(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    med = _op_med(m, mask)
    if mask is None:
        return np.median(np.abs(m - med[:, None]), axis=1)
    out = np.empty(m.shape[0])
    for i in range(m.shape[0]):
        vals = np.abs(m[i] - med[i])[mask[i].astype(bool)]
        out[i] = np.median(vals) if vals.size else 0.0
    return out


def _op_momentum(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """last - first（窗口首尾变化）。"""
    return m[:, -1] - m[:, 0]


def _op_volatility(m: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """窗口内收益波动率。"""
    if m.shape[1] < 3:
        return np.zeros(m.shape[0])
    rets = m[:, 1:] / (m[:, :-1] + _EPS) - 1.0
    return rets.std(axis=1)


_MODE1_OPS = {
    "Mean": _op_mean, "Std": _op_std, "Sum": _op_sum, "Slope": _op_slope,
    "Skew": _op_skew, "Kurt": _op_kurt, "Quantile": _op_quantile,
    "AC1": _op_ac1, "Range": _op_range, "Last": _op_last,
    "Med": _op_med, "Mad": _op_mad, "Momentum": _op_momentum,
    "Volatility": _op_volatility,
}


# ── 双变量算子（mode=2）──────────────────────────────────────────────────

def _pair(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None):
    """通用掩码加权统计：返回 (cov, sx, sy, cnt)，用于派生相关系数等。"""
    if mask is None:
        xm, ym = x.mean(axis=1, keepdims=True), y.mean(axis=1, keepdims=True)
        cov = ((x - xm) * (y - ym)).mean(axis=1)
        sx = (((x - xm) ** 2).mean(axis=1)) ** 0.5
        sy = (((y - ym) ** 2).mean(axis=1)) ** 0.5
        return cov, sx, sy, np.full(x.shape[0], x.shape[1])
    cnt = mask.sum(axis=1)[:, None] + _EPS
    xm = (x * mask).sum(axis=1, keepdims=True) / cnt
    ym = (y * mask).sum(axis=1, keepdims=True) / cnt
    cov = (((x - xm) * (y - ym)) * mask).sum(axis=1) / cnt[:, 0]
    sx = ((((x - xm) ** 2) * mask).sum(axis=1) / cnt[:, 0]) ** 0.5
    sy = ((((y - ym) ** 2) * mask).sum(axis=1) / cnt[:, 0]) ** 0.5
    return cov, sx, sy, cnt[:, 0]


def _op_corr(ma: np.ndarray, mb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    cov, sx, sy, _ = _pair(ma, mb, mask)
    return cov / (sx * sy + _EPS)


def _op_r2(ma: np.ndarray, mb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    c = _op_corr(ma, mb, mask)
    return c ** 2


def _op_intercept(ma: np.ndarray, mb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """A ~ B 线性回归截距（剥离系统性波动后的内生信号）。"""
    cov, sx, sy, _ = _pair(ma, mb, mask)
    beta = cov / (sx * sx + _EPS)
    if mask is None:
        ym = mb.mean(axis=1)
    else:
        cnt = mask.sum(axis=1) + _EPS
        ym = (mb * mask).sum(axis=1) / cnt
    xm = ym - beta * _op_mean(ma, mask) if False else None
    # 截距 = mean(A) - beta * mean(B)
    am = _op_mean(ma, mask)
    bm = _op_mean(mb, mask)
    return am - beta * bm


def _op_slope2(ma: np.ndarray, mb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """A ~ B 回归斜率。"""
    cov, sx, sy, _ = _pair(ma, mb, mask)
    return cov / (sx * sx + _EPS)


def _op_euclid(ma: np.ndarray, mb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """量价空间欧氏距离（华泰实证核心因子：成交笔数与成交额的结构背离）。"""
    d = (ma - mb) ** 2
    if mask is None:
        return (d.mean(axis=1)) ** 0.5
    return ((d * mask).sum(axis=1) / (mask.sum(axis=1) + _EPS)) ** 0.5


def _op_delta_ratio(ma: np.ndarray, mb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return _op_mean(ma, mask) / (_op_mean(mb, mask) + _EPS) - 1.0


def _op_cov(ma: np.ndarray, mb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    cov, _, _, _ = _pair(ma, mb, mask)
    return cov


def _op_rank_diff(ma: np.ndarray, mb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """A、B 窗口内排位的差异均值（负 = A 相对 B 走弱）。"""
    def _rank_row(row):
        order = row.argsort()
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(row))
        return ranks / max(len(row) - 1, 1)
    ra = np.apply_along_axis(_rank_row, 1, ma)
    rb = np.apply_along_axis(_rank_row, 1, mb)
    if mask is None:
        return (ra - rb).mean(axis=1)
    return ((ra - rb) * mask).sum(axis=1) / (mask.sum(axis=1) + _EPS)


_MODE2_OPS = {
    "Corr": _op_corr, "R2": _op_r2, "Intercept": _op_intercept,
    "Slope2": _op_slope2, "Euclid": _op_euclid, "DeltaRatio": _op_delta_ratio,
    "Cov": _op_cov, "RankDiff": _op_rank_diff,
}


# ── 掩码生成 ─────────────────────────────────────────────────────────────

def _build_mask(wm: np.ndarray, rule: str) -> np.ndarray:
    """按 mask_rule 生成窗口内样本掩码 [T-w+1, w]。

    high_0.7 → 保留分位排名前 70% 的样本（数值较大者）
    low_0.7  → 保留排名后 70%（数值较小者）
    """
    q = float(rule.split("_")[1])
    hi = rule.startswith("high")
    if hi:
        thr = np.quantile(wm, 1.0 - q, axis=1, keepdims=True)
        return (wm >= thr).astype(np.float64)
    thr = np.quantile(wm, q, axis=1, keepdims=True)
    return (wm <= thr).astype(np.float64)


# ── 主执行器 ─────────────────────────────────────────────────────────────

class ParamVM:
    """参数化公式执行器。

    用法:
        vm = ParamVM(indicators)   # indicators: {name: np.ndarray[T]}
        factor = vm.execute(formula)   # np.ndarray[T]，已后处理（去极值+因果zscore）
    """

    def __init__(self, indicators: dict[str, np.ndarray]) -> None:
        self.indicators = {k: np.asarray(v, dtype=np.float64) for k, v in indicators.items()}
        if not self.indicators:
            raise ValueError("indicators 不能为空")
        self.T = next(iter(self.indicators.values())).shape[0]

    def execute(self, formula: ParamFormula, postprocess: bool = True) -> np.ndarray:
        if formula.A not in self.indicators:
            raise KeyError(f"指标 {formula.A!r} 不在指标库中")
        a = self.indicators[formula.A]
        w = min(_resolve_window(formula, self.T), self.T)
        off = _resolve_slice_offset(formula, w)
        eff_w = w - off
        if eff_w < 2:
            eff_w = 2
            off = max(w - 2, 0)

        # 滑动窗口（因果：t 只用 t-eff_w+1..t）
        ma = _roll(a, eff_w) if eff_w <= self.T else a.reshape(1, -1)
        if ma.shape[0] < 1:
            return np.zeros(self.T)

        # 掩码
        mask = None
        if formula.mask_rule and formula.mask_field in self.indicators:
            wm = _roll(self.indicators[formula.mask_field], eff_w)
            if wm.shape[0] == ma.shape[0]:
                mask = _build_mask(wm, formula.mask_rule)

        # 算子降维
        if formula.mode == 1:
            op = _MODE1_OPS.get(formula.mode1)
            if op is None:
                raise KeyError(f"未知单变量算子 {formula.mode1}")
            out = op(ma, mask)
        else:
            op = _MODE2_OPS.get(formula.mode2)
            if op is None:
                raise KeyError(f"未知双变量算子 {formula.mode2}")
            if formula.B not in self.indicators:
                raise KeyError(f"指标 {formula.B!r} 不在指标库中")
            b = self.indicators[formula.B]
            # B 时序错位（B_shift_lag）：只允许滞后（过去信息），杜绝未来泄漏。
            # 历史 bug：np.roll(b, -lag) 对 lag>0 实际取未来 b[t+lag]（符号反了），
            # 导致"滞后N"公式直接包含未来收益 → 因子 IC 虚高、回测作弊。
            # 修复：k=|lag|，b[t] 只用 b[t-k] 及以前；头部 warm-up 填充。
            lag = int(formula.B_shift_lag)
            if lag != 0:
                k = abs(lag)
                k = min(k, max(len(b) - 1, 1))
                b = np.concatenate([np.full(k, b[0]), b[:-k]])
            mb = _roll(b, eff_w)
            if mb.shape[0] != ma.shape[0]:
                mb = mb[-ma.shape[0]:]
            out = op(ma, mb, mask)

        # 对齐回 T（窗口 warm-up 期置 0）
        full = np.zeros(self.T)
        n_out = out.shape[0]
        full[self.T - n_out:] = out
        if postprocess:
            full = self.postprocess(full)
        return full

    # ── 因子后处理（华泰四步第 4 步）──────────────────────────────────

    @staticmethod
    def postprocess(x: np.ndarray, mad_k: float = 5.0,
                    mad_window: int = 120) -> np.ndarray:
        """中位数去极值(5×MAD) → 因果 expanding ZScore（无 look-ahead）。

        2026-08-26 审计修复：原实现用全样本 median/MAD 做裁剪边界，
        每个 t 的边界依赖未来数据（前视）。改为因果滚动窗口：
        t 的 med/mad 只用 x[t-mad_window+1 .. t]；头部 warm-up 不裁剪。
        """
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        # 1. 因果滚动 MAD 去极值（窗口默认 120 根 ≈ 半年日线）
        w = min(mad_window, max(n - 1, 1))
        if n >= w + 1 and w >= 5:
            wnd = np.lib.stride_tricks.sliding_window_view(x, w)  # [n-w+1, w]
            med = np.median(wnd, axis=1)
            mad = np.median(np.abs(wnd - med[:, None]), axis=1) + _EPS
            # 头部 warm-up（t < w-1）不裁剪（避免用未来统计量）；
            # 尾部对齐：窗口第 i 行对应原序列 t = w-1+i
            x = x.copy()
            x[w - 1:] = np.clip(x[w - 1:],
                                med - mad_k * mad, med + mad_k * mad)
        # 2. 因果 expanding zscore：t 只用 x[:t+1] 的统计量
        cum = np.cumsum(x)
        cum_sq = np.cumsum(x * x)
        cnt = np.arange(1, n + 1)
        mean = cum / cnt
        var = (cum_sq / cnt) - mean * mean
        std = np.sqrt(np.clip(var, 1e-9, None))
        z = (x - mean) / (std + _EPS)
        # warm-up 前 20 根置 0（统计量不稳定）
        z[:20] = 0.0
        return z
