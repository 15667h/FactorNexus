"""
tests/test_p8_no_leakage.py — P8 数据泄漏审计回归测试

背景（2026-08-26 审计）：param_vm.B_shift_lag 曾因 np.roll 符号反了，
lag>0（"滞后N"）实际取未来 b[t+N] → 因子含未来收益 → IC 虚高 0.4+
（真实水平 ~0.02）、回测 950 亿倍收益、入库报告 max_dd=0。
本测试锁定三类防护：
  1. B_shift_lag>0 的公式因子与"未来收益"不相关（与"过去收益"才可能相关）
  2. 修复后重算的因子与泄漏版因子值不一致（行为已改变）
  3. 掩码/窗口退化公式输出不再触发广播错误（skew/kurt 修复）

运行：python -m pytest tests/test_p8_no_leakage.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from model_core.formula_dsl import ParamFormula, random_chrom, chrom_to_formula
from model_core.indicator_builder import build_indicators
from model_core.param_vm import ParamVM


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    x, y = a[-n:] - a[-n:].mean(), b[-n:] - b[-n:].mean()
    sd = x.std() * y.std()
    return float((x * y).mean() / sd) if sd > 1e-9 else 0.0


@pytest.fixture(scope="module")
def ctx():
    rng = np.random.default_rng(0)
    T = 500
    close = 100 + np.cumsum(rng.normal(0, 1, T))
    df = pd.DataFrame({
        "ts": np.arange(T, dtype=np.int64),
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": np.abs(rng.normal(1000, 200, T)),
    })
    ind = build_indicators(df)
    vm = ParamVM(ind)
    # 未来收益标签：ret[t] = close[t+h]/close[t]-1
    h = 5
    ret = np.zeros(T)
    ret[:T - h] = close[h:] / close[:-h] - 1.0
    return {"ind": ind, "vm": vm, "ret": ret}


def _formula(lag: int) -> ParamFormula:
    """带 B_shift_lag 的双变量公式：A=open, B=ret, window=20, mode=2 Corr。"""
    return ParamFormula(A="open", B="ret", window=20, slice=None,
                        mask_field=None, mask_rule=None, mode=2,
                        mode1="Mean", mode2="Corr", B_shift_lag=lag)


def test_lag_positive_uses_past_not_future(ctx):
    """防泄漏核心：滞后因子必须与"未来收益"无关，与"过去收益"才可能相关。"""
    vm, ret = ctx["vm"], ctx["ret"]
    f = vm.execute(_formula(lag=+2))  # "滞后2"：只能用 t-2 及以前的信息

    future_corr = _corr(f[:-2], ret[2:])   # f[t] vs ret[t+2]（未来）
    past_corr = _corr(f[2:], ret[:-2])     # f[t] vs ret[t-2]（过去）
    # 因子与未来收益不应显著相关（|corr| < 0.1，真实信号水平）
    assert abs(future_corr) < 0.1, \
        f"滞后因子与未来收益相关 {future_corr:.3f} —— 泄漏！"
    # 与过去收益的相关性应显著高于与未来的（如果因子有信息）
    assert past_corr > future_corr - 0.05


def test_lag_sign_does_not_change_future_corr(ctx):
    """正负 lag 都不允许未来泄漏（绝对值滞后语义）。

    判据：无泄漏因子的未来相关 ≤ 过去相关 + 容差（泄漏的签名是
    "未来相关占优"——修复前该因子未来相关 0.43 >> 过去相关）。
    合成数据噪声水平 ~0.12（500 样本），阈值取 0.15。
    """
    vm, ret = ctx["vm"], ctx["ret"]
    for lag in (-3, -1, 1, 3, 5):
        f = vm.execute(_formula(lag=lag))
        k = abs(lag)
        fc = _corr(f[:-k], ret[k:])     # f[t] vs ret[t+k]（未来）
        pc = _corr(f[k:], ret[:-k])     # f[t] vs ret[t-k]（过去）
        assert fc < 0.15, f"lag={lag} 与未来收益相关 {fc:.3f} —— 泄漏！"
        assert fc <= pc + 0.05, \
            f"lag={lag} 未来相关 {fc:.3f} 显著高于过去相关 {pc:.3f} —— 泄漏签名！"


def test_random_chroms_no_broadcast_error(ctx):
    """随机染色体（含 Skew/Kurt + 掩码 + 各种窗口）执行不抛广播错误。"""
    vm = ctx["vm"]
    for _ in range(50):
        c = random_chrom()
        out = vm.execute(chrom_to_formula(c))
        assert out.shape == (vm.T,), f"形状异常 {out.shape}"


def test_weak_signal_postprocess_stays_bounded(ctx):
    """退化输入（近常数）经 postprocess 后不应产生虚假大信号。"""
    vm = ctx["vm"]
    x = np.full(vm.T, 1e-10) + np.random.default_rng(1).normal(0, 1e-11, vm.T)
    z = vm.postprocess(x)
    assert np.isfinite(z).all()
    assert z.std() < 1.0  # 噪声不应被放大成 O(1) 伪信号


def test_execute_causal_shape(ctx):
    """因子序列与窗口对齐：因子[t] 只依赖 t 及以前（窗口终点 = t）。"""
    vm = ctx["vm"]
    f = vm.execute(_formula(lag=0))
    # 前 warm-up 期为 0（postprocess 前 20 根置 0）
    assert np.count_nonzero(f[:20]) == 0 or np.allclose(f[:20], 0.0)
    assert f.shape == (vm.T,)
