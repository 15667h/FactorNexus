"""
tests/test_p9_institutional.py — P9 机构级修复锁定测试

2026-08-26 机构对齐审计（对照 Bailey & López de Prado 2014 / 华泰金工）后修复：
  1. DSR 公式对齐原文（E[maxSR] 项随 SR̂ 一起乘 √(T−1)/√V，此前校正被削弱 ~40 倍）
  2. postprocess 去极值改因果滚动 MAD（此前全样本 median/MAD 含前视）
  3. 回测执行时点：t 收盘信号 → t+1 执行（此前收盘价即时成交有前视）

运行：python -m pytest tests/test_p9_institutional.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
from scipy import stats

from model_core.eval.significance import compute_dsr, expected_max_sharpe
from model_core.param_vm import ParamVM
from scripts.factor_backtest import backtest_factor


# ── 1. DSR 公式（Bailey 2014 原文）────────────────────────────────────────

def _dsr_old_impl(pnl, n_trials):
    """旧实现参考（审计前公式形态：E[maxSR] 未乘 √(T−1) 未除 √V）。"""
    sr = float(pnl.mean() / pnl.std())
    emax = expected_max_sharpe(n_trials)
    V = 1.0  # 近似（样本 skew/kurt 影响远小于公式形态差异）
    z_old = sr * math.sqrt(len(pnl) - 1) / math.sqrt(V) - emax
    return float(stats.norm.cdf(z_old))


def test_dsr_corrects_stronger_with_more_trials():
    """试验次数 N 越大，选择偏差校正越强，DSR 单调下降（旧实现几乎无校正）。"""
    rng = np.random.default_rng(0)
    T = 2000
    # 非年化 sr≈1.0（极强因子）：N=2 时超过 E[maxSR](2)≈0.52 → 显著；
    # N=10000 时 E[maxSR]≈3.69 → 必须坍缩（Bailey 校正的正确行为）
    pnl = rng.normal(1.0 / math.sqrt(244), 1.0 / math.sqrt(244), T)
    dsrs = [compute_dsr(pnl, n_trials=n) for n in (2, 10, 100, 1000, 10000)]
    assert all(dsrs[i] >= dsrs[i + 1] - 1e-9 for i in range(len(dsrs) - 1)), \
        "DSR 应随试验次数单调下降"
    assert dsrs[0] > 0.9, "小 N 极强因子应显著"
    assert dsrs[-1] < 0.01, "大 N 校正后必须坍缩（旧实现恒≈1 → bug）"


def test_dsr_selection_bias_correction():
    """N=383 次试验校正必须显著压低 DSR，且远低于旧实现（此前校正失效）。"""
    rng = np.random.default_rng(0)
    T = 2000
    # 非年化 sr≈0.20（强因子；泄漏时代实测 sr≈0.10 更弱）
    pnl = rng.normal(0.20 / math.sqrt(244), 1.0 / math.sqrt(244), T)
    dsr_new = compute_dsr(pnl, n_trials=383)
    dsr_old = _dsr_old_impl(pnl, 383)
    assert dsr_old > 0.9, "旧实现参考应给高 DSR（证明修复前后差异）"
    assert dsr_new < 0.5, f"选择偏差校正失效: DSR={dsr_new:.4f}（应被压低）"
    assert dsr_new < dsr_old - 0.4, "新实现必须显著低于旧实现"


def test_expected_max_sharpe_reference_values():
    """Bailey 2014 闭式近似参考值：N=100 → E[maxSR]≈2.5；N=1000 → ≈3.2。"""
    e100 = expected_max_sharpe(100)
    e1000 = expected_max_sharpe(1000)
    assert 2.3 < e100 < 2.8
    assert 3.0 < e1000 < 3.5
    assert e1000 > e100


# ── 2. postprocess 因果性（无前视）─────────────────────────────────────────

def test_postprocess_no_future_dependence():
    """修改序列尾部不影响前面所有点的 z 值（expanding zscore + 滚动 MAD 因果）。"""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 600)
    x[:20] = 0.0
    z1 = ParamVM.postprocess(x)
    x2 = x.copy()
    x2[-1] = 9999.0          # 未来出现极端值
    z2 = ParamVM.postprocess(x2)
    # 前 590 个点的 z 值必须完全一致（t 只用 t 及以前信息）
    assert np.allclose(z1[:-10], z2[:-10], atol=1e-9), "postprocess 存在前视依赖"


def test_postprocess_weak_signal_bounded():
    """微值/退化输入不会被后处理放大成 O(1) 伪信号。"""
    x = np.full(500, 1e-10) + np.random.default_rng(1).normal(0, 1e-11, 500)
    z = ParamVM.postprocess(x)
    assert np.isfinite(z).all()
    assert z.std() < 1.0


# ── 3. 回测执行时点（t 信号 → t+1 执行）───────────────────────────────────

def test_backtest_signal_executed_next_bar():
    """t 收盘信号 → t+1 执行：首日信号不产生当日收益；漂移行情下多头为正。"""
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 300))  # 正漂移
    factor = np.full(300, 10.0)  # 强烈看多信号
    b = backtest_factor(factor, close, horizon=5, cost=0.0)
    assert b["nav"][0] == 1.0, "首日信号不应产生当日收益（需 t+1 执行）"
    assert b["total_ret"] > 0, "正漂移行情下全多头滞后执行应正收益"
    # 对比：同行情下全空头应跑输全多头
    b_short = backtest_factor(np.full(300, -10.0), close, horizon=5, cost=0.0)
    assert b_short["total_ret"] < b["total_ret"]


def test_backtest_zero_signal_no_pnl():
    factor = np.zeros(200)
    close = 100 * np.cumprod(1 + np.random.default_rng(1).normal(0.001, 0.01, 300))
    b = backtest_factor(factor, close, horizon=5, cost=0.0)
    # tanh(0)=0 → 空仓 → 无收益
    assert abs(b["total_ret"]) < 1e-9
    assert abs(b["sharpe"]) < 1e-9
