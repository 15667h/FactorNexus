"""
strategy_manager/signal.py — 回测与实盘共享的信号计算模块

提供：
  compute_target_positions(factors, prev_positions)  →  连续仓位 [-1, +1] 张量
  reconcile_action(current, target)                  →  动作字符串

信号逻辑（收益优先模式，2026-07-04 重构）：
  旧模式（Neutral Band）：tanh → sign → {-1, 0, +1} 三档，天花板锁死在 1 倍仓。
  新模式（连续仓位）：factor 直接经 tanh 压缩到 (-1, +1) 作为仓位比例。
    - factor 越强 → 仓位比例越大，允许"加码"
    - 不设 Neutral Band，让模型自由决定在场时间
    - 回测与实盘共用同一逻辑，消除两者差异
  训练时用 tanh(factor) 作为连续仓位，回测也一致，避免训练/回测目标函数不对齐。
"""
from __future__ import annotations

import torch
from torch import Tensor

# ── 保留实盘用的阈值参数（实盘 Runner 可能还读取这些常量）──────────────────
ENTRY_THRESHOLD: float = 0.3
EXIT_THRESHOLD:  float = 0.1
MIN_TRADE_EXPOSURE: float = 0.05


def _min_trade_exposure() -> float:
    try:
        from config import Config
        return float(getattr(Config, "MIN_TRADE_EXPOSURE", MIN_TRADE_EXPOSURE))
    except Exception:
        return MIN_TRADE_EXPOSURE


def compute_target_positions(
    factors:        Tensor,
    prev_positions: Tensor | None = None,
) -> Tensor:
    """将因子张量转换为连续仓位 [-1, +1]（收益优先模式）。

    新逻辑：position = tanh(factor)，连续仓位，强信号→大仓。
    prev_positions 参数保留兼容性，连续模式下不影响计算。

    Args:
        factors:        [N, T] 或 [N] 的因子张量。
        prev_positions: 保留参数，连续模式下忽略。
    """
    pos = torch.tanh(factors)
    min_abs = _min_trade_exposure()
    if min_abs > 0:
        pos = torch.where(pos.abs() >= min_abs, pos, torch.zeros_like(pos))
    return pos

def compute_target_positions_stateless(factors: Tensor) -> Tensor:
    """无状态版本，供训练回测快速计算（连续仓位模式）。"""
    return compute_target_positions(factors, prev_positions=None)


def target_to_direction(target: float, min_abs: float | None = None) -> int:
    """把连续目标仓位转成 MT5 可执行方向。"""
    threshold = _min_trade_exposure() if min_abs is None else float(min_abs)
    if target >= threshold:
        return 1
    if target <= -threshold:
        return -1
    return 0


# ── 动作常量 ──────────────────────────────────────────────────────────────────
HOLD             = "HOLD"
OPEN_LONG        = "OPEN_LONG"
OPEN_SHORT       = "OPEN_SHORT"
CLOSE            = "CLOSE"
REVERSE_TO_LONG  = "REVERSE_TO_LONG"
REVERSE_TO_SHORT = "REVERSE_TO_SHORT"


def reconcile_action(current: int, target: int) -> str:
    """根据当前仓位方向和目标方向，返回应执行的动作。

    Args:
        current: 当前仓位方向，+1（多）/ -1（空）/ 0（空仓）。
        target:  目标仓位方向，+1 / -1 / 0。

    Returns:
        动作字符串，取值为模块级常量之一。
    """
    if current == target:
        return HOLD
    if current == 0:
        return OPEN_LONG if target == 1 else OPEN_SHORT
    if target == 0:
        return CLOSE
    return REVERSE_TO_LONG if target == 1 else REVERSE_TO_SHORT
