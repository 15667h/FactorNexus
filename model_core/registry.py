"""
model_core/registry.py -- 声明式注册层（Registration_Interface, R10）

提供 Feature 与 Operator 的同构声明式注册机制：每个 Feature/Operator 表达为
单个声明条目（`FeatureSpec` / `OperatorSpec`），通过 `Registry` 追加到有序列表
尾部。注册时做同步校验，遵循「先校验、全部通过才追加」的原子性约定——任何一个
校验失败都不会改动注册表，保证 Formula_Vocabulary 保持不变。

校验与异常映射（对应 requirements R10.5–R10.8）：
  - 重复名                          -> DuplicateNameError   (R10.5)
  - 声明 arity 与 transform 实际操作数不符 -> ArityMismatchError (R10.6)
  - 缺必填字段                       -> MissingFieldError    (R10.7)
  - arity 非 0..10 整数              -> InvalidArityError    (R10.8)

name 约定：非空字符串，长度 1..64（R10.1, R10.2）。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import torch

# arity 声明区间（R10.1）：允许 0..10；进 VM 执行时另有 1..3 的更严约束（见 vocab/vm）。
_ARITY_MIN = 0
_ARITY_MAX = 10

# name 长度约束（R10.1, R10.2）
_NAME_MIN_LEN = 1
_NAME_MAX_LEN = 64


# ── 异常类型（对应 design「错误类型模型」）─────────────────────────────────

class RegistrationError(Exception):
    """注册层错误基类。"""


class DuplicateNameError(RegistrationError):
    """注册名与既有 token 冲突（R10.5）。"""


class ArityMismatchError(RegistrationError):
    """声明 arity 与 transform 实际操作数不符（R10.6）。"""


class MissingFieldError(RegistrationError):
    """注册条目缺必填字段（R10.7）。"""


class InvalidArityError(RegistrationError):
    """arity 非 0..10 整数（R10.8）。"""


# ── 声明条目（frozen dataclass）─────────────────────────────────────────

@dataclass(frozen=True)
class FeatureSpec:
    """Feature 声明条目。

    name:     1..64 字符、非空、唯一的字符串标识。
    category: 类别标签（trend/momentum/volatility/volume/reversal/channel/
              statistical/cross_sectional），用于报告分组与类别覆盖校验。
    compute:  计算函数，签名 `(raw_dict: dict) -> Tensor[N, T]`。
    """
    name: str
    category: str
    compute: Callable[[dict], "torch.Tensor"]


@dataclass(frozen=True)
class OperatorSpec:
    """Operator 声明条目。

    name:      1..64 字符、非空、唯一的字符串标识。
    arity:     声明区间 0..10 的整数；实际入 VM 执行时另要求 1..3。
    transform: 变换函数，签名 `(*operands: Tensor[N, T]) -> Tensor[N, T]`。
    """
    name: str
    arity: int
    transform: Callable[..., "torch.Tensor"]


# ── 校验辅助 ────────────────────────────────────────────────────────────

def _validate_name(name) -> None:
    """校验 name 为 1..64 字符的非空字符串（R10.1, R10.2）。

    缺失（None/空/纯空白）视为缺字段 -> MissingFieldError。
    """
    if name is None or (isinstance(name, str) and name.strip() == ""):
        raise MissingFieldError("缺少必填字段: name")
    if not isinstance(name, str):
        raise MissingFieldError(
            f"字段 name 必须为字符串，实际类型为 {type(name).__name__}"
        )
    if not (_NAME_MIN_LEN <= len(name) <= _NAME_MAX_LEN):
        raise MissingFieldError(
            f"字段 name 长度必须在 {_NAME_MIN_LEN}..{_NAME_MAX_LEN} 之间，"
            f"实际长度为 {len(name)}"
        )


def _observed_arity(transform: Callable) -> int | None:
    """观测 transform 实际消费的操作数个数。

    统计必填的位置参数（无默认值、非 *args/**kwargs）。若签名不可解析或含
    可变位置参数（*args），返回 None 表示「无法确定」，此时跳过 arity 匹配校验。
    """
    try:
        sig = inspect.signature(transform)
    except (TypeError, ValueError):
        return None

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return None  # 可变位置参数，无法确定确切操作数

    count = 0
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if p.default is inspect.Parameter.empty:
                count += 1
    return count


# ── 注册表 ──────────────────────────────────────────────────────────────

class Registry:
    """Feature / Operator 的有序声明式注册表。

    维护两个有序列表（按注册顺序），并用集合保证 token 名称跨 Feature/Operator
    全局唯一。注册遵循「先校验、全部通过才追加」的原子性约定。
    """

    def __init__(self) -> None:
        self._feature_specs: list[FeatureSpec] = []
        self._operator_specs: list[OperatorSpec] = []
        # 跨 Feature/Operator 的全局唯一名称集合，用于 O(1) 重名检测
        self._names: set[str] = set()

    # ── 只读视图 ────────────────────────────────────────────────────────

    @property
    def feature_specs(self) -> tuple[FeatureSpec, ...]:
        return tuple(self._feature_specs)

    @property
    def operator_specs(self) -> tuple[OperatorSpec, ...]:
        return tuple(self._operator_specs)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._feature_specs)

    @property
    def operator_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._operator_specs)

    def __contains__(self, name: str) -> bool:
        return name in self._names

    # ── 注册接口 ────────────────────────────────────────────────────────

    def register_feature(self, spec: FeatureSpec) -> None:
        """注册单个 Feature（R10.2）。校验全部通过后才追加，否则注册表不变。"""
        # 1) 必填字段校验（R10.7）：name / category / compute
        self._require_fields(
            spec,
            {"name": "name", "category": "category", "compute": "compute"},
        )
        # 2) name 合法性（R10.2）
        _validate_name(spec.name)
        # 3) 重名校验（R10.5）
        self._check_duplicate(spec.name)

        # 全部通过 -> 原子追加
        self._feature_specs.append(spec)
        self._names.add(spec.name)

    def register_operator(self, spec: OperatorSpec) -> None:
        """注册单个 Operator（R10.1）。校验全部通过后才追加，否则注册表不变。"""
        # 1) 必填字段校验（R10.7）：name / arity / transform
        #    arity 为 0 是合法值，故用「是否为 None」判断缺失而非真值判断。
        self._require_fields(
            spec,
            {"name": "name", "arity": "arity", "transform": "transform"},
        )
        # 2) name 合法性（R10.1）
        _validate_name(spec.name)
        # 3) arity 类型与范围（R10.8）：必须为 0..10 的整数（bool 不算整数）
        self._validate_arity(spec.arity)
        # 4) arity 与 transform 实际操作数一致性（R10.6）
        observed = _observed_arity(spec.transform)
        if observed is not None and observed != spec.arity:
            raise ArityMismatchError(
                f"算子 '{spec.name}' 声明 arity={spec.arity}，"
                f"但 transform 实际消费 {observed} 个操作数"
            )
        # 5) 重名校验（R10.5）
        self._check_duplicate(spec.name)

        # 全部通过 -> 原子追加
        self._operator_specs.append(spec)
        self._names.add(spec.name)

    # ── 内部校验 ────────────────────────────────────────────────────────

    @staticmethod
    def _require_fields(spec, fields: dict[str, str]) -> None:
        """校验 spec 上每个必填字段非 None（R10.7）。"""
        for attr, label in fields.items():
            if getattr(spec, attr, None) is None:
                raise MissingFieldError(f"缺少必填字段: {label}")

    @staticmethod
    def _validate_arity(arity) -> None:
        """校验 arity 为 0..10 的整数（R10.8）。bool 是 int 的子类，需显式排除。"""
        if isinstance(arity, bool) or not isinstance(arity, int):
            raise InvalidArityError(
                f"arity 必须为 {_ARITY_MIN}..{_ARITY_MAX} 的整数，"
                f"实际类型为 {type(arity).__name__}"
            )
        if not (_ARITY_MIN <= arity <= _ARITY_MAX):
            raise InvalidArityError(
                f"arity 必须为 {_ARITY_MIN}..{_ARITY_MAX} 的整数，实际为 {arity}"
            )

    def _check_duplicate(self, name: str) -> None:
        """跨 Feature/Operator 全局重名检测（R10.5）。"""
        if name in self._names:
            raise DuplicateNameError(f"注册名冲突: '{name}' 已存在")
