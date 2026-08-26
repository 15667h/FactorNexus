"""
model_core/vocab.py -- Formula_Vocabulary 集成与确定性版本（R3）

本模块把 Formula_Vocabulary 从「手工维护的特征名元组 + 手工版本字符串」迁移为
由注册层（`model_core.registry.Registry`）驱动、版本确定性派生的实现：

  - `feature_names` 来自 `features.FEATURE_REGISTRY`（有序）。
  - `operator_names` 来自 `ops.OPERATOR_REGISTRY`（有序）。
  - token id 分段：feature id ∈ [0, F-1]，operator id ∈ [F, F+O-1]，两段严格
    不相交（`operator_offset == feature_count`，R3.3）。
  - 构建时用集合校验 token 名称全局唯一、无缺失/重复/多余（R3.1、R3.2）。
  - `VOCAB_VERSION` 由有序 token 名称列表确定性派生（R3.4、R3.5）：
        VOCAB_VERSION = "v" + sha256("\n".join(token_names)).hexdigest()[:12]
    相同有序列表 → 相同版本；任意组成/顺序变化 → 不同版本。
  - `FORMULA_VOCAB.verify(artifact_version)`：版本不匹配抛
    `VocabVersionMismatchError`，拒绝且不消费任何 token（R3.7）。
  - `VOCAB_SCHEMA_TAG`：人类可读的 schema 标签，仅供日志展示，不参与兼容判定。

import 方向说明：`features.py` / `ops.py` 只依赖 `.registry`，本模块从二者读取
注册表视图不构成循环依赖。下游 `vm.py` / `config.py` /
`alphagpt.py` / `engine.py` 对 `FEATURE_NAMES` / `FORMULA_VOCAB` / `VOCAB_VERSION`
的 import 保持兼容。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .features import FEATURE_REGISTRY
from .ops import OPERATOR_REGISTRY

# 人类可读 schema 标签（仅供日志/报告展示，不参与兼容性判定，R3.5）
VOCAB_SCHEMA_TAG = "4.0-registry"


# ── 版本层异常（R3.7）───────────────────────────────────────────────────

class VocabVersionMismatchError(Exception):
    """加载产物版本 ≠ 当前派生 VOCAB_VERSION（R3.7）。

    由 `FORMULA_VOCAB.verify()` 在版本不匹配时抛出；调用方应拒绝加载且不消费
    任何 token。
    """


# ── 确定性版本派生（R3.4、R3.5）─────────────────────────────────────────

def compute_vocab_version(token_names: tuple[str, ...]) -> str:
    """由有序 token 名称列表确定性派生紧凑版本标识。

    VOCAB_VERSION = "v" + sha256("\n".join(token_names)).hexdigest()[:12]

    性质：相同的有序列表 → 相同版本；任意组成或顺序变化 → 不同版本。使用换行
    作为稳定分隔符，避免名称拼接歧义。
    """
    joined = "\n".join(token_names)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return "v" + digest[:12]


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        # feature id 段 [0, F-1]、operator id 段 [F, F+O-1] 严格不相交（R3.3）
        return self.feature_count

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)

    @property
    def version(self) -> str:
        """当前词表组成确定性派生的 VOCAB_VERSION（R3.4）。"""
        return compute_vocab_version(self.token_names)

    def verify(self, artifact_version: str) -> None:
        """校验产物版本与当前派生版本一致（R3.7）。

        不匹配抛 `VocabVersionMismatchError`，调用方据此拒绝加载、不消费任何
        token。匹配则静默返回。
        """
        current = self.version
        if artifact_version != current:
            raise VocabVersionMismatchError(
                f"词表版本不匹配：产物版本 {artifact_version!r} != "
                f"当前派生版本 {current!r}；旧 checkpoint / best_strategy.json "
                f"需重新训练/重建后加载"
            )


# ── 构建 FORMULA_VOCAB（由 registry 派生）与完整性校验（R3.1、R3.2）──────

def _build_formula_vocab() -> FormulaVocab:
    """由 FEATURE_REGISTRY / OPERATOR_REGISTRY 构建词表并做完整性校验。

    校验（用集合，R3.1、R3.2）：
      - feature / operator 名称各自无重复；
      - feature 与 operator 名称跨段全局唯一（无交集）；
      - size == F + O（无缺失/多余）。
    任一校验失败即抛错，不产出不一致的词表。
    """
    feature_names = tuple(FEATURE_REGISTRY.feature_names)
    operator_names = tuple(OPERATOR_REGISTRY.operator_names)

    feat_set = set(feature_names)
    op_set = set(operator_names)

    # 段内唯一
    if len(feat_set) != len(feature_names):
        dup = sorted({n for n in feature_names if feature_names.count(n) > 1})
        raise ValueError(f"feature 名称存在重复: {dup}")
    if len(op_set) != len(operator_names):
        dup = sorted({n for n in operator_names if operator_names.count(n) > 1})
        raise ValueError(f"operator 名称存在重复: {dup}")

    # 跨段全局唯一（feature/operator 名称不得冲突）
    overlap = feat_set & op_set
    if overlap:
        raise ValueError(f"feature 与 operator 名称冲突: {sorted(overlap)}")

    vocab = FormulaVocab(feature_names=feature_names, operator_names=operator_names)

    # 计数一致性：size == F + O，无缺失/重复/多余（R3.2）
    expected = len(feature_names) + len(operator_names)
    if vocab.size != expected:
        raise ValueError(
            f"词表计数不一致: size={vocab.size} != F+O={expected}"
        )
    # 全局 token 名称唯一（无缺失/重复/多余）
    if len(set(vocab.token_names)) != vocab.size:
        raise ValueError("token 名称存在重复或缺失，词表完整性校验失败")

    return vocab


FORMULA_VOCAB = _build_formula_vocab()

# 由注册表导出有序特征名视图（保持下游 import 兼容）
FEATURE_NAMES = FORMULA_VOCAB.feature_names

# 词表版本：由有序 token 名称列表确定性派生（R3.4、R3.5）。
# 特征/算子的组成或顺序变化都会改变本值，旧 checkpoint / best_strategy.json 将
# 因版本不匹配而被 verify() 拒绝加载。
VOCAB_VERSION = FORMULA_VOCAB.version
