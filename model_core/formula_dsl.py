"""
model_core/formula_dsl.py — 参数化万能公式 DSL（华泰式，P1 核心）

背景（华泰《高频特征参数化：分钟级可解释因子挖掘框架》2026.4）：
  传统遗传规划"算子随机排列"易产生黑盒表达式；华泰改用固定范式万能公式，
  每个参数都有经济学含义与离散取值范围，在保障挖掘灵活性的同时抑制过拟合。

本模块定义 **参数化公式**（区别于现有 token 公式）：

    Factor = (A, B, window, slice, mask_field, mask_rule, mode, mode1, mode2, B_shift_lag)

    - A / B / mask_field : 输入指标（A=核心指标，B=双变量辅助指标，可 None）
    - window            : 时间窗口长度（{5,10,20,60,120,All}）
    - slice             : 窗口中心位置（{0.0..1.0 步长 0.1, None=尾盘}）
    - mask_rule         : 时序掩码规则（{None, high_0.3, high_0.5, high_0.7,
                           low_0.3, low_0.5, low_0.7}）
    - mode              : 1=单变量算子，2=双变量交叉算子
    - mode1             : 单变量时序降维算子（Mean/Std/Sum/Slope/Skew/Kurt/AC1...）
    - mode2             : 双变量交互算子（Corr/R2/Intercept/Euclid/DeltaRatio...）
    - B_shift_lag       : B 指标时序错位（-5..+5，0=对齐）

所有参数均以 **整数索引** 编码为染色体（离散化），供 GP 引擎直接进化。

执行：ParamVM（model_core/vm.py 扩展）按 输入切片→时序掩码→算子降维→后处理 四步执行。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── 枚举表（参数 → 候选值，索引即染色体基因值）─────────────────────────────

WINDOWS: tuple = (5, 10, 20, 60, 120, "All")
SLICES: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, None)
MASK_RULES: tuple = (None, "high_0.3", "high_0.5", "high_0.7", "low_0.3", "low_0.5", "low_0.7")
SHIFT_LAGS: tuple = tuple(range(-5, 6))  # -5..+5

# 单变量算子（mode=1）—— 对核心指标做时序聚合，输出日频值
MODE1_OPS: tuple = (
    "Mean", "Std", "Sum", "Slope", "Skew", "Kurt", "Quantile",
    "AC1", "Range", "Last", "Med", "Mad", "Momentum", "Volatility",
)
# 双变量算子（mode=2）—— 两指标交叉特征
MODE2_OPS: tuple = (
    "Corr", "R2", "Intercept", "Slope2", "Euclid", "DeltaRatio",
    "Cov", "RankDiff",
)

# 指标库占位（实际 100+ 指标来自 model_core/features.py 的 FEATURE_REGISTRY；
# 这里定义与 features 对接的接口名，GP 引擎通过 features 层解析）
INDICATORS: tuple = (
    # 价量基础
    "close", "open", "high", "low", "volume", "amount",
    # 衍生
    "ret", "ret5", "ret20", "atr", "rvol", "hl_range", "vol_regime",
    "rsi14", "macd_hist", "boll_pos", "boll_width", "obv_slope", "mfi14",
    "willr_14", "cci_14", "roc_12", "typical_dev", "ema_ratio_12_26",
    "trend_strength_50", "price_pos_50", "trix_15", "ppo", "ult_osc",
    # 微观结构（新增）
    "num_trades", "amt_per_trade", "trade_size_reg_intercept",
    "amt_vol_euclid", "num_trades_ac1",
)

# 染色体长度 = 10 个基因（全部整数索引）
CHROM_LEN = 10

# 各基因的取值空间（用于随机初始化/变异边界）
GENE_SPACES: tuple[tuple, ...] = (
    INDICATORS,       # 0: A
    INDICATORS,       # 1: B（同指标库）
    WINDOWS,          # 2: window
    SLICES,           # 3: slice
    INDICATORS,       # 4: mask_field
    MASK_RULES,       # 5: mask_rule
    (1, 2),           # 6: mode
    MODE1_OPS,        # 7: mode1
    MODE2_OPS,        # 8: mode2
    SHIFT_LAGS,       # 9: B_shift_lag
)


@dataclass(frozen=True)
class ParamFormula:
    """参数化公式（解码后的可读形式）。"""
    A: str
    B: str | None
    window: int | str
    slice: float | None
    mask_field: str | None
    mask_rule: str | None
    mode: int
    mode1: str
    mode2: str | None
    B_shift_lag: int

    def to_chrom(self) -> list[int]:
        """编码为整数染色体（10 基因，全部为参数空间索引）。"""
        return [
            _idx(INDICATORS, self.A),
            _idx(INDICATORS, self.B) if self.B else 0,
            _idx(WINDOWS, self.window),
            _idx(SLICES, self.slice),
            _idx(INDICATORS, self.mask_field) if self.mask_field else 0,
            _idx(MASK_RULES, self.mask_rule) if self.mask_rule else 0,
            0 if self.mode == 1 else 1,           # mode 索引
            _idx(MODE1_OPS, self.mode1),
            _idx(MODE2_OPS, self.mode2) if self.mode2 else 0,
            _idx(SHIFT_LAGS, self.B_shift_lag),
        ]

    def describe(self) -> str:
        """人类可读的中文描述（Web 展示/LLM 解释用）。"""
        window_s = "全区间" if self.window == "All" else f"{self.window}期"
        slice_s = "尾盘" if self.slice is None else f"切片{self.slice:.1f}"
        mask_s = f"，按{self.mask_field}掩码{self.mask_rule}" if self.mask_rule else ""
        if self.mode == 1:
            return (f"{self.A}的{self.mode1}（{window_s}·{slice_s}{mask_s}）")
        return (f"{self.A}与{self.B}(滞后{self.B_shift_lag})的{self.mode2}"
                f"（{window_s}·{slice_s}{mask_s}）")


# ── 工具函数 ──────────────────────────────────────────────────────────────

def _idx(space: tuple, value) -> int:
    # H9 修复：去掉对 None 的短路。旧实现 `if value is None: return 0` 会让
    # None 在空间中的真实索引（如 SLICES[11]=None）永远被编码成 0（SLICES[0]），
    # 导致 slice=None 的公式往返后漂移成 slice=0.0。调用方对 None 的清理应
    # 在 to_chrom 的三元表达式中完成（如 B/mask_field），而非在 _idx 短路。
    # None 若存在于 space（SLICES/MASK_RULES），由 space.index(None) 给出真实索引。
    try:
        return space.index(value)
    except ValueError:
        raise ValueError(f"值 {value!r} 不在参数空间 {space[:5]}... 中")


def normalize_chrom(chrom: list[int] | tuple) -> list[int]:
    """规范化染色体：根据 mode/mask_rule 基因把无意义基因置 0（无效位清零）。

    - mode=1（单变量）时 B(1) 与 mode2(8) 基因无意义，置 0；
    - mask_rule=None（基因 5==0，MASK_RULES[0]=None）时 mask_field(4)
      无意义——chrom_to_formula 强制 mask_field=None，故编码也必须清 0。
    保证「规范化 → 解码 → 再编码」恒等（GP 交叉/变异后必须调用）。
    H9 修复：此前未清 mask_field，且 _idx 对 None 短路，实测 200 随机种子
    18.5% 编解码不一致（slice=None 与 mask_field 基因静默丢失）。
    """
    c = [int(x) for x in chrom]
    if len(c) != CHROM_LEN:
        raise ValueError(f"染色体长度必须为 {CHROM_LEN}，实际 {len(c)}")
    if c[6] % 2 == 0:  # mode=1
        c[1] = 0  # B
        c[8] = 0  # mode2
    if c[5] == 0:  # mask_rule=None（MASK_RULES[0]=None）
        c[4] = 0  # mask_field 无意义，清零
    return c


def chrom_to_formula(chrom: list[int] | tuple) -> ParamFormula:
    """染色体（10 整数）→ 参数化公式。"""
    chrom = normalize_chrom(chrom)
    a_i, b_i, w_i, s_i, mf_i, mr_i, mode_i, m1_i, m2_i, lag_i = chrom
    mode = 2 if mode_i % 2 == 1 else 1  # mode 基因是索引：0→1, 1→2
    return ParamFormula(
        A=str(INDICATORS[a_i % len(INDICATORS)]),
        # mode=1 时 B 基因无意义，恒为 None（保证编解码往返一致）
        B=str(INDICATORS[b_i % len(INDICATORS)]) if mode == 2 else None,
        window=WINDOWS[w_i % len(WINDOWS)],
        slice=SLICES[s_i % len(SLICES)],
        mask_field=str(INDICATORS[mf_i % len(INDICATORS)]) if mr_i != 0 else None,
        mask_rule=MASK_RULES[mr_i % len(MASK_RULES)] if mr_i != 0 else None,
        mode=mode,
        mode1=str(MODE1_OPS[m1_i % len(MODE1_OPS)]),
        # mode=1 时 mode2 基因无意义，恒为 None（保证编解码往返一致）
        mode2=str(MODE2_OPS[m2_i % len(MODE2_OPS)]) if mode == 2 else None,
        B_shift_lag=SHIFT_LAGS[lag_i % len(SHIFT_LAGS)],
    )


def random_chrom(rng=None) -> list[int]:
    """随机初始化一条染色体（各基因独立均匀采样）。

    支持三种 rng：random.Random / numpy RandomState / numpy Generator；
    None 时用全局 random 模块（行为与旧版一致）。
    """
    import random as _r
    if rng is None:
        return [_r.randrange(len(space)) for space in GENE_SPACES]
    if hasattr(rng, "randrange"):            # random.Random / RandomState
        return [rng.randrange(len(space)) for space in GENE_SPACES]
    # numpy Generator（无 randrange，用 integers）
    return [int(rng.integers(len(space))) for space in GENE_SPACES]


def formula_signature(f: ParamFormula) -> str:
    """公式签名（去重/入库哈希用）。"""
    return repr((f.A, f.B, f.window, f.slice, f.mask_field, f.mask_rule,
                 f.mode, f.mode1, f.mode2, f.B_shift_lag))


if __name__ == "__main__":
    # 自检
    import random
    random.seed(42)
    c = random_chrom(random)
    f = chrom_to_formula(c)
    print("染色体:", c)
    print("公式  :", f.describe())
    assert f.to_chrom() == normalize_chrom(c)
    print("编解码往返 OK")
