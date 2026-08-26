"""
model_core/engines/llm_engine.py — LLM 多智能体因子挖掘引擎（P4）

对齐华泰《GPT因子工厂：多智能体与因子挖掘》（2024.2）+ AlphaAgent（KDD'25）：

  三角色闭环（每轮一个市场假设）：
    1. IdeaAgent    : 提出市场假设（自然语言；无 LLM 时用预置模板）
    2. FactorAgent  : 假设 → 参数化公式 JSON（限定 formula_dsl 枚举空间）
                      经 AlphaAgent 三重正则：
                      ① 复杂度控制 —— 结构校验（mode 依赖参数、掩码一致性）
                      ② 新颖性     —— 与因子库公式签名/参数距离去重
                      ③ 假设对齐   —— LLM 输出 alignment 理由 + 指标关联规则校验
    3. EvalAgent    : ParamVM 执行 → 五维 + DSR 评估 → 反馈注入下一轮（≤max_rounds）

设计要点：
  - LLM 调用通过 `llm_call(messages) -> str` 注入（默认 None → 规则化降级模式，
    用预置假设 + 模板公式 + 随机扰动探索，不依赖外部服务）
  - 可用 web.ai_providers.chat_completions 作为 llm_call（用户配置 key 后启用）
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

import numpy as np

from model_core.formula_dsl import (
    CHROM_LEN, GENE_SPACES, MODE1_OPS, MODE2_OPS, MASK_RULES,
    ParamFormula, chrom_to_formula, formula_signature, random_chrom,
)
from model_core.param_vm import ParamVM
from model_core.eval.report import build_factor_report

_EPS = 1e-9

# ── 预置市场假设模板（无 LLM 时的 IdeaAgent 降级；也作 LLM 的种子提示）──────

HYPOTHESIS_TEMPLATES: tuple[str, ...] = (
    "成交笔数自相关结构被打破往往意味着主力资金介入，尾盘窗口尤为敏感",
    "单笔成交金额与成交量的回归截距剥离系统性放量后，反映真实交易规模信号",
    "成交笔数与成交金额的欧氏距离异常放大时，微观流动性结构出现极端背离",
    "量价齐升但动量指标钝化时，上涨缺乏持续性，存在均值回归压力",
    "高波动环境下资金偏好低换手标的，波动率因子在不同 regime 下方向反转",
    "MACD 柱与成交额协变上升时趋势确认，背离时趋势衰竭",
    "RSI 极端值与量价背离组合时反转概率更高",
)

# 指标→语义关键词（假设对齐的规则校验用）
_INDICATOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "num_trades": ("成交笔数", "笔数", "交易频率"),
    "amt_per_trade": ("单笔", "每笔", "单笔成交"),
    "amt_vol_euclid": ("欧氏距离", "背离", "结构断裂"),
    "volume": ("成交量", "放量", "缩量"),
    "amount": ("成交额", "金额"),
    "close": ("价格", "收盘", "股价"),
    "ret": ("收益", "动量", "涨幅"),
    "ret5": ("5日", "短期动量", "短线"),
    "rsi14": ("RSI", "超买", "超卖"),
    "macd_hist": ("MACD", "柱", "趋势"),
    "vol_regime": ("波动", "波动率", "regime"),
    "hl_range": ("振幅", "高低价", "波动区间"),
}


@dataclass
class AgentResult:
    hypothesis: str
    formula: ParamFormula
    chrom: list[int]
    report: dict | None = None       # EvalAgent 的 FactorReport.as_dict()
    alignment: float = 0.0           # 假设对齐分 [0,1]
    novelty: float = 1.0             # 新颖性 [0,1]
    rounds: int = 1
    rejected: bool = False
    reason: str = ""


# ── JSON 提取工具 ─────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """从 LLM 输出提取 JSON 对象（容忍 ```json 围栏与前后缀文本）。"""
    # 去掉围栏
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    # 找第一个 { 到最后一个 }
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


# ── 复杂度校验（AlphaAgent 正则 ①）───────────────────────────────────────

def validate_formula_params(d: dict) -> list[str]:
    """参数结构校验，返回违规原因列表（空=合法）。"""
    violations: list[str] = []
    mode = int(d.get("mode", 1))
    if mode not in (1, 2):
        violations.append(f"mode 必须为 1/2，实际 {mode}")
    if mode == 1:
        if d.get("B"):
            violations.append("mode=1 时 B 应为空")
        if d.get("mode2"):
            violations.append("mode=1 时 mode2 应为空")
    else:
        if not d.get("B"):
            violations.append("mode=2 时必须提供 B")
        if not d.get("mode2"):
            violations.append("mode=2 时必须提供 mode2")
    if d.get("mask_rule") and not d.get("mask_field"):
        violations.append("mask_rule 存在时 mask_field 必填")
    if not d.get("A"):
        violations.append("A（核心指标）必填")
    if not d.get("mode1"):
        violations.append("mode1 算子必填")
    return violations


# ── 新颖性校验（AlphaAgent 正则 ②）───────────────────────────────────────

def formula_novelty(
    formula: ParamFormula,
    library_signatures: list[str],
    library_chroms: list[list[int]],
) -> float:
    """新颖性：1 - 与库内公式的最大相似度。

    相似度 = 0.5·签名相同 + 0.5·参数距离（同位置基因相同的比例）。
    """
    sig = formula_signature(formula)
    if sig in library_signatures:
        return 0.0
    chrom = formula.to_chrom()
    best = 0.0
    for lc in library_chroms:
        same = sum(1 for a, b in zip(chrom, lc) if a == b) / CHROM_LEN
        best = max(best, same)
    return float(1.0 - best)


# ── 假设对齐（AlphaAgent 正则 ③，规则校验）────────────────────────────────

def hypothesis_alignment(hypothesis: str, formula: ParamFormula) -> float:
    """假设与公式的语义对齐分 [0,1]（规则：指标关键词命中率）。"""
    hits = 0
    used = {formula.A, formula.B, formula.mask_field} - {None}
    for ind in used:
        for kw in _INDICATOR_KEYWORDS.get(ind, ()):
            if kw in hypothesis:
                hits += 1
                break
    total = max(len(used), 1)
    return float(hits / total)


# ── 主引擎 ────────────────────────────────────────────────────────────────

class LLMAgentMiner:
    """LLM 多智能体因子挖掘引擎（无 LLM 时规则化降级）。"""

    def __init__(
        self,
        llm_call=None,              # callable(messages: list[dict]) -> str；None=降级模式
        max_rounds: int = 3,
        novelty_threshold: float = 0.75,
        alignment_threshold: float = 0.33,
        seed: int = 42,
        hypotheses: list[str] | None = None,
    ) -> None:
        self.llm_call = llm_call
        self.max_rounds = max_rounds
        self.novelty_threshold = novelty_threshold
        self.alignment_threshold = alignment_threshold
        self.seed = seed
        self.hypotheses = list(hypotheses or HYPOTHESIS_TEMPLATES)
        self.rng = random.Random(seed)
        self._library_sigs: list[str] = []
        self._library_chroms: list[list[int]] = []

    # ── 公共入口 ─────────────────────────────────────────────────────────

    def mine(
        self,
        indicators: dict[str, np.ndarray],
        ret: np.ndarray,
        n_hypotheses: int = 3,
        library_factors: list[np.ndarray] | None = None,
        existing_formulas: list[ParamFormula] | None = None,
        progress_cb=None,
    ) -> list[AgentResult]:
        """运行多智能体挖掘。

        Args:
            indicators: {指标名: np.ndarray[T]}
            ret: 未来收益 [T]
            n_hypotheses: 挖掘的假设数
            library_factors: 因子库因子序列（多样性/评估用）
            existing_formulas: 已入库公式（新颖性去重用）
            progress_cb: optional callable(round, result)
        """
        vm = ParamVM(indicators)
        self._library_sigs = [formula_signature(f) for f in (existing_formulas or [])]
        self._library_chroms = [f.to_chrom() for f in (existing_formulas or [])]

        results: list[AgentResult] = []
        hyps = self._sample_hypotheses(n_hypotheses)

        for hyp in hyps:
            best: AgentResult | None = None
            feedback = ""
            for rnd in range(1, self.max_rounds + 1):
                formula, align, rejected, reason = self._factor_agent(hyp, feedback)
                if formula is None:
                    break
                # 新颖性正则
                novelty = formula_novelty(formula, self._library_sigs, self._library_chroms)
                if novelty < self.novelty_threshold:
                    # 与库内重复 → 微扰后重试一次（同轮）
                    formula = self._perturb(formula)
                    novelty = formula_novelty(formula, self._library_sigs, self._library_chroms)
                # 执行 + 评估（异常 → 尝试一次规则化重试，不直接丢弃假设）
                try:
                    factor = vm.execute(formula)
                except Exception:  # noqa: BLE001
                    formula2 = self._rule_formula(hyp)
                    if formula2 is not None:
                        try:
                            factor = vm.execute(formula2)
                            formula = formula2
                        except Exception:  # noqa: BLE001
                            break
                    else:
                        break
                report = build_factor_report(
                    factor, ret, formula.to_chrom(), formula.describe(),
                    n_trials=max(len(hyps) * self.max_rounds, 10),
                    library_factors=library_factors,
                )
                feedback = self._feedback_message(report)
                res = AgentResult(
                    hypothesis=hyp, formula=formula, chrom=formula.to_chrom(),
                    report=report.as_dict(), alignment=align, novelty=novelty,
                    rounds=rnd, rejected=rejected, reason=reason,
                )
                if progress_cb:
                    progress_cb(rnd, res)
                if best is None or report.dsr > (best.report or {}).get("dsr", 0.0):
                    best = res
                # 达标即停（DSR 显著且五维综合>0.5）
                if report.dsr > 0.95 and report.five_dim.total > 0.5:
                    break
            if best is not None:
                results.append(best)
                # 入库去重记忆
                self._library_sigs.append(formula_signature(best.formula))
                self._library_chroms.append(best.formula.to_chrom())
        results.sort(key=lambda r: (r.report or {}).get("dsr", 0.0), reverse=True)
        return results

    # ── IdeaAgent：假设 ──────────────────────────────────────────────────

    def _sample_hypotheses(self, n: int) -> list[str]:
        if self.llm_call is None or n <= len(self.hypotheses):
            return self.rng.sample(self.hypotheses, min(n, len(self.hypotheses)))
        # LLM 生成新假设
        prompt = (
            "你是量化研究员。基于以下可用指标，提出一个简洁的 A股市场假设"
            "（一句话，说明因果逻辑）：\n"
            f"指标: {list(_INDICATOR_KEYWORDS.keys())}\n"
            "输出 JSON: {\"hypothesis\": \"...\"}"
        )
        try:
            out = self.llm_call([
                {"role": "system", "content": "你只输出 JSON。"},
                {"role": "user", "content": prompt},
            ])
            d = _extract_json(out)
            if d and d.get("hypothesis"):
                return [str(d["hypothesis"])] + self.rng.sample(
                    self.hypotheses, min(n - 1, len(self.hypotheses)))
        except Exception:  # noqa: BLE001
            pass
        return self.rng.sample(self.hypotheses, min(n, len(self.hypotheses)))

    # ── FactorAgent：假设 → 公式（含三重正则）────────────────────────────

    def _factor_agent(self, hypothesis: str, feedback: str) -> tuple[ParamFormula | None, float, bool, str]:
        if self.llm_call is not None:
            formula = self._llm_formula(hypothesis, feedback)
            if formula is not None:
                violations = validate_formula_params({
                    "A": formula.A, "B": formula.B, "mode": formula.mode,
                    "mode2": formula.mode2, "mode1": formula.mode1,
                    "mask_rule": formula.mask_rule, "mask_field": formula.mask_field,
                })
                align = hypothesis_alignment(hypothesis, formula)
                if violations:
                    return None, 0.0, True, "; ".join(violations)
                if align < self.alignment_threshold:
                    # 对齐不足 → 规则化降级重试
                    formula = self._rule_formula(hypothesis)
                    align = hypothesis_alignment(hypothesis, formula)
                return formula, align, False, ""
        # 降级模式：规则化公式（指标从假设关键词匹配）
        formula = self._rule_formula(hypothesis)
        align = hypothesis_alignment(hypothesis, formula)
        return formula, align, False, "rule-based fallback"

    def _llm_formula(self, hypothesis: str, feedback: str) -> ParamFormula | None:
        prompt = (
            "你是量化因子工程师。把下面的市场假设实现为参数化公式。\n"
            f"假设: {hypothesis}\n"
            f"反馈（上一轮评估，可为空）: {feedback or '无'}\n\n"
            "公式规范（严格 JSON，键如下）:\n"
            '{"A": 指标名, "B": 指标名或null, "window": 5|10|20|60|120|"All", '
            '"slice": 0.0~1.0或null, "mask_field": 指标名或null, '
            '"mask_rule": null|"high_0.3"|"high_0.5"|"high_0.7"|"low_0.3"|"low_0.5"|"low_0.7", '
            '"mode": 1|2, "mode1": 单变量算子, "mode2": 双变量算子或null, '
            '"B_shift_lag": -5~5}\n'
            f"可选指标: {list(_INDICATOR_KEYWORDS.keys())}\n"
            f"单变量算子: {list(MODE1_OPS)}\n双变量算子: {list(MODE2_OPS)}\n"
            "注意: mode=1 时 B/mode2 必须为 null；mask_rule 存在时 mask_field 必填。"
            "只输出 JSON。"
        )
        try:
            out = self.llm_call([
                {"role": "system", "content": "你只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ])
            d = _extract_json(out)
            if not d:
                return None
            return ParamFormula(
                A=str(d["A"]), B=d.get("B"), window=d.get("window", 20),
                slice=d.get("slice"), mask_field=d.get("mask_field"),
                mask_rule=d.get("mask_rule"), mode=int(d.get("mode", 1)),
                mode1=str(d.get("mode1", "Mean")), mode2=d.get("mode2"),
                B_shift_lag=int(d.get("B_shift_lag", 0)),
            )
        except Exception:  # noqa: BLE001
            return None

    def _rule_formula(self, hypothesis: str) -> ParamFormula:
        """规则化降级：从假设关键词匹配指标，随机算子组合。"""
        # 找假设中提到且库中存在的指标
        matched = [
            ind for ind, kws in _INDICATOR_KEYWORDS.items()
            if any(kw in hypothesis for kw in kws)
        ]
        a = matched[0] if matched else self.rng.choice(list(_INDICATOR_KEYWORDS.keys()))
        mode = 2 if self.rng.random() < 0.4 else 1
        if mode == 2:
            pool = [i for i in _INDICATOR_KEYWORDS if i != a]
            b = self.rng.choice(pool) if pool else a
            return ParamFormula(
                A=a, B=b, window=self.rng.choice([10, 20, 60, 120, "All"]),
                slice=self.rng.choice([None, 0.3, 0.5, 0.8]),
                mask_field=self.rng.choice([None, "volume", "close"]),
                mask_rule=None, mode=2,
                mode1=self.rng.choice(list(MODE1_OPS)),
                mode2=self.rng.choice(list(MODE2_OPS)),
                B_shift_lag=self.rng.choice([-2, -1, 0, 1, 2]),
            )
        return ParamFormula(
            A=a, B=None, window=self.rng.choice([10, 20, 60, 120, "All"]),
            slice=self.rng.choice([None, 0.3, 0.5, 0.8]),
            mask_field=self.rng.choice([None, "volume", "close"]),
            mask_rule=None, mode=1,
            mode1=self.rng.choice(list(MODE1_OPS)),
            mode2=None, B_shift_lag=0,
        )

    def _perturb(self, formula: ParamFormula) -> ParamFormula:
        """微扰（与库内重复时）：只扰动"安全基因"（window/slice/mask_rule/算子/移位），
        不动指标基因（A/B/mask_field），避免引入指标库中不存在的指标。"""
        chrom = formula.to_chrom()
        safe_genes = [2, 3, 5, 6, 7, 8, 9]  # window/slice/mask_rule/mode/mode1/mode2/lag
        i = self.rng.choice(safe_genes)
        chrom[i] = self.rng.randrange(len(GENE_SPACES[i]))
        return chrom_to_formula(chrom)

    # ── EvalAgent 反馈 ───────────────────────────────────────────────────

    @staticmethod
    def _feedback_message(report) -> str:
        d = report.five_dim
        return (
            f"上一轮: IC={report.ic:+.3f} DSR={report.dsr:.2f} 五维综合={d.total:.2f} "
            f"(预测力{d.pps:.2f}/稳定{d.stability:.2f})。"
            "请针对弱点改进公式（保持假设不变）。"
        )
