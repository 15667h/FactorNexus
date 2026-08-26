"""
model_core/engines/gp_engine.py — NSGA-III 多目标遗传规划因子挖掘引擎（P1 核心）

对齐华泰《高频特征参数化》（2026.4）：
  - 五目标评价：|IC|、IC胜率、多头绝对收益、多头夏普、多头胜率
  - NSGA-III 解决高维目标空间"维数灾难"（pymoo 0.6.2 官方实现）
  - 动态短板惩罚：某目标百分位 < 短板阈值 → 前沿面层级下修（清除畸形因子）
  - 个体 = 参数化万能公式染色体（model_core/formula_dsl）

用法：
    engine = NSGA3FactorMiner(n_gen=50, pop_size=200, seed=42)
    results = engine.mine(factor_matrix, ret_matrix, progress_cb=...)
    # results: list[CandidateFactor]
"""
from __future__ import annotations

import numpy as np
import torch
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from model_core.formula_dsl import (
    CHROM_LEN, GENE_SPACES, chrom_to_formula, normalize_chrom, random_chrom,
)

# ── 五目标（华泰口径，向量化）──────────────────────────────────────────────

def evaluate_five_objectives(factor: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """五目标评分（对单条因子序列）。

    factor: [T] 因子值；ret: [T] 未来收益（已对齐 t+1）
    目标:
      f1 = |IC|           截面不可用（单标的），用时序 IC 绝对值（对齐现有引擎）
      f2 = IC胜率          IC_t > 0 的窗口占比
      f3 = 多头绝对收益    top10% 分位组合 5 日平均收益
      f4 = 多头夏普        多头组合收益/波动（年化 244）
      f5 = 多头胜率        多头组合日收益 > 0 占比
    """
    n = min(len(factor), len(ret))
    if n < 20:
        return np.zeros(5)
    f = factor[:n].astype(np.float64)
    r = ret[:n].astype(np.float64)

    # 时序 IC（滚动 20 窗口的逐窗口相关）
    ic_list = []
    for t in range(20, n):
        x, y = f[t - 20:t], r[t - 20:t]
        xm, ym = x - x.mean(), y - y.mean()
        sx, sy = (xm ** 2).mean() ** 0.5, (ym ** 2).mean() ** 0.5
        if sx < 1e-9 or sy < 1e-9:
            continue
        ic_list.append((xm * ym).mean() / (sx * sy))
    if not ic_list:
        return np.zeros(5)
    ic_arr = np.array(ic_list)
    ic_mean = ic_arr.mean()

    # 多头组合：因子 top 10% 时段的未来收益
    thresh = np.quantile(f, 0.9)
    long_mask = f >= thresh
    if long_mask.sum() < 5:
        return np.array([abs(ic_mean), (ic_arr > 0).mean(), 0.0, 0.0, 0.0])
    long_ret = r[long_mask]
    sharpe = (long_ret.mean() / (long_ret.std() + 1e-9)) * np.sqrt(244) if long_ret.std() > 1e-9 else 0.0
    return np.array([
        abs(ic_mean),
        (ic_arr > 0).mean(),
        long_ret.mean(),
        float(np.clip(sharpe, -10, 10)),
        (long_ret > 0).mean(),
    ])


def shortboard_penalize(objectives: np.ndarray, weak_quantile: float = 0.10,
                        penalty_per_weak: float = 0.5) -> np.ndarray:
    """动态短板惩罚：目标百分位 < weak_quantile 的短板，每个下修 penalty 层。

    返回修正后的目标（对应前沿面层级下修，pymoo 中通过把短板目标值劣化实现）。
    """
    out = objectives.copy()
    for j in range(objectives.shape[1]):
        col = objectives[:, j]
        pct = (col < np.quantile(col, weak_quantile)).mean()
        if pct > 0.5:  # 该目标在当代是"短板"（多数个体低于阈值分位）
            # 把该目标劣化：所有个体该目标乘以 (1 - penalty_per_weak * 0.1)
            # 注意：方向是"越小越好"的问题中惩罚为加大；这里五目标均为越大越好，
            # 短板惩罚 = 将该目标压缩（劣化），越差的个体压缩越多
            rank = np.argsort(np.argsort(col)) / max(len(col) - 1, 1)
            out[:, j] = col * (1.0 - penalty_per_weak * 0.1 * (1.0 - rank))
    return out


# ── pymoo Problem 封装 ─────────────────────────────────────────────────────

class FormulaProblem(Problem):
    """NSGA-III 优化问题：染色体 [0..len(space)-1]^10 → 五目标（越大越好）。

    真实执行：ParamVM 按 10 参数公式从指标库计算因子 → 五目标评分。
    """

    def __init__(self, indicators: dict[str, np.ndarray], ret: np.ndarray,
                 weak_quantile: float = 0.10, vm=None) -> None:
        self.indicators = {k: np.asarray(v, dtype=np.float64) for k, v in indicators.items()}
        self.ret = np.asarray(ret, dtype=np.float64)
        self.weak_quantile = weak_quantile
        self.eval_count = 0
        from model_core.param_vm import ParamVM
        self.vm = vm or ParamVM(self.indicators)
        super().__init__(
            n_var=CHROM_LEN,
            n_obj=5,
            xl=np.array([0] * CHROM_LEN, dtype=int),
            xu=np.array([len(s) - 1 for s in GENE_SPACES], dtype=int),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        objs = np.zeros((n_pop, 5))
        for i in range(n_pop):
            chrom = normalize_chrom(X[i].tolist())
            formula = chrom_to_formula(chrom)
            try:
                factor = self.vm.execute(formula)  # [T]，已后处理
                objs[i] = evaluate_five_objectives(factor, self.ret)
            except (KeyError, ValueError):
                objs[i] = np.zeros(5)  # 指标缺失/执行失败 → 全零目标（被 Pareto 淘汰）
        # 动态短板惩罚（华泰）
        objs = shortboard_penalize(objs, self.weak_quantile)
        out["F"] = -objs  # pymoo 默认最小化，取负转最大化
        self.eval_count += n_pop


# ── 引擎主类 ───────────────────────────────────────────────────────────────

class NSGA3FactorMiner:
    """NSGA-III 多目标因子挖掘引擎。"""

    def __init__(self, pop_size: int = 200, n_gen: int = 50, seed: int = 42,
                 weak_quantile: float = 0.10) -> None:
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed
        self.weak_quantile = weak_quantile
        self.history: list[dict] = []

    def mine(self, indicators: dict[str, np.ndarray], ret: np.ndarray,
             progress_cb=None, init_chroms: list[list[int]] | None = None
             ) -> list[dict]:
        """运行挖掘。

        Args:
            indicators: {指标名: np.ndarray[T]}（model_core.indicator_builder 产出）
            ret:        np.ndarray[T] 未来收益（已对齐 t+1）
            progress_cb: optional callable(gen, best_F) 每代回调
            init_chroms: optional 初始种群种子（P6 联动：LLM/矿池精英注入）。
                         每条为 [CHROM_LEN] 整数染色体；数量 < pop_size 时其余随机补齐；
                         数量 > pop_size 时截断。None 时保持纯随机初始化（原行为）。
        Returns:
            非支配前沿的解列表 [{chrom, formula, describe, objectives}]
        """
        problem = FormulaProblem(indicators, ret, self.weak_quantile)

        # das-dennis 参考点数量随种群自适应（必须 ≤ pop_size）：
        #   n_partitions=3 → 35 点；4 → 70；5 → 126；6 → 210
        if self.pop_size >= 210:
            n_part = 6
        elif self.pop_size >= 126:
            n_part = 5
        elif self.pop_size >= 70:
            n_part = 4
        else:
            n_part = 3
        ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=n_part)
        # P6 联动：初始种群 = 种子染色体 + 随机补齐（种子优先占据前半）
        sampling = IntegerRandomSampling()
        if init_chroms:
            seed_rows: list[np.ndarray] = []
            rng = np.random.default_rng(self.seed)
            for c in init_chroms:
                if len(seed_rows) >= self.pop_size:
                    break
                norm = normalize_chrom([int(v) for v in c])
                # 基因越界防护（外部种子可能手改/越界）：裁剪到枚举空间内
                norm = [min(max(v, 0), len(GENE_SPACES[i]) - 1)
                        for i, v in enumerate(norm)]
                seed_rows.append(np.asarray(norm, dtype=int))
            if len(seed_rows) < self.pop_size:
                bounds = np.asarray([len(s) - 1 for s in GENE_SPACES], dtype=int)
                rand = rng.integers(0, bounds + 1, size=(self.pop_size - len(seed_rows), CHROM_LEN))
                seed_rows.extend(rand)
            sampling = np.asarray(seed_rows[:self.pop_size], dtype=int)
        algorithm = NSGA3(
            pop_size=self.pop_size,
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=SBX(prob=0.8, eta=15, vtype=int),
            mutation=PM(prob=0.1, eta=20, vtype=int),
            eliminate_duplicates=True,
            seed=self.seed,
        )

        res = minimize(
            problem, algorithm, ("n_gen", self.n_gen), seed=self.seed,
            save_history=False, verbose=False,
        )

        # 收集结果
        results = []
        F = np.asarray(res.F)  # [n_pop, 5]，pymoo 最小化方向（负值）
        for i in range(F.shape[0]):
            chrom = normalize_chrom(res.X[i].tolist())
            formula = chrom_to_formula(chrom)
            results.append({
                "chrom": chrom,
                "formula": formula,
                "describe": formula.describe(),
                "objectives": (-F[i]).tolist(),  # 转回"越大越好"
            })
        # 按 |IC| 排序（f1）
        results.sort(key=lambda r: r["objectives"][0], reverse=True)
        self.history.append({"n_eval": problem.eval_count, "n_solutions": len(results)})
        return results
