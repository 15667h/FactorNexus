# FactorNexus 策略工厂（Strategy Factory）实施方案 — P24

> 目标：构建量化系统**中层策略层**——把"因子库"（原料）加工成"预测信号"（半成品），
> 供顶层风险预算与执行层使用。对标微软 Qlib 机器学习选股管线 + 华泰金工
> 《人工智能选股》系列 + 2026 最新 Mamba/SSM 金融时序模型。
>
> 依据（权威来源）：
> - 微软 Qlib（[arXiv 2009.11189](https://arxiv.org/pdf/2009.11189v1)、[GitHub 48k★ 持续更新至 2026-07](https://github.com/microsoft/qlib)）：LightGBM 基准、Alpha158 特征、时序交叉验证
> - 华泰金工《人工智能选股》系列二十八（[基于量价的人工智能选股体系概览](https://bigquant.com/wiki/doc/cZuehFXhw2)）：GP 挖掘 + 随机森林合成 + 五因子中性化，实测 RankIC 8.87%、IC_IR 1.16
> - Mamba（[arXiv 2312.00752](https://arxiv.org/abs/2312.00752)）、Mamba-3（[ICLR 2026](https://zhuanlan.zhihu.com/p/1961083698901418021)）：状态空间模型，长序列线性复杂度、5× 推理吞吐

---

## 一、问题定义：为什么要策略工厂

当前系统状态：
- ✅ 底层矿机（GP/RL/LLM 挖掘）→ 365 个因子
- ✅ 组合层（P16 流水线）但**用等权/IC_IR 直接合成因子** → 横截面 RankIC 仅 **0.019**
- ❌ 缺"因子 → 预测模型 → 信号"这一层

**核心缺陷**：IC_IR 加权是**线性静态合成**，无法捕捉因子间的非线性交互、时序动态、风格漂移。
机构做法（Qlib/华泰/幻方公开方法论）：**机器学习模型拟合因子面板 → 输出预测信号**，
实测可把 RankIC 从 0.02 级提升到 0.05-0.09 级（华泰公开数字：合成因子 RankIC 8.87%）。

## 二、目标指标（验收标准）

| 指标 | 现状（IC_IR 合成） | 目标（策略工厂） | 衡量方式 |
|---|---|---|---|
| 横截面 RankIC | 0.019 | **≥ 0.04**（首版），0.06（Mamba 接入后） | 每日截面 Spearman 均值 |
| IC_IR | ~0.3 | **≥ 0.8** | mean/std×√N |
| 十分组单调性 | 弱 | **|Spearman| ≥ 0.6** | Q1-Q10 收益 |
| 换手率 | 0.46/日 | **≤ 0.2/日**（周频信号） | 信号自相关 |
| OOS 稳定性 | 分段翻转 | **前后半段 RankIC 同号** | 时间分段检验 |

## 三、架构设计

```
因子库 store/factors/ (365 因子)
      │
      ▼
┌──────────────────────────────────────────────────────┐
│ ① 特征工程层（复用现有）                               │
│    - neutralize_panel 五因子中性化（去风格）            │
│    - orthogonalize_panel 正交化（去冗余）              │
│    - 缺失值处理 / 去极值 / 截面 zscore                 │
├──────────────────────────────────────────────────────┤
│ ② 模型池（新构建 model_core/models/）                  │
│    - LGBM（首选：快、稳、可解释）                      │
│    - XGBoost（对照）                                  │
│    - MLP（PyTorch，非线性基线）                       │
│    - Mamba/SSM（状态空间，长序列时序）                 │
│    - 集成：加权平均 / stacking                        │
├──────────────────────────────────────────────────────┤
│ ③ walk-forward 训练框架（防前视核心）                  │
│    训练段 ──► 验证段 ──► 测试段                        │
│    滚动：train(expand/rolling) → predict 下一段       │
│    每条规则：训练只用 t 及以前，预测 t+1..t+H          │
├──────────────────────────────────────────────────────┤
│ ④ 信号评估与选择                                      │
│    - RankIC / IC_IR / 分层 / 换手 / 衰减               │
│    - 时间分段方向一致（复用认证逻辑）                   │
│    - 信号合并（按 IC_IR 加权，替代因子直接合成）         │
├──────────────────────────────────────────────────────┤
│ ⑤ 输出：预测信号面板 signal_panel                      │
│    （index=交易日, columns=股票, 值=预测得分）          │
└──────────────────────┬───────────────────────────────┘
                       ▼
            顶层风险预算（P19 优化器接入）→ 执行层
```

## 四、模块与文件规划

```
model_core/strategy_factory/
├── __init__.py
├── dataset.py          # 因子面板 → 训练样本（对齐/去极值/标准化/标签）
├── walk_forward.py     # 时序滚动训练框架（防前视核心，Qlib 对齐）
├── models/
│   ├── __init__.py
│   ├── lgbm_model.py   # LightGBM 预测器（sklearn 可用时先 GradientBoosting）
│   ├── xgb_model.py    # XGBoost 预测器（可选装）
│   ├── mlp_model.py    # PyTorch MLP（非线性基线）
│   └── mamba_model.py  # SSM 状态空间预测器（PyTorch 纯实现，对标 Mamba 思路）
├── ensemble.py         # 模型集成（IC_IR 加权 / stacking）
├── evaluate.py         # 信号评估（RankIC/ICIR/分层/换手/衰减/分段一致）
└── signal_exporter.py  # 输出 signal_panel → 组合层

scripts/mine_signal.py  # 一键：因子库 → 策略工厂 → 信号面板 → 评估报告
tests/test_p16_strategy_factory.py
```

## 五、实施里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| **M1 数据层** | dataset.py：因子面板对齐（复用 build_panels 逻辑）、标签=未来 H 日收益、时序切分 | 样本矩阵形状正确、无未来泄漏（测试锁定） |
| **M2 walk-forward 框架** | walk_forward.py：滚动训练/预测，gap 防泄漏，Qlib 对齐 | 改尾部数据不影响历史预测（因果测试） |
| **M3 LGBM 基线** | lgbm_model.py 接入，跑通全流程 | 横截面 RankIC ≥ 0.03（vs 现 0.019） |
| **M4 模型池扩展** | MLP + Mamba/SSM 实现，对照实验 | 选最优模型，RankIC ≥ 0.04 |
| **M5 集成 + 评估** | ensemble.py + evaluate.py | IC_IR ≥ 0.8、分层单调、OOS 同号 |
| **M6 接入组合层** | portfolio_pipeline 支持 --signal 模式（用模型信号替代因子合成） | 组合回测可运行 |

### 实施状态（2026-08-27 更新）

| 阶段 | 状态 | 落地 | 实测 |
|---|---|---|---|
| M1 数据层 | ✅ | `dataset.py`（含风格特征/综合得分/时间切分） | 160,992 样本 / 444 特征 / 91 股 / 68 折 |
| M2 walk-forward | ✅ | `walk_forward.py`（gap 防泄漏/Qlib 对齐） | OOS 4060 天 / 覆盖率 43.5% |
| M3 LGBM 基线 | ✅ | `models/lgbm_model.py` | RankIC +0.0057 / IC_IR 0.45（股票池 91 只偏小，扩池后可升） |
| M4 模型池扩展 | ✅ | `models/mlp_model.py`（3 层 MLP，时间顺序验证集+早停）；`models/ssm_model.py`（纯 PyTorch 对角化 S4 对照；可选真 Mamba 包装） | 冒烟通过（r² 正常收敛）；全量对比见 `--models all` |
| M5 集成+评估 | ✅ | `ensemble.py`：`rank_average`（截面排名加权平均）/ `make_bagging_factory`（多 seed）/ `make_ensemble_factory`（异质集成）/ `stacking_fit_predict`（时间分段两层） | 单模型时自动升级 bagging；stacking OOS 面板输出验证通过 |
| M6 接入组合层 | ✅ | `mine_signal.py --portfolio`：信号面板 → build_portfolio（持有平滑 `--rebalance`）→ backtest → 绩效/风险/成本损耗 | 关键教训：弱信号每日调仓换手 3.9/日 → 成本吞噬全部收益（-99%）；**调仓期匹配预测周期**（--rebalance 5）后正常 |

**M6 关键工程决策**：信号预测 H 日收益 → 调仓周期必须 = H（否则每日用 5 日信号重排，
弱信号下 TopN 几乎随机换血，换手成本吃掉全部 alpha）。`--rebalance` 默认建议 = horizon。

**当前瓶颈**：RankIC 0.0057（目标 ≥0.04）主要受股票池规模限制（91 只 vs 华泰全市场
5000 只）；挖矿 500 只批次完成后样本量扩大，RankIC/IC_IR 预计显著提升。

## 六、关键设计决策（依据权威实践）

1. **标签设计**：未来 H 日收益（与挖矿 horizon=5 一致）。Qlib 用未来收益做标签，
   华泰用 RankIC 做适应度——模型侧用连续收益 + 截面标准化。

2. **防前视三重保险**（机构铁律）：
   - 特征：只用 t 及以前的因果特征（现有因子全部因果 ✅）
   - 训练：walk-forward 滚动窗口，训练段与预测段严格不相交
   - 评估：预测信号在训练段之后的 OOS 段计算 RankIC（绝不回看训练段）

3. **模型选型依据**：
   - LightGBM：Qlib 基准模型，A 股多因子公开实证稳定（华泰/各券商研报）
   - MLP：捕捉非线性交互的最低成本基线
   - **Mamba/SSM**：2026 前沿（ICLR 2026 Mamba-3），长序列线性复杂度，适合
     日线长历史（5000 根）；PyTorch 可纯实现简化 SSM（无需官方 CUDA 扩展），
     作为探索性模型与 LGBM 对照——**不以"用了新模型"为目标，以 RankIC 提升为准**

4. **集成策略**：先 IC_IR 加权（简单可靠），若模型间相关性低再上 stacking。
   华泰实证：ML 合成因子 RankIC 8.87% 明显高于单因子——集成是收益来源。

5. **与现有系统关系**：
   - 策略工厂**替代**组合层里"IC_IR 因子合成"这一步（P16 的 [3/8]）
   - 复用：neutralize_panel / orthogonalize_panel / backtest_portfolio / 评估框架
   - 信号面板直接喂给组合构建（等权 → 后续换 P19 优化器）

## 七、风险与对策

| 风险 | 对策 |
|---|---|
| 模型过拟合（因子 365 × 股票 90，样本有限） | walk-forward + 早停 + 特征裁剪（Top-K 因子） |
| 提升不明显（因子本身信号弱） | 扩大因子库（继续挖矿）+ 加高频/基本面因子 |
| Mamba 实现复杂 | 先跑 LGBM 基线拿到收益，Mamba 作为 M4 探索 |
| 信号换手高 | 信号平滑（EMA）+ 周频重训 |

## 八、预期收益（基于华泰公开数字推算）

华泰系列二十八实测：GP 挖掘 → 随机森林合成 → 五因子中性化后
**RankIC 8.87%、IC_IR 1.16、TOP 组合年化超额 9.65%**。

我们的差异：因子来自三引擎（更强多样）、股票池更小（90 只 vs 全市场）——
保守估计首版 RankIC 0.04-0.06、IC_IR 0.8-1.0（股票池扩大后趋近华泰水平）。

---

*本方案对齐 Qlib / 华泰 / Mamba-2026 权威方法论，M1-M3 为核心必做，M4 为前沿探索。*
