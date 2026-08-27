# FactorNexus — 全市场量化因子联动挖掘中心

> **FactorNexus · Full-Market Quant Factor Mining Hub**
>
> 基于 **遗传规划（GP）+ 强化学习（RL）+ 大语言模型（LLM）三引擎联动**的全市场量化因子工厂：
> 把全 A 股 5000+ 只股票当作一条矿脉，三引擎每次运行一起挖、互相喂知识，
> CUDA GPU 加速，断点续跑，机构级认证入库，组合层全链路闭环。

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

---

## 目录

1. [系统全景](#一系统全景)
2. [快速开始](#二快速开始)
3. [组合流水线与因子监控](#二五组合流水线与因子监控p16p17炼油厂点火)
4. [策略工厂（P24：M1-M6 全链路）](#二六策略工厂p24因子--预测信号--组合m1-m6-全链路)
5. [全市场挖矿机命令手册](#三全市场挖矿机命令手册)
6. [因子库浏览与回测命令手册](#四因子库浏览与回测命令手册)
7. [高频因子挖掘命令手册](#五高频因子挖掘命令手册)
8. [数据与库健康审计](#六数据与库健康审计)
9. [组合层 API 手册（P14）](#七组合层-api-手册p14)
10. [数据管道与质量保障](#八数据管道与质量保障)
11. [机构级认证与回测口径](#九机构级认证与回测口径)
12. [输出产物一览](#十输出产物一览)
13. [项目结构](#十一项目结构)
14. [测试](#十二测试)
15. [机构级标准对照](#十三机构级标准对照)
16. [常见问题 FAQ](#十四常见问题-faq)
17. [版本历史](#十五版本历史)

---

## 一、系统全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FactorNexus 架构                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【数据层】三源兜底 · 机构 D3 健康检查                                 │
│   腾讯(qfq复权) → 新浪(不复权) → 通达信(pytdx全历史)                    │
│   └─ KlineStore 增量入库（复权口径冲突检测 + 来源元数据溯源）             │
│   └─ quality.py：重复剔除 / OHLC一致性 / 粘滞检测 / 跳变分类(混库拒挖)   │
│                                                                     │
│  【挖掘层】三引擎联动（每次运行全市场一起挖）                            │
│   ┌─────────┐   种子注入   ┌─────────┐   精英写入    ┌─────────┐    │
│   │ GP      │◄────────────│ LLM     │──────────────►│ 矿 池   │    │
│   │ NSGA-III│   批级假设    │ 多智能体 │   新颖性约束   │MarketPool│   │
│   └─────────┘              └─────────┘               └────┬────┘   │
│        ▲                                                  │        │
│        │ 候选统一裁决（五维+DSR+拥挤度）                      │        │
│   ┌─────────┐   精英预热（跨品种迁移）                      │        │
│   │ RL      │◄────────────────────────────────────────────┘        │
│   │REINFORCE│   GPU · token 公式                                   │
│   └─────────┘                                                      │
│                                                                     │
│  【认证层】机构三段式（宁缺毋滥）                                      │
│   训练段选优 → OOS 段认证（块自助 p≤0.05 + 方向一致 + 五维≥0.45）       │
│   └─ 横截面认证：跨股票池每日 RankIC → 中心化块自助（股票池≥30 只）      │
│   └─ DSR / PBO / CPCV 多重检验控制                                   │
│                                                                     │
│  【组合层】P14（华泰因子工厂 + Qlib Portfolio + Barra 对齐）           │
│   中性化(行业/市值/收益/波动/换手) → 正交化(增量挖掘) → 合成(IC_IR/ML)  │
│   → 组合构建 → 组合回测 → 风险模型 → Brinson/风格归因                  │
│                                                                     │
│  【高频层】P15（分钟级 → 日频特征）                                    │
│   通达信 1h/30m/15m/5m 全历史 → 14 个日内特征 → OOS 认证入库           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**核心亮点**

| 亮点 | 说明 |
|---|---|
| 三引擎联动 | GP/RL/LLM 共享矿池：LLM 批级假设注入 GP 种群、GP 精英写回矿池、RL 精英跨品种预热——不再是三个孤立的挖掘器 |
| 全市场矿脉 | 新浪 hs_a 全 A 股清单（沪深 5000+ 只，过滤北交所），全历史日线自动回填 |
| 机构级认证 | 训练/OOS 严格分离、横截面 RankIC + 中心化块自助、DSR/PBO/CPCV、拥挤度 0.85 去重 |
| 数据质量防线 | 复权口径冲突检测（防止 qfq/不复权混库）、跳变分类（混库标的自动拒挖）、OHLC 一致性、粘滞价格检测 |
| 组合层闭环 | 从单因子到组合：中性化 → 正交化 → 合成 → 组合回测 → Brinson 归因 |
| 高频因子 | 分钟级数据 → 14 个日内特征（跳空/振幅/波动/量价相关/尾部风险等） |
| 策略工厂（P24） | 因子库 → ML 预测信号：walk-forward + 模型池（LGBM/MLP/S4）+ 集成（rank_avg/bagging/stacking），IC_IR 年化 >1.0 |
| 顶层风险预算（P19） | markowitz / risk_parity / black_litterman 面板优化，滚动协方差防前视，换手 0.8→0.05/日 |

---

## 二、快速开始

### 2.1 安装

```bash
# Python 3.12+（实测 3.12.10），建议 NVIDIA GPU（RL 引擎加速，无 GPU 自动回退 CPU）
pip install -r requirements.txt

# 可选：配置 DeepSeek API Key（LLM 批级真挖掘；缺省时 LLM 走规则化降级，流程不受影响）
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY=sk-xxx
```

### 2.2 首次挖掘

```bash
# 先试 5 只（约 1 分钟）
python scripts/mine_full_market.py --limit 5

# 查看因子库
python scripts/factor_backtest.py --list

# 正式全市场（5000+ 只，三引擎全开，GPU，断点续跑）
python scripts/mine_full_market.py --skip-done
```

### 2.3 一句话流程

**挖矿（`mine_full_market.py`）→ 因子入库（FactorStore）→ 浏览/回测（`factor_backtest.py`）→ 组合流水线（`portfolio_pipeline.py`）→ 因子监控（`factor_monitor.py`）**

---

## 二·五、组合流水线与因子监控（P16/P17，炼油厂点火）

```bash
# 组合层一键流水线：因子库 → 股票得分面板 → 中性化 → 组合构建 → 回测 → 归因
python scripts/portfolio_pipeline.py

# 组合规模 Top10（多空各 10 只）
python scripts/portfolio_pipeline.py --n-top 20

# 纯多头组合
python scripts/portfolio_pipeline.py --long-only

# 拉取行业数据做行业中性化（默认仅风格中性化）
python scripts/portfolio_pipeline.py --industry

# 输出 Markdown 报告
python scripts/portfolio_pipeline.py --report store/meta/portfolio_report.md

# 顶层风险预算（P19）：用优化器替代等权/得分权重
python scripts/portfolio_pipeline.py --optimizer markowitz          # 均值-方差
python scripts/portfolio_pipeline.py --optimizer risk_parity        # 风险平价
python scripts/portfolio_pipeline.py --optimizer black_litterman    # BL 观点融合
python scripts/portfolio_pipeline.py --optimizer markowitz --opt-window 120 \
    --rebalance 10 --risk-aversion 3.0   # 协方差窗口/持有期/风险厌恶

# 因子监控：全库 IC 衰减/方向/失效预警
python scripts/factor_monitor.py

# 监控详情（每个因子的认证/实时 RankIC + 衰减轨迹）
python scripts/factor_monitor.py --detail

# 监控参数：实时段 60 根 / 10 日预测周期
python scripts/factor_monitor.py --recent 60 --horizon 10
```

**P16 流水线架构修正**（2026-08-27 真实数据验证）：挖掘层产出「单标的因子」，
组合层需要「横截面股票面板」——流水线按股票聚合：每只股票的多因子先做
**IC_IR 时序加权合成**（滚动窗口防前视）→ 形成股票得分面板 → 横截面中性化
→ 组合构建/回测 → Brinson/风格归因。产物：`store/meta/portfolio_report.md`。

**P17 监控预警规则**：实时段（最近 N 根）RankIC 与入库方向相反（方向翻转）/
|RankIC| 跌破认证段一半（IC 衰减）/ 块自助 p 不显著 → 预警。
监控快照：`store/meta/factor_monitor.json`。

---

## 二·六、策略工厂（P24：因子 → 预测信号 → 组合，M1-M6 全链路）

策略工厂是系统的**中层策略层**：把因子库（原料）加工成 ML 预测信号（半成品），
对标微软 Qlib 机器学习选股管线 + 华泰金工《人工智能选股》系列。

```
因子库 store/factors/ (676+ 因子)
   │
   ▼
[M1 数据层]   dataset.py：因子面板 → 行式样本（特征矩阵 + 未来 H 日收益标签）
   │          全因果特征 + 风格列(ret20/vol20/mcap/turn/score) 解决因子稀疏
   ▼
[M2 walk-forward]  walk_forward.py：滚动训练 → OOS 预测（gap 防标签泄漏）
   ▼
[M3/M4 模型池]    lgbm（基线）/ mlp（PyTorch）/ s4（纯 PyTorch 状态空间对照）
   ▼
[M5 集成]        ensemble.py：rank_avg（截面排名加权）/ bagging（多 seed）/
   │            异质集成 / stacking（时间分段两层）
   ▼
[M6 组合联动]    信号面板 → 组合构建（--optimizer 风险预算）→ 回测/绩效/风险
```

```bash
# M1→M6 一键全链路（默认 LGBM + 集成 + 评估对比）
python scripts/mine_signal.py --portfolio --n-top 20 --optimizer risk_parity --rebalance 5 --report out/signal_report.md

# 模型池全对比（LGBM/MLP/S4/集成，NN 12 epochs；全量约 1-2 小时）
python scripts/mine_signal.py --models lgbm,mlp,s4,ensemble

# 叠加时间分段 stacking（两层）
python scripts/mine_signal.py --stacking

# 接入组合层 + 顶层风险预算优化器
python scripts/mine_signal.py --portfolio --optimizer risk_parity --rebalance 5

# 调参：折长/窗口/树数/NN 轮数
python scripts/mine_signal.py --step 180 --window 360 --trees 300 --nn-epochs 15
```

**当前实测**（139 只股票 / 676 因子 / 16 万样本）：LGBM 横截面 RankIC +0.013~0.028
（受限于股票池规模，全市场铺满后预计进入 Qlib 基准区间 0.03-0.05）、IC_IR 年化 >1.0、
前后半段方向一致、十分组单调。**M6 关键工程决策**：信号预测 H 日收益 → 调仓周期
必须匹配（`--rebalance 5`），弱信号每日重排的换手成本会吞噬全部 alpha。

---

## 三、全市场挖矿机命令手册

### 3.1 基础用法

```bash
# 全市场三引擎联动挖掘（默认全 A 股 5000+ 只）
python scripts/mine_full_market.py

# 只挖前 N 只（试跑/分片）
python scripts/mine_full_market.py --limit 30

# 自定义股票清单（每行一个代码）
python scripts/mine_full_market.py --symbols-file my_pool.txt

# 断点续跑（跳过已挖标的——默认进度存在 store/meta/full_market_progress.json）
python scripts/mine_full_market.py --skip-done

# 终端显示每只股票的详细挖掘过程（GP 非支配解 / LLM 假设 / RL 每步 / 统一裁决明细）
python scripts/mine_full_market.py --verbose
```

### 3.2 引擎控制

```bash
# 只开 GP+LLM（快速模式，不跑 RL）
python scripts/mine_full_market.py --engines gp,llm

# 只开 GP（最快）
python scripts/mine_full_market.py --engines gp

# 只开 RL（GPU 深度挖掘）
python scripts/mine_full_market.py --engines rl

# 三引擎全开（默认）
python scripts/mine_full_market.py --engines gp,rl,llm

# 指定计算设备（默认 auto：有 CUDA 自动用 GPU）
python scripts/mine_full_market.py --device cuda      # 强制 GPU
python scripts/mine_full_market.py --device cpu       # 强制 CPU
```

### 3.3 挖掘规模与窗口

```bash
# 挖掘窗口：2000 根 ≈ 8 年日线（默认）
python scripts/mine_full_market.py --bars 2000

# 全历史（量大管饱）
python scripts/mine_full_market.py --bars 0

# RL 训练窗口（默认取训练段，可单独缩小提速）
python scripts/mine_full_market.py --rl-bars 500

# 收益预测周期（默认 5 日）
python scripts/mine_full_market.py --horizon 10

# 股票级并行线程数（默认 4；8 线程全市场约 2-3 小时）
python scripts/mine_full_market.py --workers 8

# 周期（腾讯支持 1d/1w/1M/60m 等，默认 1d）
python scripts/mine_full_market.py --tf 1d
```

### 3.4 GP 引擎参数（NSGA-III 五目标）

```bash
# 种群大小（默认 64）
python scripts/mine_full_market.py --pop 96

# 进化代数（默认 8）
python scripts/mine_full_market.py --gen 12

# 随机种子（复现）
python scripts/mine_full_market.py --seed 42
```

### 3.5 RL 引擎参数（REINFORCE + token 公式）

```bash
# 训练步数（默认 8）
python scripts/mine_full_market.py --rl-steps 10

# 每步采样公式数（默认 64）
python scripts/mine_full_market.py --rl-batch 96

# walk-forward 折数（默认 3）
python scripts/mine_full_market.py --rl-folds 5
```

### 3.6 LLM 引擎参数（批级联动）

```bash
# 每批假设数（默认 3）
python scripts/mine_full_market.py --llm-hyp 5

# 反馈轮数（默认 1）
python scripts/mine_full_market.py --llm-rounds 2

# 批级联动间隔：每 N 只股票注入一次真 LLM 假设（默认 10）
python scripts/mine_full_market.py --llm-batch 20
```

### 3.7 认证门槛（机构三段式）

```bash
# OOS 段比例（默认 0.25 = 最后 25% 样本外认证）
python scripts/mine_full_market.py --oos-frac 0.3

# OOS 段整体 RankIC 最低门槛（默认 0.02）
python scripts/mine_full_market.py --min-oos-rankic 0.03

# OOS 块自助 p 值上限（默认 0.05）
python scripts/mine_full_market.py --min-oos-p 0.01

# DSR 报告门槛（Bailey 修正后为报告项，默认 0.0 不拦截）
python scripts/mine_full_market.py --dsr-gate 0.5

# 拥挤度去重阈值（默认 0.85，与新因子相关性超过即丢弃）
python scripts/mine_full_market.py --crowd-corr 0.9

# 横截面认证批大小（每 N 只完成后统一跨股票认证，默认 20）
python scripts/mine_full_market.py --cert-batch 30
```

### 3.8 预选过滤与数据控制

```bash
# 预选过滤：|IC| < 0.03 的股票跳过深度挖掘（省时模式）
python scripts/mine_full_market.py --quick-gate 0.03

# 不自动拉取缺失股票的 K 线（只挖本地已有数据）
python scripts/mine_full_market.py --no-backfill

# 强制刷新全 A 股清单缓存
python scripts/mine_full_market.py --refresh-universe

# 自定义存储根目录
python scripts/mine_full_market.py --store-dir /path/to/store
```

### 3.9 典型组合命令

```bash
# 生产模式：全市场、三引擎、GPU、断点续跑、详细日志
python scripts/mine_full_market.py --skip-done --verbose --device auto

# 快速巡检：30 只、GP+LLM、预选过滤、4 线程
python scripts/mine_full_market.py --limit 30 --engines gp,llm --quick-gate 0.03 --workers 4

# 深度模式：小样本、大种群、多代数、多步 RL
python scripts/mine_full_market.py --limit 20 --pop 128 --gen 16 --rl-steps 12 --rl-batch 128

# 严格认证：p<0.01、RankIC≥0.03
python scripts/mine_full_market.py --min-oos-p 0.01 --min-oos-rankic 0.03
```

---

## 四、因子库浏览与回测命令手册

### 4.1 因子库列表

```bash
# 交互模式（列表 → 输入编号 → 查看体质 + 回测 + ASCII 资金曲线）
python scripts/factor_backtest.py

# 只列因子库（编号/品种/hash/引擎/类型/IC/五维/Sharpe/公式）
python scripts/factor_backtest.py --list

# 排序：按回测 Sharpe 降序（默认）
python scripts/factor_backtest.py --list --sort sharpe

# 排序：按横截面认证 RankIC（机构预测力标准）
python scripts/factor_backtest.py --list --sort ic

# 排序：按 DSR 降序（Bailey 修正后真实因子普遍≈0，仅供查看）
python scripts/factor_backtest.py --list --sort dsr

# 只看前 N 个
python scripts/factor_backtest.py --list --top 10
```

### 4.2 因子详细体质

```bash
# 查看第 3 个因子的完整体检报告（预测力/认证/显著性/绩效/IC衰减/分层/五维）
python scripts/factor_backtest.py --factor 3
```

输出包含：IC / rankIC / ICIR、横截面认证（RankIC + 块自助 p + 认证股票数）、DSR / PBO / CPCV、Sharpe / 最大回撤 / 换手、IC 衰减曲线（lag 1-10）、十分组分层（单调性 + 多空收益）、AlphaEval 五维。

### 4.3 因子回测（三种选择器）

```bash
# ① 编号（推荐，对应 --list 的第 N 个）
python scripts/factor_backtest.py --backtest 5

# ② 纯 symbol（该品种唯一因子直接回测；多个时报错并列出编号）
python scripts/factor_backtest.py --backtest sh600519

# ③ symbol + hash 前缀（精确指定）
python scripts/factor_backtest.py --backtest sh600519 2f0134
```

### 4.4 回测参数

```bash
# 收益预测周期（默认 5 日）
python scripts/factor_backtest.py --backtest 5 --horizon 10

# 单边换手成本（默认 0.0003）
python scripts/factor_backtest.py --backtest 5 --cost 0.0005

# 单边滑点（机构标准建议 0.0005；成本 = 佣金 + 冲击）
python scripts/factor_backtest.py --backtest 5 --slippage 0.0005

# 关闭涨跌停不可成交限制（默认开启，机构 B3）
python scripts/factor_backtest.py --backtest 5 --no-limit-filter

# 按入库截面方向翻转（默认按本标的实际 IC 符号翻转，防前视）
python scripts/factor_backtest.py --backtest 5 --use-cert-direction

# 自定义存储根目录
python scripts/factor_backtest.py --list --store-dir /path/to/store
```

### 4.5 批量回测

```bash
# 回测 Sharpe 前 5 个因子（批量体检）
python scripts/factor_backtest.py --top 5 --backtest-all

# 回测 DSR 前 10 个
python scripts/factor_backtest.py --sort dsr --top 10 --backtest-all

# 回测横截面认证 IC 前 10 个（含滑点）
python scripts/factor_backtest.py --sort ic --top 10 --backtest-all --slippage 0.0005
```

### 4.6 回测口径说明

- 因子序列与 K 线尾部对齐 → `tanh(因子 zscore)` 连续仓位 × 未来收益 − 换手成本 − 滑点
- t 收盘信号 → t+1 执行（防前视）；非重叠持有期调仓，每日按 1 日收益 mark-to-market
- 板块感知涨跌停：主板 9.9% / 创业板科创板 19.9%
- 跳变日（|1日收益|>21%，混库/复权瑕疵）收益置 0 且建仓冻结
- 方向只用回测段前一半 IC 符号估计（防全样本前视）

---

## 五、高频因子挖掘命令手册

### 5.1 基础用法

```bash
# 单标的挖掘（默认 sh600519，1h 周期，通达信全历史）
python scripts/mine_high_freq.py

# 指定标的
python scripts/mine_high_freq.py --symbol sh600519

# 标的清单（每行一个）
python scripts/mine_high_freq.py --symbols-file pool.txt

# 指定分钟周期（1h 覆盖约 2 年；30m/15m/5m 数据量递减）
python scripts/mine_high_freq.py --symbol sh600519 --tf 30m
```

### 5.2 参数

```bash
# 分钟 K 线数量上限（默认 6000）
python scripts/mine_high_freq.py --bars 4000

# 收益预测周期（交易日，默认 5）
python scripts/mine_high_freq.py --horizon 5

# OOS 段比例（默认 0.25）
python scripts/mine_high_freq.py --oos-frac 0.25

# OOS RankIC 门槛（华泰高频因子同量级，默认 0.02）
python scripts/mine_high_freq.py --min-oos-rankic 0.02

# OOS p 值门槛（默认 0.05）
python scripts/mine_high_freq.py --min-oos-p 0.05
```

### 5.3 高频特征清单（14 个日内特征）

| 特征 | 含义 |
|---|---|
| `hf_open_gap` | 开盘跳空（open/昨收 - 1） |
| `hf_intra_ampl` | 日内振幅（high-low）/open |
| `hf_intra_vol` | 日内分钟收益波动率 |
| `hf_intra_skew` | 日内分钟收益偏度 |
| `hf_intra_kurt` | 日内分钟收益峰度 |
| `hf_intra_ac1` | 日内分钟收益一阶自相关（微观结构） |
| `hf_tail_mom` | 尾盘动量（最后 20% 分钟收益） |
| `hf_morning_ratio` | 上午成交量占比 |
| `hf_vwap_dev` | VWAP 偏离（close/vwap - 1） |
| `hf_vol_corr` | 分钟收益与成交量相关 |
| `hf_big_trade` | 大分钟单占比（量 > 均值+2σ） |
| `hf_ret5m_min` | 日内 5 分钟最差收益（尾部风险） |
| `hf_ret5m_max` | 日内 5 分钟最好收益 |
| `hf_range_pos` | 日内位置（close-low)/(high-low) |

达标特征以 `kind=highfreq` 入库，`feature` 字段标注特征名，可被因子库浏览/回测/组合层使用。

---

## 六、数据与库健康审计

### 6.1 库健康审计（只读，不重拉）

```bash
# 全库扫描：污染清单 + 混库/单向分类 + 来源统计
python _refetch_kline.py --scan-only
```

输出示例：
```
K线库审计: 402 个文件，62 个异常
按来源: {'unknown': 402}
  [异常] sh600519: mix 261次跳变 源=None 问题=[...]
```

### 6.2 污染 K 线重拉（可选）

```bash
# 审计 + 删除污染文件 + 三源链（腾讯 qfq → 新浪 → 通达信）重拉
python _refetch_kline.py
```

> 提示：重拉需要网络与几分钟时间。你也可以不重拉——挖矿机会自动跳过混库标的（`dirty_data`），全市场重训时也会自动重拉缺失数据。

### 6.3 数据质量检查 API

```python
from data_pipeline.quality import check_series, clean_series, classify_jumps
issues = check_series(df)               # 健康检查 → list[str]（空=健康）
df, jump_dates = clean_series(df)       # 清洗 → (df, 跳变日 set)
kind, info = classify_jumps(df)         # 跳变分类 → ("mix"|"one_way"|"clean", 明细)
```

### 6.4 KlineStore 健康审计 API

```python
from data_pipeline.store.kline_store import KlineStore
ks = KlineStore()
ks.audit_kline("sh600519", "1d")   # 单标的：健康检查 + 跳变分类 + 来源元数据
ks.audit_all()                      # 全库：{"total", "dirty", "polluted", "by_source"}
ks.source_info("sh600519", "1d")    # 数据溯源：来源/复权口径/更新时间
```

---

## 七、组合层 API 手册（P14）

组合层五个模块，全部防前视（滚动窗口/历史训练）。以下是从因子库到组合归因的完整代码示例：

### 7.1 因子面板构建（从 FactorStore）

```python
import numpy as np, pandas as pd
from data_pipeline.store.kline_store import FactorStore, KlineStore

store, kstore = FactorStore("store"), KlineStore("store")
factors = store.list_factors()
panels, ret_panel, klines = [], None, {}
for f in factors:
    sym = f["symbol"]
    fdf = store.load(sym, f["hash"])
    kdf = kstore.load(sym, "1d")
    if fdf is None or kdf.empty:
        continue
    factor = fdf["factor"].values.astype(float)
    close = kdf["close"].values.astype(float)
    n = min(len(factor), len(close))
    ts = pd.to_datetime(kdf["ts"].values.astype("int64")[-n:], unit="s")
    ret = np.zeros(n); ret[:-5] = close[5:] / close[:-5] - 1.0
    panels.append(pd.DataFrame({sym: factor[-n:]}, index=ts))
    rp = pd.DataFrame({sym: ret}, index=ts)
    ret_panel = ret_panel.add(rp, fill_value=0.0) if ret_panel is not None else rp
    klines[sym] = kdf
```

### 7.2 五因子中性化

```python
from model_core.portfolio.neutralization import fetch_industry_map, neutralize_panel

# 行业数据（东财分类，缓存 store/meta/industry_map.json；失败自动降级风格中性化）
industry_map = fetch_industry_map("store", refresh=False)

# 逐日横截面 OLS 残差 + zscore：因子 ~ 行业 + log市值 + ret20 + vol20 + turn20
neutral_panels = []
for p in panels:
    neu, report = neutralize_panel(p, klines, industry_map=industry_map or None)
    neutral_panels.append(neu)
    print(report)   # {n_days, n_stocks, industries, degraded, r2_mean}
```

### 7.3 因子正交化（增量信息挖掘）

```python
from model_core.portfolio.orthogonalization import (
    orthogonalize_panel, orthogonalize_series, incremental_rankic,
)

# 横截面正交化：新因子 ~ [1, 已入库因子...] OLS 残差
inc_panel = orthogonalize_panel(panels[0], panels[1:])

# 单标的时序正交化
resid = orthogonalize_series(factor_array, [bench_factor_array])

# 增量信息评估：正交化前后 RankIC 对比
info = incremental_rankic(panels[0], panels[1:], ret_panel)
# {"raw_rankic", "orth_rankic", "incremental"}
```

### 7.4 多因子合成

```python
from model_core.portfolio.combination import combine_icir, combine_ml, combine_equal

# IC_IR 加权合成（权重 ∝ 滚动窗口 IC_IR，防前视；负 ICIR 自动反向）
composite, weights = combine_icir(panels, ret_panel, window=60)
print("权重:", weights)

# 随机森林 ML 合成（滚动训练：每 20 天用前 120 天训练，预测后续，防前视）
composite_ml, report = combine_ml(panels, ret_panel, window=120, n_estimators=100)

# 等权合成（基准对照）
composite_eq = combine_equal(panels)
```

### 7.5 组合构建 + 组合回测

```python
from model_core.portfolio.portfolio import (
    build_portfolio, backtest_portfolio, performance, risk_model,
)

# 组合构建：横截面排序选股（多空 / 纯多；等权 / 因子加权）
weights = build_portfolio(composite, n_top=30, weights="equal", long_short=True)
# weights: 每日权重矩阵（多空 Σ|w|=2，纯多 Σw=1）

# 组合回测（t+1 执行、换手成本、板块涨跌停、跳变防御）
bt = backtest_portfolio(weights, ret_panel, cost=0.0003)
# {nav, daily_ret, total_ret, annual_ret, annual_vol, sharpe, sortino,
#  max_dd, calmar, turnover, n}

# 绩效（含基准超额与信息比率）
perf = performance(bt, bench_ret=benchmark_daily_returns)

# 简化风险模型（波动/方差/风格风险贡献）
rm = risk_model(bt["daily_ret"], style_exposure=exposures, style_returns=style_rets)
```

### 7.6 Brinson 绩效归因 + 风格归因

```python
from model_core.portfolio.attribution import brinson_attribution, style_attribution

# Brinson（行业维度）：配置效应 + 选股效应 + 交互效应
att = brinson_attribution(
    port_ret=0.15,                      # 组合收益
    bench_ret=0.08,                     # 基准收益
    port_industry_weights=np.array([0.3, 0.5, 0.2]),
    bench_industry_weights=np.array([0.4, 0.4, 0.2]),
    industry_returns=np.array([0.02, -0.01, 0.03]),
)
# {"allocation", "selection", "interaction", "total"}

# 风格归因：组合收益 = Σ 暴露×风格收益 + 特质（OLS 分解，含 R²）
sa = style_attribution(daily_ret, style_exposure, style_returns)
# {"style_contrib": {...}, "idiosyncratic", "r2"}
```

### 7.7 高频特征 API

```python
from model_core.highfreq_features import build_highfreq_features, HIGHFREQ_FEATURES

# 分钟 K 线 df（ts/open/high/low/close/volume）→ 14 个日频特征
feats = build_highfreq_features(minute_df)
# {name: np.ndarray[交易日]}
```

---

## 八、数据管道与质量保障

### 8.1 三源数据链（断联自动兜底）

| 优先级 | 源 | 复权 | 特性 |
|---|---|---|---|
| 1 | 腾讯 `qt.gtimg.cn` | qfq 前复权 | 日线翻页全历史、分钟线、实时行情（含涨跌停价/市值） |
| 2 | 新浪 `finance.sina.com.cn` | 不复权 | A股日线/分钟线翻页、国内期货（主力连续） |
| 3 | 通达信 `pytdx` | 不复权 | 全历史翻页（60 页×800 根）、**分钟线全历史**（1h 覆盖约 2 年） |

每源带重试退避（腾讯 501 / 新浪 456 均为 IP 级临时限流，等待后自动恢复）。

### 8.2 复权口径冲突防护

腾讯 qfq（复权价）与通达信/新浪（不复权价）混入同一库会导致价格在口径间来回跳变
（实测茅台 -85%/+590%）。`KlineStore.update` 自动检测：

- 逐日价格比异常比例法：公共日期价格比 >2x/<0.5x 占比 >10%，或中位数偏离 → **新数据整体覆盖**
- 公共日期 ≥3 即可检测（旧实现 ≥10，会放过小重叠混入）
- 来源/复权口径写入 `store/meta/kline_sources.json`（数据溯源）

### 8.3 机构 D3 健康检查（quality.py）

| 检查项 | 处理 |
|---|---|
| 重复日期 | 剔除（保留最后一条） |
| 非正/非有限价格 | 前值填充（首日异常整行剔除） |
| OHLC 倒挂（high<max(o,c) 等） | 自动修复 |
| 未来时间戳 | 剔除 |
| 跳变 >22%（主板/创业板合法上限之上） | 标记跳变日 → 收益标签置 0 |
| 跳变方向交替（qfq/不复权混库特征） | 分类为 `mix` → **挖矿拒绝（dirty_data）** |
| 粘滞价格（连续≥5日收盘价相同） | 报告（停牌/数据冻结） |
| 成交量恒 0 但价格变动 | 报告（缺量数据） |
| A股面值线（<1 元） | 不参与跳变判定（仙股免疫） |

### 8.4 挖矿中的数据防御

- 每只股票挖掘前自动 `check_series + clean_series + classify_jumps`
- 混库标的（`mix`）→ `status=dirty_data` 拒绝挖掘并提示删除重拉
- 跳变日收益标签置 0（数据瑕疵不产生伪收益）

---

## 九、机构级认证与回测口径

### 9.1 认证范式（C1-C5，宁缺毋滥）

| 环节 | 标准 |
|---|---|
| C1 样本外三段 | 挖掘（train）与认证（OOS）严格分离；OOS ≥ 250 根（约 1 年） |
| C2 多重检验 | DSR（Bailey 2014 原文）、PBO(CSCV)、CPCV——修正后真实因子 DSR≈0，作报告项 |
| C3 横截面认证 | 候选公式跨股票池执行 → 每日截面 RankIC（Spearman）→ 中心化块自助 p≤0.05；股票池 ≥30 只；时间分段方向一致 |
| C4 拥挤度 | 与库内因子 \|corr\| > 0.85 拒绝（防重复发明） |
| C5 方向翻转 | 负 IC 因子记录 direction，消费端翻转 |

**为什么不是滚动窗口秩相关？** 滚动 20 根窗口内的秩相关对"单调因子 × 短窗口趋势"极度敏感，
会把价格水平类因子伪高估到 |RankIC| 0.5+。块自助保留序列自相关结构，p 值直接给出显著性。

### 9.2 回测口径（B1-B6）

| 环节 | 标准 |
|---|---|
| B1 成交时点 | t 收盘信号 → t+1 执行（防收盘价成交前视） |
| B2 成本 | 单边费率（默认 0.0003）+ 可选滑点（--slippage，建议 0.0005） |
| B3 涨跌停 | 板块感知：主板 9.9% / 创业板科创板 19.9%；涨停不可买入、跌停不可卖出 |
| B4 跳变防御 | 混库/复权瑕疵日收益置 0 + 建仓冻结（不交易不付成本） |
| B5 方向防前视 | 单标的回测方向只用回测段前一半 IC 符号估计 |
| B6 组合回测 | t+1 执行、换手成本、板块涨跌停、跳变防御（P14） |

---

## 十、输出产物一览

| 产物 | 位置 |
|---|---|
| 联动矿池（三引擎知识交换中枢） | `store/meta/market_pool.json` |
| 全市场逐股三引擎挖掘汇总 | `store/meta/full_market_report.csv` |
| 断点进度（续跑用） | `store/meta/full_market_progress.json` |
| 因子库（矩阵 + 元数据索引） | `store/factors/{symbol}_{hash}.parquet` + `store/meta/factors_index.json` |
| K 线库 | `store/kline/{code}_1d.parquet` |
| K 线来源元数据（数据溯源） | `store/meta/kline_sources.json` |
| 行业分类缓存（东财） | `store/meta/industry_map.json` |
| 全 A 股清单缓存 | `store/meta/a_share_universe.json` |
| RL 训练检查点（每步自动清理） | `checkpoints/ckpt_{symbol}_*.pt` |

---

## 十一、项目结构

```
FactorNexus/
├── scripts/
│   ├── mine_full_market.py   # 全市场三引擎联动挖矿机（主入口，断点续跑）
│   ├── mine_signal.py        # 策略工厂一键流程（P24 M1→M6 全链路）
│   ├── portfolio_pipeline.py # 组合层一键流水线（中性化/合成/风险预算/归因）
│   ├── factor_backtest.py    # 因子库浏览 + 因子回测（终端工具）
│   ├── factor_monitor.py     # 因子监控（IC 衰减/方向翻转/失效预警）
│   └── mine_high_freq.py     # 高频因子挖掘（分钟级 → 日频特征）
├── model_core/
│   ├── engines/              # GP（NSGA-III 五目标）+ LLM（多智能体三重正则）
│   ├── engine.py             # RL 引擎（REINFORCE + token 公式，GPU）
│   ├── formula_dsl.py        # 10 参数染色体 DSL（param 公式）
│   ├── param_vm.py           # 参数化公式执行器（34 指标，因果后处理）
│   ├── vm.py / ops.py / vocab.py  # token 公式 StackVM（65 特征 / 66 算子）
│   ├── feature_bridge.py     # RL 特征桥（K线 → 65 特征面板）
│   ├── fundamentals.py       # 基本面管线（P18：东财业绩 + 腾讯估值）
│   ├── highfreq_features.py  # 高频因子特征（14 个日内特征）
│   ├── portfolio/            # 组合层 P14-P21（中性化/正交化/合成/优化器/
│   │                         #   barra_risk/impact_cost/归因）
│   ├── strategy_factory/     # 策略工厂 P24（dataset/walk_forward/models/
│   │                         #   ensemble/evaluate）
│   └── eval/                 # 五维评估 + DSR/PBO/CPCV 过拟合控制
├── web/
│   ├── ai_providers.py       # DeepSeek LLM 调用（批级联动）
│   └── data_sources/         # 行情源（腾讯 qfq/hfq / 新浪 / 通达信 pytdx）
├── data_pipeline/
│   ├── quality.py            # 机构 D3 健康检查与清洗
│   └── store/                # K线 / 因子 / 标签 四库分层存储
├── docs/
│   ├── INSTITUTIONAL_SPEC.md     # 机构级规范（单一事实来源）
│   ├── INSTITUTIONAL_AUDIT_2026.md # 机构级审计报告
│   ├── STRATEGY_FACTORY_PLAN.md  # 策略工厂实施方案（M1-M6 状态）
│   └── ARCHITECTURE_PRINCIPLES.md # 架构原理系统评估说明书
├── tests/                    # P6-P26 共 151 项测试
├── store/                    # 数据产物（K线 / 因子 / 矿池 / 清单）
├── strategies/               # 策略输出 best_{symbol}.json
├── _refetch_kline.py         # 污染 K 线审计 + 重拉工具
└── requirements.txt
```

---

## 十二、测试

```bash
python -m pytest tests/ -q                    # 全部 151 项（约 80 秒）
```

| 套件 | 覆盖 | 数量 |
|---|---|---|
| test_p6_full_market.py | 三引擎联动挖矿机 | 13 |
| test_p7_factor_backtest.py | 因子浏览与回测 | 19 |
| test_p8_no_leakage.py | 数据泄漏防护（B_shift_lag 等） | 5 |
| test_p9_institutional.py | 机构级修复（DSR 原文等） | 7 |
| test_p10_institutional_spec.py | 机构规范落地 | 7 |
| test_p11_cross_sectional.py | 横截面认证 | 5 |
| test_p12_data_quality.py | 数据质量层（D3 强化） | 12 |
| test_p13_backtest_engine.py | 回测引擎（滑点/涨跌停/防前视） | 9 |
| test_p14_portfolio.py | 组合层（中性化/正交化/合成/组合/归因） | 13 |
| test_p15_high_freq.py | 高频因子（特征/因果/认证） | 6 |
| test_p16_pipeline.py | 组合流水线 + 因子监控（P16/P17） | 7 |
| test_p18_20_institutional.py | 基本面/优化器/Barra/冲击/停牌（P18-P22） | 15 |
| test_p25_strategy_factory_m45.py | 策略工厂模型池/集成/持有平滑（M4-M6） | 9 |
| test_p26_top_level_risk_budget.py | 顶层风险预算面板优化（P19 接入） | 11 |

---

## 十三、机构级标准对照

详细规范见 `docs/INSTITUTIONAL_SPEC.md`（单一事实来源，依据微软 Qlib / 华泰金工 /
Bailey & López de Prado / AlphaEval / AlphaAgent）。

| 层 | 标准 | 状态 |
|---|---|---|
| 数据 D1-D4 | 复权（腾讯 hfq 后复权，P22）/停牌日识别（P22）/健康检查（超出标准）/中国模式 | ✅ | |
| 因子 F1-F3 | 因果性（测试锁定）/滚动 MAD 去极值/因果时序 zscore | ✅ |
| 评估 E1-E5 | IC/RankIC/ICIR/分层/换手/衰减/五维/块自助 | ✅ |
| 认证 C1-C5 | OOS 三段/DSR·PBO·CPCV/横截面/拥挤度/方向翻转 | ✅ |
| 回测 B1-B6 | t+1/成本+滑点/板块涨跌停/跳变防御/防前视/组合回测 | ✅ |
| 组合 P1-P5 | 五因子中性化/正交化/IC_IR·ML 合成/风险模型/Brinson 归因 | ✅ |
| 组合 P6-P10 | Markowitz·风险平价·BL 优化器/Barra+Ledoit-Wolf 风险/冲击成本/基本面管线/因子监控 | ✅ |
| 高频 | 分钟管线 + 14 日内特征 + OOS 认证入库 | ✅ |

**已披露的架构性限制**（诚实标注，见 SPEC 第六章）：
1. 单标的时序挖掘 vs 机构横截面（组合层已弥补）
2. 前复权数据（收益率/比例类不受影响）
3. 涨跌停近似（无交易所标记数据）
4. 冲击成本固定费率 + 滑点（无市场深度冲击项）
5. 市值/换手用成交额代理（无流通股本数据源）
6. 高频特征五维从简（占位 0.5，主门槛是 OOS 显著性）

---

## 十四、常见问题 FAQ

### Q1：挖矿时提示 `dirty_data`（混库）怎么办？
该标的 K 线疑似 qfq/不复权混库，已被自动拒绝（不会产出污染因子）。删除对应文件后重拉：
```bash
python _refetch_kline.py --scan-only        # 查看污染清单
python _refetch_kline.py                    # 删除 + 三源重拉
```
或直接忽略——全市场重训时会自动重拉缺失数据。

### Q2：`too_short`（清洗后仅 67 根）？
本地 K 线文件为污染短数据（历史遗留）。同 Q1 处理：删除该文件，下次挖掘自动三源重拉全历史。

### Q3：行业缓存未生成 / 行业拉取失败？
`fetch_industry_map` 首次调用自动拉取东财行业分类（缓存 `store/meta/industry_map.json`）。
东财接口偶发临时风控（与腾讯 501 同类），已加 3 次重试退避；失败自动降级为风格中性化
（`degraded=True` 在报告中标注）——不影响挖掘与入库，只影响中性化精度。

### Q4：市值/换手用成交额代理，准确吗？
机构常见做法（流动性代理）。日频截面中性化对市值暴露的剥离效果与真实市值高度相关；
有流通股本数据源时替换 `build_style_features(mcap_proxy=...)` 即可。

### Q5：DSR 为什么全是 0？
Bailey 2014 原文修正后，对 383 次挖掘试验的校正要求年化夏普 >46 才可能 DSR>0.9——
真实市场不存在这种因子，DSR=0 是**正确**表现（此前恒 1.0 是公式 bug）。入库主门槛是
横截面块自助显著性，DSR 作报告项。

### Q6：没有 DeepSeek Key 能跑吗？
能。LLM 引擎自动降级为规则化挖掘（仍产出种子与假设），GP/RL 完全不受影响。
`--engines gp,rl` 可完全绕过 LLM。

### Q7：RL 引擎需要 GPU 吗？
不需要，自动回退 CPU（`--device cpu` 强制）。但 GPU 显著加速（RTX 4060 Laptop 实测 RL 每步约 1 秒）。

### Q8：全市场 5000 只要跑多久？
实测：4 线程、三引擎全开、2000 根窗口，约 100 秒/只 → 全市场约 14 小时（可 `--skip-done` 分多次跑）。
快速巡检：`--quick-gate 0.03 --engines gp,llm` 可提速 5-10 倍。

### Q9：因子库里的 IC 列是什么口径？
横截面认证 RankIC（机构预测力标准）。`--sort ic` 按此排序；回测 Sharpe 见 `--sort sharpe`。

### Q10：组合层需要多少只股票？
横截面操作（中性化/合成/正交化）要求股票池 ≥10 只（低于自动降级：中性化退化为风格、
合成退化为等权）。认证要求 ≥30 只。建议挖到 ≥50 只后再启用完整组合层。

---

## 十五、版本历史

| 日期 | 里程碑 |
|---|---|
| 2026-08-26 | 机构级系统化对齐：数据泄漏修复（B_shift_lag/DSR/后处理）、OOS 三段认证、横截面认证、涨跌停限制 |
| 2026-08-26 晚 | 数据质量层强化（OHLC/粘滞/未来戳/跳变分类/混库防护）、KlineStore 冲突检测升级、来源元数据、回测引擎增强（滑点/板块涨跌停/跳变冻结/方向防前视） |
| 2026-08-27 | **P14 组合层**（五因子中性化/正交化/IC_IR·ML 合成/组合回测/风险模型/Brinson 归因）+ **P15 高频因子**（分钟管线 + 14 日内特征 + OOS 认证入库）；全量 100 项测试 |
| 2026-08-27 | 更名为 **FactorNexus**（全市场量化因子联动挖掘中心），因子库清空重建，30 只实测全链路验证通过 |
| 2026-08-27 | **P18-P21 机构化补齐**：基本面管线（东财业绩 + 腾讯估值）、因子监控、Barra 风险模型 + Ledoit-Wolf、冲击成本模型 + 归因；全量 131 项测试 |
| 2026-08-27 | **P24 策略工厂 M1-M6**：数据层（dataset.py）→ walk-forward（gap 防泄漏）→ 模型池（LGBM/MLP/S4）→ 集成（rank_avg/bagging/stacking）→ 评估对比 → 组合层联动；IC_IR 年化 >1.0 |
| 2026-08-27 | **挖矿稳定性修复**：RL 引擎 strategy_manager 依赖降级（兜底实现）、认证批性能修复（股票池截断 300 + 候选裁剪 + 特征面板缓存，消除第 20 只卡死）、LLM 批联动预算护栏（90s）、批级 LLM 移出持锁路径 |
| 2026-08-27 | **顶层风险预算（P19 接入）**：optimize_portfolio_panel（markowitz/risk_parity/black_litterman 面板优化，滚动协方差 + 持有期 + 防前视）接入组合流水线与策略工厂；优化器路径换手 0.8→0.05/日；全量 151 项测试 |

---

*FactorNexus · 把全 A 股当作一条矿脉，三引擎联动，机构级认证，组合层闭环。*
