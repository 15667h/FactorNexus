## FactorNexus

基于深度神经网络强化学习 + 遗传规划 + LLM 多智能体的**全市场量化因子联动挖掘中心**：把全 A 股 5000+ 只股票作为一条矿脉，GP / RL / LLM 三个因子挖掘引擎每次运行一起挖、互相联动（共享矿池 / 批级 LLM 引导 GP / RL 精英跨品种迁移），CUDA GPU 加速，断点续跑。

---

## 它做什么

1. **全市场矿脉**：新浪 `hs_a` 节点拉取全 A 股清单（沪深 5000+ 只，过滤北交所，本地缓存）
2. **三引擎联动挖掘**：每次运行 `scripts/mine_full_market.py`，GP / RL / LLM 一起挖：
   - **联动① LLM→GP**：批级 LLM 多智能体挖掘（DeepSeek key 注入，无 key 规则化降级），产出的公式染色体注入批内每只股票的 GP 初始种群
   - **联动② GP→LLM**：GP 精英写入共享矿池，下批 LLM 假设由矿池发现日志驱动 + 新颖性约束防重复发明
   - **联动③ RL 跨品种迁移**：矿池 token 精英预热进每只股票 RL 的精英回放池
   - **联动④ 统一裁决**：三引擎候选统一五维 + DSR 评估，批内拥挤去重，达标入因子库并回灌矿池
3. **数据量大管饱**：全历史日线自动回填（腾讯财经 fqkline 翻页），挖掘窗口可调（默认 2000 根 ≈ 8 年）
4. **CUDA GPU 加速**：自动检测并路由 RL 引擎（AlphaEngine / AlphaGPT）到 GPU

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填写 DEEPSEEK_API_KEY（LLM 批级真挖掘，可选）

# 全市场三引擎联动挖矿（默认全 A 股 5000+ 只，三引擎全开）
python scripts/mine_full_market.py

# 先试 5 只
python scripts/mine_full_market.py --limit 5

# 断点续跑（跳过已挖标的）
python scripts/mine_full_market.py --skip-done

# 终端显示每只股票的详细挖掘过程（GP 非支配解/LLM 假设/RL 每步/统一裁决明细）
python scripts/mine_full_market.py --verbose

# 预选过滤省时（|IC| < 0.03 的股票跳过深度挖掘）
python scripts/mine_full_market.py --quick-gate 0.03

# 只开 GP+LLM（快速模式，不跑 RL）
python scripts/mine_full_market.py --engines gp,llm
```

关键参数：`--workers` 并行线程、`--gen/--pop` GP 规模、`--rl-steps/--rl-batch/--rl-folds` RL 训练、`--bars` 挖掘窗口（0=全历史）、`--llm-batch` 批级 LLM 联动间隔、`--device auto|cuda|cpu`、`--verbose` 详细过程。

## 因子库浏览 + 因子回测（终端）

```bash
# 交互模式：显示已入库因子列表 → 输入编号 → 查看体质 + 回测 + ASCII 资金曲线
python scripts/factor_backtest.py

# 只列因子库（编号/品种/hash/引擎/公式/IC/DSR/五维/Sharpe）
python scripts/factor_backtest.py --list              # 按 Sharpe（默认）
python scripts/factor_backtest.py --list --sort ic    # 按横截面认证 RankIC（机构预测力标准）

# 查看第 3 个因子的详细体质（五维五分量/DSR/PBO/CPCV/ICIR/染色体）
python scripts/factor_backtest.py --factor 3

# 直接回测指定因子（三种定位方式，推荐编号——见 --list）
python scripts/factor_backtest.py --backtest 5              # 编号（列表第 5 个）
python scripts/factor_backtest.py --backtest sh600519       # 品种唯一因子直接回测
python scripts/factor_backtest.py --backtest sh600519 2f0134  # symbol + hash 前缀

# 回测 DSR 前 5 个因子（批量体检）
python scripts/factor_backtest.py --top 5 --backtest-all
```

回测口径：因子序列与 K 线尾部对齐 → `tanh(因子 zscore)` 连续仓位 × 未来收益 − 换手成本，
输出总收益/年化/夏普/索提诺/最大回撤/Calmar/盈亏比/胜率/换手/IC/ICIR/rankIC + 资金曲线。

## 输出

| 产物 | 位置 |
|------|------|
| 联动矿池（三引擎知识交换中枢） | `store/meta/market_pool.json` |
| 全市场逐股三引擎挖掘汇总 | `store/meta/full_market_report.csv` |
| 断点进度（续跑用） | `store/meta/full_market_progress.json` |
| 达标因子库 | `store/factors/{symbol}_{hash}.parquet`（`store/meta/factors_index.json`） |
| 全 A 股清单缓存 | `store/meta/a_share_universe.json` |
| K 线库（后复权日线） | `store/kline/{code}_1d.parquet` |

## 项目结构

```
FactorNexus/
├── scripts/
│   ├── mine_full_market.py   # 全市场三引擎联动挖矿机（入口，--verbose 详细过程）
│   └── factor_backtest.py    # 已入库因子浏览 + 因子回测（终端工具）
├── model_core/
│   ├── engines/              # GP（NSGA-III 五目标）+ LLM（多智能体三重正则）
│   ├── engine.py             # RL 引擎（REINFORCE + token 公式，GPU）
│   ├── formula_dsl.py        # 10 参数染色体 DSL（param 公式）
│   ├── param_vm.py           # 参数化公式执行器（34 指标）
│   ├── vm.py / ops.py / vocab.py # token 公式 StackVM（65 特征 / 66 算子）
│   ├── feature_bridge.py     # RL 特征桥（K线 → 65 特征面板）
│   └── eval/                 # 五维评估 + DSR/PBO/CPCV 过拟合控制
├── web/
│   ├── ai_providers.py       # DeepSeek LLM 调用（批级联动）
│   └── data_sources/         # 行情源（腾讯 / 新浪 / 通达信 等）
├── data_pipeline/store/      # K线 / 因子 / 标签 四库分层存储
├── strategy_manager/signal.py # 信号口径（RL 引擎内部使用）
├── tests/                    # test_p6_full_market.py + test_p7_factor_backtest.py
├── store/                    # 数据产物（K线 / 因子 / 矿池 / 清单）
└── strategies/               # 策略输出 best_{symbol}.json
```

## 环境要求

- Python 3.12+（实测 3.12.10）
- PyTorch（CUDA 版，RL 引擎 GPU 加速）、pandas、pyarrow、numpy、pymoo、requests、tqdm、python-dotenv（见 `requirements.txt`）
- 可选：NVIDIA GPU（自动检测，无 GPU 自动回退 CPU）
- `.env` 中 `DEEPSEEK_API_KEY` 用于 LLM 批级真挖掘（缺省时 LLM 走规则化降级，不影响流程）

## 测试

```bash
python -m pytest tests/test_p6_full_market.py -v        # P6 挖矿机（13 项）
python -m pytest tests/test_p7_factor_backtest.py -v    # P7 因子回测（19 项）
python -m pytest tests/test_p8_no_leakage.py -v         # P8 数据泄漏防护（5 项）
python -m pytest tests/ -q                              # 全部 37 项
```

## 数据泄漏审计记录（2026-08-26）

对因子库最强因子的回测（总收益 950 亿倍、IC 0.43）触发全面审计，发现并修复：

| 问题 | 根因 | 修复 |
|------|------|------|
| **B_shift_lag 未来泄漏**（严重） | `param_vm` 用 `np.roll(b, -lag)` 实现"滞后"，符号反了：lag>0（"滞后N"）实际取**未来 N 天**的 B 值 → 因子含未来收益 → IC 虚高至 0.4（真实水平 ~0.02）、回测收益爆炸、入库报告 max_dd=0 | 改为绝对值滞后（`b[t]` 只用 `b[t-k]` 及以前，k=\|lag\|），任何方向都杜绝未来信息 |
| **Skew/Kurt 掩码广播错误** | `_op_skew`/`_op_kurt` 掩码分支 `(n,)/(n,1)` 广播成 `(n,n)` 矩阵 → 该类公式执行崩溃被静默淘汰 | 归一化分母改一维 |
| **回测双重标准化** | 因子入库时已是因果 expanding zscore，回测再全样本 zscore → 引入全样本前视 + 改变因子分布 | 回测直接 `tanh(因子)` |

修复后同一公式 IC 从 +0.4321 降至 -0.0037（与随机公式基准 |IC|≈0.018 一致）。
**泄漏时代的 32 个入库因子已全部作废**（备份于 `_deleted_backup_20260826/factors_leaked/`），需重新挖掘。
已知限制：RL token 因子评估为全窗口样本内口径（选样偏差致 IC 偏乐观），后续可加 OOS 验证段。
