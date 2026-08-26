# FactorNexus 升级技术设计文档 v2.0（实施版）

> **目标**：升级至华泰因子工厂级（A股/国内期货市场）。数据源：腾讯财经 / 通达信 / 新浪财经（弃 MT5 与 AKShare）。
> **状态**：P0-P6 全部实施完成 ✅（108 项单元测试通过，真实数据端到端验证）
> **实施日期**：2026-08（Mirage 沙盒插件卸载后，工作区恢复可写）

---

## 1. 目标架构

```
入口/编排: run_web.py · pipeline/rollout.py(半年度滚动, 规划) · scripts/backfill_a_share.py(已实施)
  ├─ Web/AI: 三页控制台 + LLM因子对话挖掘(规划) + 因子中文解释 + 五维雷达图(规划)
  ├─ 组合层: 去极值→行业市值中性化→ZScore→聚类去重→正交化→合成 (规划, factor_combine/)
  ├─ 评价层: 五维评分 + DSR + PBO(CSCV) + CPCV (规划, model_core/eval/)
  ├─ 引擎层: NSGA-III GP ✅(model_core/engines/gp_engine.py) · REINFORCE(增强,规划) · LLM多智能体(规划)
  ├─ 特征层: 指标库 65→100+（微观结构/A股/期货类），全因果实现 (规划)
  ├─ 数据层: TencentSource ✅ · SinaSource ✅ · TongdaxinSource(增强,规划) → 四库Parquet+DuckDB ✅(store/)
  └─ 监控: 因子衰减(滚动IC/拥挤度) · 涨停跌停/停牌日历 ✅(amkt.py) · 复权因子表 ✅(adjust.py)
```

## 2. 数据源矩阵（2026-08 实测锁定 ✅）

| 市场 | 通道 | 接口/要点 | 状态 |
|---|---|---|---|
| A股/指数/ETF | 腾讯(主) | 实时 `qt.gtimg.cn/q=sh600519`：[47]涨停价 [48]跌停价 [44]总市值 [45]流通市值；日线 `fqkline/get?param=code,day,start,end,640,qfq` ⚠️行序`[date,open,close,high,low,vol]`；分钟 `kline/mkline?param=code,m5,,n` 时间戳`YYYYMMDDHHMM` | ✅ tencent_source.py |
| | 新浪(备) | `CN_MarketData.getKLineData?symbol=sh600519&scale=240&datalen≤1023`(分页)，vol=股，无复权，需 Referer | ✅ sina_source.py |
| | 通达信 | pytdx 批量回填（800根/次循环），不复权，指数走 get_index_bars | 已有(增强规划) |
| 国内期货 | 新浪期货(主) | 日线 `InnerFuturesNewService.getDailyKLine?symbol=RB0`（主力连续，2009年起，含持仓量p）；分钟 `getFewMinLine?symbol=RB0&type=5`（含夜盘） | ✅ sina_source.py |
| | 通达信 | pytdx 期货服务器（实施时验证列表） | 规划 |
| | 腾讯 | ❌ 实测不支持国内期货 | - |

## 3. 已实施模块清单（P0+P1）

```
web/data_sources/
  base.py              [改] Bar 增加 extra 字段（期货持仓量 oi）
  code_map.py          [新] 三源统一代码映射（A股/指数/ETF/期货）
  tencent_source.py    [新] 腾讯源：实时行情(含涨跌停价/市值) + 日线(qfq/hfq) + 分钟线
  sina_source.py       [新] 新浪源：A股日线/分钟 + 期货日线(持仓量)/分钟(夜盘)
  factory.py           [改] 注册 tencent/sina；SOURCE_KINDS 更新为三源
data_pipeline/
  adjust.py            [新] 除权日检测 + 后复权构造 + 因子表缓存
  amkt.py              [新] 涨跌停/市值快照 + 交易日历 + 停牌辅助
  store/kline_store.py [新] K线库/因子库/标签库（Parquet 幂等增量）
  store/__init__.py    [新]
model_core/
  formula_dsl.py       [新] 华泰式 10 参数万能公式 DSL（编解码/规范化/描述）
  engines/gp_engine.py [新] NSGA-III 五目标 GP 引擎（pymoo 0.6.2 + 动态短板惩罚）
  engines/__init__.py  [新]
scripts/
  backfill_a_share.py  [新] K线回填/增量入库 CLI
  smoke_p0_sources.py  [新] 数据源实网冒烟
tests/
  test_p0_data_sources.py [新] 14 项离线单元测试（全部通过）
docs/
  UPGRADE_DESIGN_v2.0.md [本文件]
```

## 4. 五目标评价（GP 引擎已实现，华泰 2026.4 口径）

| # | 目标 | 定义 |
|---|---|---|
| f1 | \|IC\| | 时序滚动 IC 绝对值（单标的；截面版 P2 接入） |
| f2 | IC胜率 | IC > 0 的窗口占比 |
| f3 | 多头绝对收益 | 因子 top 10% 时段的未来收益均值 |
| f4 | 多头夏普 | top10% 组合年化夏普（√244） |
| f5 | 多头胜率 | top10% 组合日收益 > 0 占比 |

**动态短板惩罚**：目标当代百分位 < 10% 视为短板，按排名压缩该目标（对应前沿面下修）。

## 5. 实施完成清单（P0-P5）

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 数据层：腾讯/新浪源 + 复权 + 四库存储 + 代码映射 | ✅ |
| P1 | 挖掘引擎：参数化公式 DSL + ParamVM + NSGA-III GP | ✅ |
| P1.2 | ParamVM 真实执行器 + 指标库（34 指标）+ GP 真实数据挖掘 | ✅ |
| P2 | 评价层：AlphaEval 五维 + DSR/PBO/CPCV + FactorReport 入库 | ✅ |
| P3 | 组合层：处理流水线（去极值/中性化/去重/正交化）+ 三档合成 | ✅ |
| P4 | LLM 多智能体：Idea/Factor/Eval 三角色 + AlphaAgent 三重正则 | ✅ |
| P5 | 工程化：滚动重挖流水线 + 日更 + 衰减监控 + Web API + 信号导出 | ✅ |

**P5 新增交付**：
- `pipeline/rollout.py` 半年度滚动重挖（多窗口、多引擎、DSR 门槛入库、发布策略）
- `pipeline/daily_update.py` 日更（增量拉取幂等合并 + 因子重算 + 衰减检测 + 信号输出）
- `monitor/factor_decay.py` 衰减监控（滚动 IC/趋势/拥挤度 + FAILED/DECAY/CROWDED 三级告警）
- `web/factor_factory_api.py` 因子工厂 API（/api/factory/factors|mine|mine/status|monitor）
- `execution/signal_exporter.py` 信号清单导出（负 IC 因子方向自动翻转）
- `model_core/config.py` GPU 配置（AUTO_DEVICE/USE_TORCH_COMPILE，兼容历史 CPU 结论）

**P5.2 新增交付**：
- `scripts/mine_market.py` 全市场批量挖掘：
  - 预选层：5 个固定公式滚动 IC 快速筛选（IC<0.03 跳过深度挖掘，省 80%+ 时间）
  - 深度层：NSGA-III GP（gen/pop 可调）→ 五维+DSR 门槛入库
  - ThreadPool 并行（numpy 释放 GIL）+ 断点续跑（--skip-done）+ 汇总 CSV
  - 内置 demo 池（20 只权重股）+ 股票池文件/列表；无本地数据时自动实拉腾讯
- `model_core/engines/gp_engine.py` 参考点自适应（pop 40→35 点，消除 pymoo 警告）

实测：8 只预选 1s；宁德/招行深度挖掘各 5.9s（并行），入库 5/3 因子（DSR 1.00/0.89）

**P5.3 新增交付**：
- `web/data_sources/index_components.py` 指数成分股自动拉取：
  - 新浪 Market_Center.getHQNodeData（免费实测可用），分页拉取 + 限速
  - 支持 hs300/zz500/zz1000/sz50/cyb/zxq 六个指数节点
  - 本地缓存 store/meta/index_components_{node}.json（--refresh 强制更新）
- `scripts/mine_market.py` `--pool` 支持指数池 + `--refresh-index`

实测：沪深300 拉取 300 只成功；`--pool hs300 --limit 5 --quick` 全流程 2s

**P5.4 新增交付**：
- `scripts/tune_rl.py` RL 超参网格调优：
  - 网格：训练步数 × 批量 × walk-forward 折数（串行独立实例）
  - 每组合：best_score + 耗时 + 最优公式 IC/DSR/五维（StackVM 真实执行）
  - 文件隔离（TUNE_xx tag）+ 自动清理 + 结果 JSON
- 调优结论（茅台 500 根日线，12 组合实测）：
  - **folds=3 优于 5**（单标的 500 根下 5 折 val 段过碎）
  - **steps=10 为甜点**；推荐 `rl_steps=10 / rl_batch=64 / rl_folds=3`
    （score=77.8 全场最高，IC=+0.168，五维 0.64，17.5s）
  - 已写入 `RolloutConfig` 默认参数（rl_steps/rl_batch/rl_folds）

**P5.5 新增交付（Web 因子工厂页面）**：
- `web/static/index.html` 第 4 步「因子工厂」页：
  - 挖掘控制面板（品种/引擎 GP|LLM|both/GP代数/种群 + 触发/刷新 + 任务状态 pill）
  - 因子库表格（公式/引擎/IC/DSR/五维/Sharpe/PBO，按 DSR 排序）
  - 衰减监控面板（告警条 FAILED/DECAY/CROWDED + 滚动 IC/趋势/拥挤度表）
- `web/static/app.js`：switchPage 支持 factory + refreshFactory/loadFactoryFactors/
  loadFactoryMonitor/startFactoryMine/pollFactoryStatus（3s 轮询）/escapeHtml
- `web/static/style.css`：.fac-table/.fac-alert 等样式
- 实测：run_web.py 启动 → 页面渲染 ✓ → /api/factory/factors 返回真实因子 ✓
  （招行 DSR=0.89、茅台 RL 因子 IC=+0.35）

**P6 新增交付（全市场三引擎联动挖矿机）**：
- `scripts/mine_full_market.py` [新]：把全 A 股 5000+ 只（新浪 hs_a 全量清单，缓存本地）
  作为一条矿脉，每次运行 GP/RL/LLM 三引擎一起挖、互相联动：
  - 联动① LLM→GP：批级 LLM 多智能体挖掘（DeepSeek key 注入，无 key 规则化降级），
    假设由矿池发现日志驱动（GP→LLM 反馈），产出的染色体注入批内每只股票的
    GP 初始种群（`NSGA3FactorMiner.mine` 新增 `init_chroms` 种子支持）
  - 联动② GP→LLM：GP 精英写入共享矿池 `store/meta/market_pool.json`，
    下批 LLM 假设由矿池精英改写而成，矿池公式作新颖性约束防重复发明
  - 联动③ RL 跨品种迁移：矿池 token 精英在每只股票训练前预热进 AlphaEngine
    精英回放池（实测：sh600004 上发现的公式，sh600000 第 0 步即被评估为新最优）
  - 联动④ 统一裁决：三引擎候选统一五维 + DSR 评估，批内相关 >0.95 拥挤去重，
    达标入 FactorStore 并回灌矿池
  - CUDA GPU 加速：自动检测并路由 RL 引擎到 GPU（实测 RTX 4060）；
    线程感知控制台路由代理抑制 RL 每步刷屏日志（tqdm.write 走 stdout，仅回显"新最优"）
  - 量大管饱：全历史日线自动回填（腾讯翻页），`--bars` 控挖掘窗口；
    断点续跑（progress JSON + 因子库双保险）；`--workers` 并行
  - 新增 `model_core/engines/gp_engine.py` 的 `init_chroms` 初始种群种子注入
    （向后兼容，默认行为不变）
  - `tests/test_p6_full_market.py` [新] 13 项单元测试（矿池/清单/种子注入/联动冒烟）
- 实测：2 只 × 三引擎端到端（GPU RL 训练 + DeepSeek 批级联动 + 入库 7 因子，
  矿池 18 条精英），日志干净

**已知后续任务（P5.6+）**：
- 通达信期货服务器验证（期货分钟线备选通道）
- 多品种 RL 调优（不同周期/品种的 rl 参数可能不同）
- 全市场三引擎联动挖掘的 Web 化（因子工厂页展示矿池/联动状态）
- 因子工厂页增强（挖掘历史/因子详情抽屉/五维雷达图）

## 6. 实施备注

- 环境：Python 3.12.10、torch 2.6.0+cu124、pymoo 0.6.2（已安装）、deap 1.4
- 关键坑（实测）：
  1. 腾讯 K 线行序为 [date, open, close, high, low, vol]，非标准 OHLC
  2. 腾讯 quote 返回纯代码 key（600519），需规范化为 sh600519
  3. 新浪 JSONP 格式为 `var _=([...]);`（括号包裹），正则需剥括号
  4. 新浪 vol 单位=股，腾讯=手（1手=100股）
  5. 腾讯实时含涨停价/跌停价/市值字段——涨跌停约束与中性化免额外数据源
