# FactorNexus 全面审计报告（2026-08-28）

> 审计范围：全仓库 5 大模块（回测/组合、策略工厂、因子工程、RL/GP/LLM 引擎、脚本与测试、web/数据管线）逐行深度审查 + 跨模块集成核查 + 实机验证。
> 方法：6 个并行审计代理逐行审读 + 独立复核所有高危发现 + 运行 159 项测试 + 实证复现关键 bug（如编解码非恒等 18.5% 触发率）。
> 结论：**159 项测试全绿不代表无 bug**——测试由同一作者编写，未覆盖以下高危缺陷。

---

## 修复状态（2026-08-28 更新）

### ✅ 第一优先已全部修复（159 项测试通过，无回归）

| 编号 | 位置 | 修复 |
|---|---|---|
| H3 | `walk_forward.py:58` | gap 默认 `None` → 自动取 `ds.meta["horizon"]`，`gap<horizon` 告警；`mine_signal.py --gap 0` = 自动 |
| H1 | `indicator_builder.py:137` | `np.gradient` → 后向差分（因果） |
| H2 | `neutralization.py:156` | 风格特征按日因果滚动（`searchsorted` 定位 ≤t 的 bar），不再用 K 线末端快照 |
| H4 | `portfolio_pipeline.py:162` | 用 `axis_index`（并集轴 ts→行）映射写回，弃用股票内 `t_idx` |
| H5 | `attribution.py:43` | 交互效应改余量法，三效应之和恒等于 total |
| H6 | `fundamentals.py:230` | ep/bp 入库前取 `1/PE`、`1/PB` |
| H7 | `features.py:312` | WILLR 改标准公式 `(close-hh)/(hh-ll)` |
| H8 | `gp_engine.py:89` | 短板惩罚改为"中位数归一化值 < 阈值"判据，可正常触发 |
| H9 | `formula_dsl.py` | 删 `_idx` 的 None 短路 + `normalize_chrom` 清零 mask_field；实测 0/200 不一致（原 37/200） |
| H11 | `neutralization.py:183` | 只对 `usable`（有 K 线标的）建 X，因子宇宙≠K线宇宙不再 KeyError |
| H12 | `portfolio.py:41` | 纯多头只需 `n_top`（不再强制 `2×n_top`） |
| M1 | `mine_full_market.py:979` | 横截面认证只评估 OOS 段；天数门槛随 OOS 段自适应（配套更新测试断言） |

> 修复过程中还发现并修掉 2 个连带 bug：`neutralization` 的 `int(Timestamp)` 纳秒/秒单位错位（改为 `ts.value//1e9`）、`_certify_batch` OOS 段有效截面日不足导致的连坐拒绝。

### ✅ 第二优先已全部修复（159 项测试通过，无回归）

| 编号 | 位置 | 修复 |
|---|---|---|
| M2 | `portfolio.py:96` | `backtest_portfolio` 加 `limit_pct`（dict 按前缀分板块），`run_pipeline` 传科创/创业 20% |
| M3 | `ensemble.py` `_EnsembleRankModel` | 新增 `predict_with_ts` 按 ts 逐日截面排名；无 ts 时退化均值集成并告警；walk_forward 自动传 ts |
| M4 | `ensemble.py` `rank_average` | 每模型秩归一化 [0,1]（消除覆盖数不一的量纲偏置），numpy 计算避免索引对齐陷阱 |
| M6 | `dataset.py` | 因子对齐与样本行共用同一份「截断+清洗」K 线（缓存），消除清洗/截断错位 |
| M7 | `dataset.py` | `clean_series` 的 `jump_dates` 不再丢弃，跳变日收益标签置 0（`mask_jump_returns`） |
| M8 | `backtest.py:196` + `engine.py:460` | IC 与 pnl 同下标对齐（factor[t]↔ret[t]），移除偏移一个 bar |
| M9 | `optimizer.py` `risk_parity` | 对称方阵视为协方差（不再二次协方差化） |
| M10 | `optimizer.py` `_optimize_weights` | 长空不再按 Σ\|w\| 归一化（保持 Σw=1 净暴露可控） |
| M11 | `barra_risk.py` | 用 `computed` 掩码过滤回归被跳过日（旧 isfinite 把伪 0 纳入协方差） |
| M12 | `factor_monitor.py:106` | 期望收益尾部 horizon 根置 NaN（非 0），不再污染 RankIC/p |
| M13 | `report.py:105` | 空缺分组不填 0.0（避免破坏单调性/多空），不足 10 组跳过计算 |
| M16 | `five_dim.py:150` | 鲁棒性扰动 `keep` 按 `np.sort` 保时序 |
| M19 | `highfreq_features.py` | `hf_open_gap` 只取当日首根 bar 的 open/昨收 |
| M20 | `highfreq_features.py` | `hf_intra_ampl` = (当日最高-最低)/当日开盘 |
| M21 | `highfreq_features.py` | `hf_vol_corr` 循环后 flush 最后一天 |
| M22 | `indicator_builder.py` | `mfi14` 改为正/负资金流比 MFI |
| M23 | `indicator_builder.py` | `ult_osc` 改为 3 周期(7/14/28)加权买压 UO |
| M24 | `indicator_builder.py` | `obv_slope` 改为线性回归斜率（新增 `_linear_slope`） |
| M25 | `tencent_source.py` quote | 用请求前缀还原代码键（上证指数不再误映射深市） |
| M26 | `sina/tencent` | 日线 ts 统一为收盘时刻(15:00)；分钟 drop_forming 用 bar 时长 |
| M27 | `sina/tencent` | 分页边界重复 bar 按 ts 去重 |
| M28 | `sina_source.py` | 解析行字段兜底 + 多格式时间，坏行跳过不拖垮拉取 |

> 连带修复：`rank_average` 单值日 span=0 给中性 0；测试 `test_p25` 两个 rank_average 断言随归一化语义更新。

### ✅ 第三优先已全部修复（代码层可行的部分；159 项测试通过）

| 编号 | 位置 | 修复 |
|---|---|---|
| PBO | `report.py:117` + `significance.py:118` | 单因子不再恒返回伪 0.5——用 `library_factors` 构造 ≥3 组 PnL 对照矩阵真算 CSCV PBO；不足时 `pbo_valid=False`，两处打印显示 `N/A` 而非误导的 0.5 |
| CPCV embargo | `significance.py:151-187` | `embargo` 参数真正生效（训练 fold 尾部在 purge 基础上额外剔除 embargo 根，相关泄漏隔离） |
| impact 接入 | `portfolio.py` + `impact_cost.py` | `backtest_portfolio` 新增 `impact_rates` 参数（分股票单边成本率），调仓成本 = Σ\|Δw_s\|·(cost+impact)；docstring 不再误导 |

### ✅ 第四优先已全部修复（162 项测试通过，告警清零）

| 类别 | 位置 | 修复 |
|---|---|---|
| D1 | `ops.py` | 删除恒返回 0 的 `_ema` 占位 |
| D5 | `dataset.py` | 删除未使用的 `min_bars` 参数（签名/缓存/docstring） |
| D6 | `feature_bridge.py` | 修正标签 docstring（收盘到收盘，与实现一致） |
| D7 | `fundamentals.py` | 删除未用的 `close_map`/`n` 参数（+ 调用方/测试） |
| D8 | `walk_forward.py` | 删除 `tr_end_ts` 死变量 |
| D9 | `ensemble.py` | 删除 `_train_predict` 未使用导入 |
| D10 | `param_vm.py` | 清理 `_op_intercept` 的 `if False else None` 死代码 |
| D11 | `migration_audit.py` | 移除死变量 `n` 与冗余 `if True` |
| D2 | `portfolio_pipeline.py` | 删除 4 个死参数（--ml/--window/--ml-window/--ml-trees）+ `combine_ml` 未用 import |
| D3 | `portfolio_pipeline.py` | 移除 Brinson 硬编码伪数字（[0.6,0.4] 等），改为风格归因 + 标注"需行业数据跳过" |
| T1 | `test_p13` | 跳变日建仓冻结测试从 `pass` 改为真实断言（nav 在冻结段连续） |
| T2 | `test_p13` | 跳变日收益置 0 从假阳性（total_ret<1.0）改为 nav 连续断言 |
| T3 | `test_p13` | 删除两处 no-op 死代码（`*1.0`、`-0`） |
| T4 | `test_p25` | 新增 3 个回归测试：walk-forward gap 默认取 horizon、gap 泄漏预防、公式编解码往返（200 种子 0 不一致） |
| S1 | `ai_providers.py` | `_safe_endpoint` 白名单（本地回环/可信域名），阻止 token 外泄到任意地址 |
| S2 | `ai_providers.py` | 401/403 不再视为"可连通"（token 失效即不可用） |
| S3 | `ai_providers.py` | 畸形 port 字符串回退默认 51187，不再引爆 `int()` |
| T5 | `mine_full_market.py:303` | `spearmanr` 常数输入防护（std<1e-12 → 0），消除 ConstantInputWarning 与 NaN 传播 |

### ⏳ 剩余待办（依赖外部/大工程）
- 执行层（P1）：需 QMT/券商 API + 实盘权限，无法在代码库凭空实现
- regime 动态风险（P3）：`model_core/portfolio/regime.py` 需新建 + 面板优化器改造，大工程
- web UI：`web/` 仅数据源层，需全新服务端/前端
- 标签库 LabelStore：`store/labels/` 未接入管线（基础设施就绪，缺调用）
- `make_mamba`：需 mamba-ssm CUDA 依赖（文档化桩，S4 对照已可用）
- 高频合成：需真实分钟数据验证 P15 高频×日线正交化

---

## 〇、全局结论（先说最重要的）

| 维度 | 结论 |
|---|---|
| 基础设施 | ✅ 存储/数据管线/VM/词表/GP 框架**扎实**，复权冲突防护、因果算子、版本校验做得好 |
| 执行层 | ❌ **整体缺失**——无 broker/下单/实盘/模拟盘代码（IMPROVEMENT_PLAN 中 P1「最大硬伤」坐实） |
| 前视泄漏 | ⚠️ 找到 **4 处高危** + 2 处中危（TRIX 中心差分、中性化终端快照、walk-forward gap、score_panel 行错位） |
| 数学错误 | ⚠️ 找到 **5 处高危**（Brinson 交互效应恒 0、EP/BP 存反、WILLR 恒 0、短板惩罚失效、编解码非恒等） |
| 信号可信度 | ⚠️ 组合收益与 RankIC 脱节已被 P0 审计证实（-22.5% vs 随机中位 -15.9%），**当前"策略有效"不被证据支持** |
| 未完成模块 | ⚠️ 执行层/真 Mamba/高频实战/动态风险预算/web UI/基本面标签库/全市场铺满 均未完成 |

---

## 一、高危：前视泄漏（数据泄漏，会虚高策略表现）

### H1 【前视】`indicator_builder.py:137` TRIX 用 `np.gradient` 中心差分
```python
ind["trix_15"] = np.gradient(trix) / (trix + eps)
```
`np.gradient` 默认**中心差分**：`gradient[i] = (trix[i+1] - trix[i-1])/2`，t 时刻用了 t+1 未来值。
- 对照：`features.py` 的 `_trix` 用后向差分（因果）——两模块对同一指标定义矛盾。
- **影响**：TRIX 因子含未来信息，单品种回测虚高；GP/RL 若消费该指标会放大作弊因子。
- **修复**：改后向差分 `trix - np.concatenate([[trix[0]], trix[:-1]])`。

### H2 【前视】`neutralization.py:156-166` 用 K 线**终端**风格快照中性化全部历史日期
`build_style_features` 返回 `close[-1]`/`ret20=close[-1]/close[-21]`/`vol20=std(ret[-20:])`/`turn20`/`mcap`，是**序列末端**值。`neutralize_panel` 对面板里**每一天**（含很早的历史日期）用这套"未来才可知"的风格做横截面回归 → 早期日期的因子残差被未来市值/波动/换手信息污染。
- **影响**：中性化后的因子被系统性高估。
- **修复**：风格特征按日期滚动计算（`as_of` 参数），或按 t 对齐后再逐期中性化。

### H3 【前视】`walk_forward.py:58` gap 默认=5，不随 horizon
docstring 写「gap 默认=horizon」，实现硬编码 `gap: int = 5`。标签是未来 H 日收益，训练最晚样本的标签需读到 `close[ts+H]`。当 `horizon=10`（或任何 ≠5）时，gap=5 < H，训练标签越过测试首日 → **训练直接看到测试期价格**。
- **影响**：horizon≠5 时所有 walk-forward 结果（RankIC/IC_IR/组合）被泄漏污染。
- **修复**：`gap = gap if gap is not None else ds.meta.get("horizon", 5)`，并校验 `gap>=horizon`。

### H4 【对齐错位】`portfolio_pipeline.py:162-164` score_panel 行索引用股票内 `t_idx[t]`
`t_idx` 是单只股票内部 K 线位置映射，却用作 `axis_dt`（**全市场因子日期并集轴**）的行下标。当不同股票因子覆盖长度不同（新股 1200 根 vs 老股 2000 根）时，`t_idx[t]`（小）≠ axis 真实位置（大），得分被**静默错位到更早日期**，与收益面板失配。
- **影响**：横截面排序、组合回测、归因全部基于错位数据；且难察觉。
- **修复**：用 `axis_index = {t: i for i, t in enumerate(axis)}` 映射。

---

## 二、高危：数学/逻辑错误

### H5 【数学】`attribution.py:43-44` Brinson 交互效应恒为 0
```python
interaction = float(np.sum((pw - bw) * (ir - bench))) * 0.0 + \
    float(np.sum((pw - bw) * ir)) - allocation
```
第一项 `*0.0` 是死代码。代入 `allocation = Σ(pw-bw)(ir-bench)` 后：
`interaction = Σ(pw-bw)·bench = bench·Σ(pw-bw)`，权重归一化时**恒等于 0**。真正的交互效应应为 `Σ(pw-bw)(ir_selection - bench)`。
- **影响**：超额收益全部被错误归因到配置/选股效应，交互效应永远为 0。

### H6 【数学】`fundamentals.py:230-241` 入库的 EP/BP 因子实际存成 PE/PB
`field_map = {"ep": "pe", "bp": "pb", ...}`，`v = rec.get(field_map[name])` 直接取原始 PE/PB 值存入名为 `ep`/`bp` 的因子。而 `build_fundamental_factors`（:193-194）正确定义 `ep=1/PE`。两处定义矛盾——入库因子数值互为倒数，**方向完全反了**。

### H7 【数学】`features.py:312-313` Williams %R 符号反转 + clamp 塌缩
```python
willr = (hw - close) / (hw - lw + eps)
willr = torch.clamp(willr, -1.0, 0.0)
```
标准 %R 应为 `(close - hw)/(hw-lw)`（值域 [-1,0]）。当前实现正常行情下分子 `hw-close ∈ [0, hw-lw]`，比值 ∈ [0,1]，再被 clamp 到 [-1,0] → **恒为 0**，`WILLR_14` 特征基本失去信息量。对照 `indicator_builder.py:123` 有正确写法——两模块矛盾。

### H8 【逻辑】`gp_engine.py:89` 动态短板惩罚**完全失效**
```python
pct = (col < np.quantile(col, weak_quantile)).mean()
if pct > 0.5:  # 永不成立
```
`col < 第10百分位` 的比例恒 ≈ weak_quantile(0.10)，`pct > 0.5` **永远为 False**，整个短板惩罚分支永不执行。声称的"NSGA-III 动态短板惩罚（清除畸形因子）"实际无效。

### H9 【逻辑】`formula_dsl.py` 染色体编解码**非恒等**（实测 200 种子 37 个不一致，18.5%）
- `chrom_to_formula`（:158）只要 `mask_rule` 基因=0 就强制 `mask_field=None`；但 `normalize_chrom`（:132）**不清零** `mf_i` → 往返后 mask_field 静默丢失。
- **更隐蔽**：`_idx()`（:123）对 None 恒返回 0，导致 `slice=None`（SLICES[11]）解码后再编码变成 `slice=0.0`（SLICES[0]），**任何 None-slice 公式往返都漂移**。
- 实证：`python -c` 复现 18.5% 种子下 `chrom_to_formula(c).to_chrom() != normalize_chrom(c)`。
- **影响**：GP 搜索空间被污染，`__main__` 自检的 `assert` 在部分随机种子下直接失败。
- **修复**：`_idx` 需返回枚举中 None 的真实索引；`normalize_chrom` 在 mr_i==0 时清零 mf_i。

### H10 【统计】`evaluate.py:100` quantile_analysis 只取第一个有效交易日的分组
`group_rets` 把每天每组均收益追加进扁平 list，但 `g = np.array(group_rets[:n_groups])` 只截取**首日**。文档宣称的"十分组收益单调性/多空收益"实际只反映第一天。

### H11 【崩溃】`neutralization.py:183-184` 因子宇宙≠K线宇宙时 KeyError 崩溃
`n_feat = len(ind_dummy[symbols[0]])`、`X = np.array([ind_dummy[s] for s in symbols])` 用 panel 全部列 symbol 访问 `ind_dummy`，而 `ind_dummy` 只覆盖有 kline 的标的 → 只要 panel 有一列缺 K 线就抛 KeyError，流水线中断。

### H12 【崩溃】`portfolio.py:41` 纯多头也强制 `len(vals) >= n_top*2`
`long_short=False` 时本可持有 n_top 只，但股票数在 `[n_top, 2·n_top)` 时整日组合被置空。

---

## 三、中危：统计/对齐/逻辑错误

| # | 文件:行 | 问题 |
|---|---|---|
| M1 | `mine_full_market.py:979-993` | 横截面认证用全窗口（含训练段），非真 OOS——训练段过拟合残留进入"认证" |
| M2 | `backtest_portfolio`（portfolio.py:96）| 涨跌停固定 9.9%，不区分创业板/科创板 20%，与 `factor_backtest.py:280` 不一致，科创板正常 15% 涨幅被误判涨停砍仓 |
| M3 | `ensemble.py:148` | `_EnsembleRankModel` 的 rank_avg 对**全测试集**排名，非逐日截面（与独立 `rank_average` 的 `rank(axis=1)` 行为不一致） |
| M4 | `rank_average`（ensemble.py:64）| 平均原始秩未归一化，模型间覆盖股票数不同时量纲偏置 |
| M5 | `mlp_model.py:95` / `ssm_model.py:120` | 验证集按行序取尾 10%，但行序按**股票分组**非时间序——"最后 10%"是最后一只股票的尾部，破坏其防泄漏声明 |
| M6 | `dataset.py:156-163` | 日线因子无 ts 列，按**未清洗/未截断** K 线尾部对齐，行索引用清洗后 K 线——清洗/截断后整体错位 |
| M7 | `dataset.py:183` | `clean_series` 的 `jump_dates` 被丢弃，跳变伪收益标签未清零（工具 `mask_jump_returns` 存在但未用） |
| M8 | `backtest.py:196` | IC 用 `factor[t]` 对齐 `target_ret[t+1]`，而 pnl 用同下标 `position[t]·target_ret[t]`——reward 的 IC 项奖励/惩罚到错误的预测窗口 |
| M9 | `optimizer.py:91-92` | `risk_parity` 文档允许传协方差矩阵，但 ndim==2 必然被 `_cov_returns` 二次协方差化 |
| M10 | `optimizer.py:66` | 按 `Σ|w|` 归一化破坏 `Σw=1` 约束，长空净暴露不受控 |
| M11 | `barra_risk.py:156-168` | 回归被跳过日的 F/resid=0 被 `isfinite` 误纳入协方差，扭曲风险估计 |
| M12 | `factor_monitor.py:106-108` | 期望收益尾部 horizon 个 **0**（非真实收益）参与 RankIC/自助 p 值，系统性拉低 |IC| |
| M13 | `report.py:105-106` | 空十分组填 0.0，污染单调性与多空收益 |
| M14 | `significance.py:118` + `report.py:117` | PBO 因单因子 `n_f<3` 恒返回 0.5——**PBO 功能未实现**（注释误导） |
| M15 | `significance.py:151-156` | CPCV 的 `embargo` 参数从未使用（相关泄漏隔离未实现） |
| M16 | `five_dim.py:150` | 鲁棒性扰动 `rng.choice(replace=False)` 打乱时间轴，滚动 IC 无意义 |
| M17 | `mine_high_freq.py:170` | `pd.Timestamp(d).timestamp()` 按系统本地时区解释北京时间日期，非 +8 机器错位 |
| M18 | `robustness_audit.py:58-68` | `random_entry_ev` 是"固定组合持有全程"，与文档"每日随机"不符，且成本不可比 |
| M19 | `highfreq_features.py:108` | `hf_open_gap` 对全天每分钟 open 求均值，非开盘跳空 |
| M20 | `highfreq_features.py:112` | `hf_intra_ampl` 取分钟 (H-L)/O 最大值，非"日内最高-最低)/开盘" |
| M21 | `highfreq_features.py:182-185` | `hf_vol_corr` 最后一个交易日恒 NaN（循环后未 flush） |
| M22 | `indicator_builder.py:116` | `mfi14` 实为资金流强度均值，非 MFI（正/负资金流比） |
| M23 | `indicator_builder.py:140` | `ult_osc` 实为 28 期 (C-L)/(H-L) 位置比，非三周期加权 UO |
| M24 | `indicator_builder.py:114` | `obv_slope` 实为 OBV 滚动 std，非斜率 |
| M25 | `tencent_source.py:183` | `quote()` 把 000xxx 上证指数/代码推断为深市，round-trip 键错（`sh000001` 查 `sz000001`） |
| M26 | `base.py:27` vs `tongdaxin_source.py:64-75` | ts 语义不一致（开盘 vs 收盘）→ `drop_forming` 对 sina/tencent 失效（形成中 bar 泄漏）+ 跨源时序错位 |
| M27 | `tencent_source.py:112` / `sina_source.py:123` | 分页 `end`/`end_date` 设为最旧日期可能包含式重复，且整条管线无 ts 去重 |
| M28 | `sina_source.py:131-143` | 解析行无字段兜底，坏行拖垮整次拉取 |

---

## 四、未完成功能 / 死代码 / 占位

### 大模块级
| 模块 | 状态 |
|---|---|
| **执行层（P1 最大硬伤）** | ❌ 全仓库无 broker/下单/实盘/模拟盘/订单跟踪代码 |
| **web 模块** | ❌ 仅 `web/__init__.py` docstring，无任何服务实现（ai_providers/data_sources 是数据源层） |
| **真 Mamba**（`ssm_model.py:219`） | ❌ `raise NotImplementedError`，文档化但未实现 |
| **动态风险预算**（regime-aware） | ❌ IMPROVEMENT_PLAN P3 提及的 `model_core/portfolio/regime.py` 不存在 |
| **标签库 LabelStore** | ❌ `store/labels/` 目录不存在，LabelStore 未被任何管线实际调用 |
| **全市场铺满** | ⚠️ `full_market_progress.json` 仅 `["sh600000"]` **1 只**（README 宣称 5000+） |
| **高频因子实战** | ⚠️ mine_high_freq 已产出 1407 因子，但 P15 高频合成/正交化未做，仅有 14 特征 |

### 代码级占位/死代码
| # | 位置 | 问题 |
|---|---|---|
| D1 | `ops.py:143` | `_ema` 返回 `... * 0` 恒为全零，占位 stub（未调用，但属失效代码） |
| D2 | `portfolio_pipeline.py:40,364-368` | `--ml`/`--window`/`--ml-window`/`--ml-trees`/`combine_ml` 为死参数（run_pipeline 从不引用） |
| D3 | `portfolio_pipeline.py:276-279` | Brinson/风格归因用硬编码 `[0.6,0.4]` 权重与合成收益——**伪数字** |
| D4 | `impact_cost.py:13-14` | 文档承诺 `backtest_portfolio(..., impact=True)`，实际无此参数、未接入 |
| D5 | `dataset.py:81,97` | `min_bars` 参数从头到尾未使用（误导性参数） |
| D6 | `feature_bridge.py:54` | 标签 doc 写"开盘-开盘"，实现为收盘-收盘 |
| D7 | `fundamentals.py:165-212` | `close_map`/`n` 参数未使用，输出长度≠n |
| D8 | `walk_forward.py:90` | `tr_end_ts` 死代码 |
| D9 | `ensemble.py:35-36` | `_train_predict` 未使用导入 |
| D10 | `param_vm.py:231` | `xm = ym - beta*_op_mean(...) if False else None` 死代码 |
| D11 | `migration_audit.py:94-101` | `n` 死变量 + `if True` 冗余，循环浪费 |
| D12 | `mine_high_freq.py:249` 等 6 处 | 空 `pass` 分支（吞异常/未实现） |

---

## 五、测试盲区 / 假阳性

| # | 位置 | 问题 |
|---|---|---|
| T1 | `test_p13_backtest_engine.py:69-87` | `test_jump_day_freezes_position_no_turnover` 函数体只有 `pass`，唯一断言 `jump_days>=1`，核心行为零覆盖 |
| T2 | `test_p13_backtest_engine.py:90-110` | `test_jump_day_no_pnl_contribution` 的 `total_ret<1.0` 在**删除跳变防御后仍必然成立**（:95 `*1.0` 是 no-op），假阳性 |
| T3 | `test_p13_backtest_engine.py:53,95` | 两处"调整后续价格"均为恒 0/恒 1 的 no-op |
| T4 | 全仓 | **无执行层测试**（因执行层不存在）；**无 walk-forward gap 泄漏测试**（H3 未被覆盖）；**无公式编解码往返测试**（H9 的 `assert` 不在测试套件中） |
| T5 | 运行时警告 | `mine_full_market.py:303` 常数输入 `spearmanr` → ConstantInputWarning → NaN 传播 |

---

## 六、安全

| # | 位置 | 问题 |
|---|---|---|
| S1 | `ai_providers.py:663-677` | 环境变量可控 token 发送目标（`WORKBUDDY_API_ENDPOINT` 被用作 `Authorization: Bearer` 的 POST URL），共享环境可被置值外泄凭证 |
| S2 | `ai_providers.py:117-119` | 401/403 被当作"可连通"，失效 token 误报可用 |
| S3 | `ai_providers.py:368` | 畸形 port 字符串引爆 `int()`，status 接口 500 |
| S4 | `.env` 含真实 `DEEPSEEK_API_KEY` | 未被 git 跟踪（.gitignore 生效，未泄露进仓库）——风险低，但建议轮换 |

---

## 七、审计建议优先级（修复顺序）

**第一优先（正确性，先修再谈收益）**
1. H3 walk-forward gap（训练标签直接泄漏）
2. H1/H2/H4 三处前视泄漏（TRIX 中心差分 / 中性化终端快照 / score_panel 错位）
3. H5/H6/H7 三处数学错误（Brinson / EP-BP / WILLR）
4. H8/H9 两处逻辑错误（短板惩罚 / 编解码非恒等）
5. H11/H12 两处崩溃（neutralization KeyError / portfolio 纯多头）
6. M1 横截面认证含训练段

**第二优先（统计口径）**
- M2 涨跌停分板块、M12/M13 伪样本污染、M14 PBO、M16 五维扰动、M21 高频最后日 NaN

**第三优先（未完成功能）**
- 执行层（若有实盘意图）、PBO 真实现、regime 动态风险、高频合成、`impact` 接入

**第四优先（清理）**
- D1-D12 死代码/死参数、T1-T5 补真实测试、S1-S3 安全加固

---

*审计基于 2026-08-28 工作树；所有行号以当日版本为准。* 详见各模块代理报告（策略工厂/因子工程/脚本/回测/数据管线五份子报告已归档于会话）。
