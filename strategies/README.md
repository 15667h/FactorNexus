# MT5 AlphaGPT — 策略因子库

> **注意**：`*.json` 策略文件仅保存在本地，已加入 `.gitignore`，不会上传到 GitHub。

> 生成时间: 2026-07-06 09:38
> 训练框架: AlphaGPT (REINFORCE + Transformer + 公式 DSL)
> 回测周期: H1 时间帧，独立验证脚本 `verify_all_strategies.py`

## 策略一览

| 策略名 | 品种组 | 品种 | 数据长度 | 年化收益 | Sharpe | MDD | 多空比 | 判定 |
|--------|--------|------|----------|----------|--------|-----|--------|------|
| forex_v1 | forex | EURUSD, USDJPY | 8.01年 (49998 bars) | +3.23% | 0.879 | 7.03% | 38/38 | ✅ VALID |
| forex_v2 | forex | EURUSD, USDJPY | 8.01年 (49998 bars) | +2.85% | 0.684 | 6.88% | 49/49 | ✅ VALID |
| index_v1 | index | US30.cash, US100.cash, US500.cash | 5.15年 (32134 bars) | -1.79% | -1.136 | 9.68% | 11/7 | ❌ INVALID |
| index_v2 | index | US30.cash, US100.cash, US500.cash | 5.15年 (32134 bars) | -1.79% | -1.136 | 9.68% | 11/7 | ❌ INVALID |
| metals_comm_v1 | metals_comm | XAUUSD, AAVUSD, COCOA.c | 1.37年 (8546 bars) | +125.63% | 3.420 | 22.18% | 49/49 | ❌ INVALID |
| metals_comm_v2 | metals_comm | XAUUSD, AAVUSD, COCOA.c | 1.37年 (8546 bars) | +151.58% | 4.072 | 13.62% | 55/42 | ⚠️ SUSPICIOUS |

> index_v1 和 index_v2 公式相同（v2 是 v1 的副本）

## 公式详情

### forex_v1
- **Token IDs**: [20, 53, 57, 73, 34, 73, 56, 50]
- **Decoded**: AROON_OSC_25 → TS_MIN_10 → EMA_20 → CS_SCALE → SIGN → CS_SCALE → EMA_5 → MOMENTUM_5
- **训练 best_score**: 0.4851
- **公式逻辑**: Aroon 振荡器的 10 周期最小值经 EMA 平滑后横截面标准化，取符号再标准化，最后用 EMA+动量平滑

### forex_v2
- **Token IDs**: [21, 3, 60, 29, 60, 42, 73, 54]
- **Decoded**: DMI_DIFF_14 → HL_RANGE → TS_MIN_20 → SUB → TS_MIN_20 → TS_MEAN_20 → CS_SCALE → WMA
- **训练 best_score**: 0.6354
- **公式逻辑**: DMI 差值减去高低价差的 20 周期最小值，再减去另一个 TS_MIN_20，经 TS_MEAN_20 平滑后横截面标准化，最后 WMA 加权

### index_v1 (= index_v2)
- **Token IDs**: [13, 47, 75, 47, 77, 40, 86, 82]
- **Decoded**: PPO → TS_RANK_10 → TS_SUM_5 → TS_RANK_10 → TS_SUM_20 → TS_MEAN_5 → CLIP → SQRT
- **训练 best_score**: 0.4464
- **问题**: TS_RANK 输出 [0,1) 恒正，导致整个公式链恒正，因子退化为 beta 因子

### metals_comm_v1
- **Token IDs**: [27, 57, 52, 88, 57, 17, 79, 50]
- **Decoded**: CS_ZSCORE_RET20 → EMA_20 → TS_MAX_10 → TANH_SQUASH → EMA_20 → AMIHUD_ILLIQ → MAX → MOMENTUM_5
- **训练 best_score**: 2.3198
- **注意**: 仅 1.37 年数据，MDD 22.18%，XAUUSD 亏损 -11.23%

### metals_comm_v2
- **Token IDs**: [25, 11, 47, 61, 26, 71, 70, 89]
- **Decoded**: SAR_DIST → TREND_STRENGTH_50 → TS_RANK_10 → TS_MAX_20 → ROLL_KURT_20 → DELTA_5 → TS_DECAY_EXP_5 → IF_GT
- **训练 best_score**: 2.6153
- **注意**: 仅 1.37 年数据，MDD 13.62%

## 回测方法

- **target_ret**: log(open[t+2] / open[t+1]) — 无前视偏差
- **仓位**: pos = tanh(factor)，|pos| < 0.05 时归零
- **PnL**: pos[t] × target_ret[t] - |pos[t] - pos[t-1]| × cost_rate
- **成本**: 单边 0.01% (1bp)
- **年化**: mean(daily_pnl) × 6240 (H1 bars/year)
- **Walk-Forward**: 4 折时间序列切分，每折独立计算年化/Sharpe/MDD
- **成本压力**: 1x/2x/3x/5x 成本倍数

## 文件结构

```
strategies/
├── README.md                          # 本文件
├── STATUS_20250705.md                 # 训练状态记录
├── best_forex.json                    # forex_v2 策略
├── best_index.json                    # index_v2 策略
├── best_metals_comm.json              # metals_comm_v2 策略
├── best_group_forex.json              # forex 组策略
├── best_EURUSD.json                   # EURUSD 单品种
├── best_USDJPY.json                   # USDJPY 单品种
├── archive/
│   ├── best_forex_20250705_pre_refactor.json       # forex_v1 策略
│   ├── best_index_20250705_pre_refactor.json       # index_v1 策略
│   └── best_metals_comm_20250705_pre_refactor.json # metals_comm_v1 策略
└── verification_results.json           # 独立回测验证详细结果
```

## 使用方式

```python
import json
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.data_manager import MT5DataManager
from config import Config
from strategy_manager.signal import compute_target_positions_stateless

# 加载策略
with open("strategies/best_forex.json") as f:
    strat = json.load(f)
formula = strat["formula"]

# 解码公式
decoded = [FORMULA_VOCAB.token_names[t] for t in formula]
print(" → ".join(decoded))

# 加载数据
Config.SYMBOLS = ["EURUSD", "USDJPY"]
with MT5DataFetcher(offline=True) as fetcher:
    mgr = MT5DataManager(fetcher)
    mgr.load()
    
    # 执行公式
    vm = StackVM()
    factor = vm.execute(formula, mgr.feat_tensor)
    
    # 计算仓位
    positions = compute_target_positions_stateless(factor)
    # positions: [N, T] in [-1, +1]
```

## 免责声明

- metals_comm 组仅 1.37 年数据，统计显著性不足
- index 组策略已确认无效（TS_RANK 恒正问题），保留仅供分析
- 实盘交易需考虑滑点、点差、流动性等额外成本
- 过去表现不代表未来收益
