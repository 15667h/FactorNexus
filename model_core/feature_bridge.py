"""
model_core/feature_bridge.py — 特征层统一桥（P5.1）

目标：让 RL 引擎（AlphaEngine，token 公式 + FORMULA_VOCAB 65 特征）能消费
A股/期货数据，与参数化公式引擎（ParamVM + indicator_builder 34 指标）并存。

统一方案：
  - RL 引擎继续使用原生 MT5FeatureEngineer（纯 OHLCV → [N, 65, T]，N=1 单标的），
    特征顺序 = FORMULA_VOCAB.feature_names（token 索引直接可用）
  - ParamVM/GP/LLM 引擎继续使用 indicator_builder（34 指标命名空间）
  - 两套特征系统由本模块桥接：RLDataManager 把 K线 DataFrame 包装成
    AlphaEngine 所需的数据视图（feat_tensor / target_ret / raw_dict）

用法：
    from model_core.feature_bridge import RLDataManager
    dm = RLDataManager(df, horizon=5)
    eng = AlphaEngine(data_manager=dm, target_symbol=symbol, n_folds=3)
    eng.train(start_step=0, end_step=10, verbose_header=False)
    # eng.best_formula 是 token 列表 → StackVM 执行
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from model_core.vocab import FORMULA_VOCAB


def build_feature_panel(df: pd.DataFrame) -> torch.Tensor:
    """K线 DataFrame → [1, F, T] 特征面板（F=FORMULA_VOCAB.feature_count）。

    走 MT5FeatureEngineer.compute_features（纯 OHLCV，全部 65 特征因果计算）。
    """
    from model_core.features import MT5FeatureEngineer

    n = 1
    raw: dict[str, torch.Tensor] = {}
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"K线缺列 {col}（需要 open/high/low/close/volume）")
        vals = np.asarray(df[col].values, dtype=np.float32)
        raw[col] = torch.tensor(vals.reshape(n, -1), dtype=torch.float32)
    feat = MT5FeatureEngineer.compute_features(raw)  # [1, F, T]
    assert feat.shape[1] == FORMULA_VOCAB.feature_count, (
        f"特征数不匹配: {feat.shape[1]} != {FORMULA_VOCAB.feature_count}"
    )
    return feat


def build_target_ret(df: pd.DataFrame, horizon: int = 5) -> torch.Tensor:
    """K线 → [1, T] 未来收益标签（收盘到收盘的简单收益）。

    D6 修复：旧 docstring 写"开盘到开盘"，实现为收盘到收盘
    （ret[t] = close[t+horizon]/close[t] - 1）。与挖矿/回测口径一致。
    末尾 horizon 位置为 0。
    """
    close = df["close"].values.astype(np.float64)
    T = len(close)
    target = np.zeros(T, dtype=np.float32)
    if T > horizon:
        # 简单收益（A股口径常用）：ret[t] = close[t+horizon]/close[t] - 1
        target[:T - horizon] = close[horizon:] / close[:-horizon] - 1.0
    return torch.tensor(target.reshape(1, -1), dtype=torch.float32)


@dataclass
class RLDataManager:
    """AlphaEngine 兼容数据视图（单标的）。"""

    df: pd.DataFrame
    horizon: int = 5

    def __post_init__(self) -> None:
        if self.df.empty:
            raise ValueError("K线 DataFrame 为空")
        self._feat: torch.Tensor | None = None
        self._ret: torch.Tensor | None = None
        self._time: torch.Tensor | None = None
        self._symbols = ["single"]

    # ── AlphaEngine 需要的属性 ───────────────────────────────────────────

    @property
    def feat_tensor(self) -> torch.Tensor:
        if self._feat is None:
            self._feat = build_feature_panel(self.df)  # [1, F, T]
        return self._feat

    @property
    def target_ret(self) -> torch.Tensor:
        if self._ret is None:
            self._ret = build_target_ret(self.df, self.horizon)  # [1, T]
        return self._ret

    @property
    def raw_dict(self) -> dict:
        return {
            "open": self.feat_tensor.new_tensor(self.df["open"].values.reshape(1, -1)),
            "time": self._time_tensor(),
        }

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def _time_tensor(self) -> torch.Tensor:
        if self._time is None:
            if "ts" in self.df.columns:
                ts = np.asarray(self.df["ts"].values, dtype=np.int64).reshape(1, -1)
                self._time = torch.tensor(ts, dtype=torch.int64)
            else:
                # 无时间戳 → 用 bar 序号（estimate_periods_per_year 会告警回退）
                self._time = torch.zeros(1, len(self.df), dtype=torch.int64)
        return self._time

    # ── 兼容方法（engine 未用到，但保持 DataManager 约定）────────────────

    def load(self, symbols: list[str] | None = None) -> None:
        pass

    def reload(self) -> None:
        self._feat = self._ret = self._time = None

    @property
    def bar_time(self) -> torch.Tensor:
        return self._time_tensor()[:, -1].long()


def execute_token_formula(tokens: list[int], dm: RLDataManager) -> torch.Tensor:
    """StackVM 执行 token 公式 → [T] 因子（已标准化）。"""
    from model_core.vm import StackVM

    vm = StackVM()
    feat = dm.feat_tensor  # [1, F, T]
    res = vm.execute([int(t) for t in tokens], feat)
    if res is None:
        raise ValueError(f"公式执行失败: {tokens}")
    return res[0]  # [T]
