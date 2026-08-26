"""
data_pipeline/data_manager.py — 多品种数据管理模块

负责从 MT5DataFetcher 加载多品种 OHLCV 数据，执行时间轴对齐，
构建 raw_dict、target_ret 张量，并委托 MT5FeatureEngineer 生成特征张量。
数据缓存在内存中，通过 reload() 刷新。
"""

from __future__ import annotations

import pandas as pd
import torch
from loguru import logger

from config import Config
from data_pipeline.fetcher import MT5DataFetcher


class MT5DataManager:
    """多品种数据管理器。

    用法：
        with MT5DataFetcher() as fetcher:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            tensor = mgr.feat_tensor   # [N, 6, T]
            ret    = mgr.target_ret    # [N, T]

    Args:
        fetcher: 已连接的 MT5DataFetcher 实例（可通过上下文管理器或
                 手动调用 connect() 确保已连接）。
    """

    def __init__(self, fetcher: MT5DataFetcher) -> None:
        self._fetcher = fetcher

        # 缓存状态
        self._symbols: list[str] = []          # 有效品种列表（>= MIN_BARS）
        self._raw_dict: dict | None = None     # {field: Tensor[N, T]}
        self._target_ret: torch.Tensor | None = None  # [N, T]

    # ──────────────────────────────────────────────────────────────────────
    # 公开方法
    # ──────────────────────────────────────────────────────────────────────

    def load(self, symbols: list[str] | None = None) -> None:
        """加载指定品种（默认 Config.SYMBOLS）的 OHLCV 数据到内存。

        - 遍历 Config.SYMBOLS，调用 fetcher.fetch() 获取每个品种数据。
        - 排除 bars < Config.MIN_BARS 的品种并记录 WARNING。
        - 对剩余品种做时间轴对齐（时间戳并集 + forward-fill）。
        - 构建 raw_dict 和 target_ret 并缓存。

        Raises:
            ValueError: 所有品种均不满足 MIN_BARS 要求时抛出。
        """
        symbol_list = list(symbols) if symbols is not None else list(Config.SYMBOLS)
        logger.info(
            f"Loading data for {len(symbol_list)} symbols: {symbol_list}"
        )

        # ── 步骤 1：拉取原始数据 ────────────────────────────────────────
        raw_dfs: dict[str, pd.DataFrame] = {}
        for symbol in symbol_list:
            df = self._fetcher.fetch(symbol, Config.TIMEFRAME, Config.BARS_COUNT)
            if len(df) < Config.MIN_BARS:
                logger.warning(
                    f"Symbol '{symbol}' has only {len(df)} bars "
                    f"(< MIN_BARS={Config.MIN_BARS}). Excluding."
                )
                continue
            raw_dfs[symbol] = df

        if not raw_dfs:
            raise ValueError(
                "No valid symbols loaded: all symbols have fewer than "
                f"{Config.MIN_BARS} bars."
            )

        self._symbols = list(raw_dfs.keys())
        logger.info(
            f"Valid symbols ({len(self._symbols)}): {self._symbols}"
        )

        # ── 步骤 2：时间轴对齐 ────────────────────────────────────────
        aligned = self._align_timelines(raw_dfs)

        # ── 步骤 3：构建 raw_dict ─────────────────────────────────────
        self._raw_dict = self._build_raw_dict(aligned)

        # ── 步骤 4：计算 target_ret ───────────────────────────────────
        self._target_ret = self._compute_target_ret(self._raw_dict["open"])

        logger.info(
            f"Data loaded. raw_dict shape: N={len(self._symbols)}, "
            f"T={self._raw_dict['open'].shape[1]}"
        )

    def reload(self) -> None:
        """刷新缓存：清空当前缓存并重新从 MT5 加载。"""
        logger.info("Reloading data from MT5...")
        self._symbols = []
        self._raw_dict = None
        self._target_ret = None
        self.load()

    # ──────────────────────────────────────────────────────────────────────
    # 属性
    # ──────────────────────────────────────────────────────────────────────

    @property
    def raw_dict(self) -> dict:
        """返回 OHLCV 原始张量字典。

        Returns:
            dict，键为 "open"、"high"、"low"、"close"、"volume"，
            每个值为形状 [N, T] 的 torch.Tensor（float32）。

        Raises:
            RuntimeError: 未调用 load() 时访问此属性。
        """
        self._ensure_loaded()
        return self._raw_dict  # type: ignore[return-value]

    @property
    def feat_tensor(self) -> torch.Tensor:
        """返回特征张量，形状 [N, F, T]（F=6）。

        委托 MT5FeatureEngineer.compute_features(raw_dict) 计算。
        使用懒导入避免循环依赖；若 model_core.features 尚未创建，
        返回全零占位张量。

        Raises:
            RuntimeError: 未调用 load() 时访问此属性。
        """
        self._ensure_loaded()
        raw = self._raw_dict  # type: ignore[assignment]

        try:
            from model_core.features import MT5FeatureEngineer  # lazy import
            # 因果安全：MT5FeatureEngineer._robust_norm 已改为滚动因果实现，
            # 全量序列传入不引入 look-ahead 泄露
            return MT5FeatureEngineer.compute_features(raw)
        except ImportError:
            # model_core/features.py 尚未在任务 5.1 中创建
            logger.warning(
                "model_core.features not found (task 5.1 not yet implemented). "
                "Returning zero placeholder feat_tensor."
            )
            n = len(self._symbols)
            t = raw["open"].shape[1]
            f = Config.INPUT_DIM
            return torch.zeros(n, f, t, dtype=torch.float32)

    @property
    def target_ret(self) -> torch.Tensor:
        """返回目标收益率张量，形状 [N, T]。

        target_ret[n, t] = log(open[n, t+2] / open[n, t+1])
        最后两个时间步设为 0（边界）。

        Raises:
            RuntimeError: 未调用 load() 时访问此属性。
        """
        self._ensure_loaded()
        return self._target_ret  # type: ignore[return-value]

    @property
    def bar_time(self) -> torch.Tensor:
        """返回最新已收盘 K 线的时间戳张量，形状 [N]（Unix 秒，int64）。

        用于实盘 runner 检测新 K 线收盘：
            if (bar_time != last_bar_time).any(): 触发调仓

        当 raw_dict 中没有 "time" 字段时（老版本 data_manager）返回全零张量。
        """
        self._ensure_loaded()
        raw = self._raw_dict
        if "time" in raw:
            # raw_dict["time"] 形状 [N, T]，取最后一列
            return raw["time"][:, -1].long()
        n = len(self._symbols)
        return torch.zeros(n, dtype=torch.int64)

    @property
    def symbols(self) -> list[str]:
        """返回有效品种列表（bars >= MIN_BARS 的品种）。"""
        return list(self._symbols)

    # ──────────────────────────────────────────────────────────────────────
    # 内部辅助方法
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """确保数据已加载，否则抛出 RuntimeError。"""
        if self._raw_dict is None:
            raise RuntimeError(
                "Data not loaded. Call MT5DataManager.load() first."
            )

    def _align_timelines(
        self, raw_dfs: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """将多品种 DataFrame 对齐到统一时间轴（时间戳交集）。

        使用交集而非并集：只保留所有品种都有真实报价的时间戳，
        彻底消除因休市 forward-fill 导致的"重复K线"问题。

        Args:
            raw_dfs: 品种名 → DataFrame（含 time 列，以及 OHLCV 列）的字典。

        Returns:
            品种名 → 对齐后 DataFrame（以 time 为索引，含 open/high/low/close/volume 列）。
            所有品种行数完全相同，无任何 NaN（交集保证每个时间戳各品种均有真实数据）。
        """
        fields = ["open", "high", "low", "close", "volume"]

        # 为每个品种建立以 time 为索引的 DataFrame
        indexed: dict[str, pd.DataFrame] = {}
        for symbol, df in raw_dfs.items():
            volume_col = "tick_volume" if "tick_volume" in df.columns else "volume"
            sub = df[["time", "open", "high", "low", "close", volume_col]].copy()
            sub = sub.rename(columns={volume_col: "volume"})
            sub = sub.set_index("time")
            # 去掉同一时间戳重复的行（取最后一条）
            sub = sub[~sub.index.duplicated(keep="last")]
            indexed[symbol] = sub

        # 构建时间戳交集索引：只保留所有品种都有报价的 bar
        inter_index: pd.Index = next(iter(indexed.values())).index
        for sub in indexed.values():
            inter_index = inter_index.intersection(sub.index)
        inter_index = inter_index.sort_values()

        if len(inter_index) < Config.MIN_BARS:
            # 交集太小时降级回并集+ffill，并记录警告
            logger.warning(
                f"Intersection timeline has only {len(inter_index)} bars "
                f"(< MIN_BARS={Config.MIN_BARS}). Falling back to union+ffill."
            )
            union_index: pd.Index = pd.Index([], dtype="int64")
            for sub in indexed.values():
                union_index = union_index.union(sub.index)
            union_index = union_index.sort_values()
            aligned: dict[str, pd.DataFrame] = {}
            for symbol, sub in indexed.items():
                reindexed = sub.reindex(union_index)
                # 仅使用 ffill（因果填充），禁止 bfill 以避免未来信息泄漏
                reindexed = reindexed.ffill()
                # 起始处的 NaN（无历史数据）填充为 0，避免下游 log/divide 出 -inf
                reindexed = reindexed.fillna(0.0)
                aligned[symbol] = reindexed[fields]
            return aligned

        logger.info(
            f"Intersection timeline: {len(inter_index)} bars "
            f"(from union of {sum(len(s) for s in indexed.values())} total)"
        )

        # 用交集索引直接切片，无需 ffill
        aligned = {}
        for symbol, sub in indexed.items():
            aligned[symbol] = sub.reindex(inter_index)[fields]

        return aligned

    def _build_raw_dict(
        self, aligned: dict[str, pd.DataFrame]
    ) -> dict[str, torch.Tensor]:
        """将对齐后的 DataFrames 转换为 {field: Tensor[N, T]} 格式。

        Args:
            aligned: 品种名 → 对齐 DataFrame 的字典。

        Returns:
            字典，键为 "open"、"high"、"low"、"close"、"volume"，
            值为 torch.float32 张量，形状 [N, T]。
        """
        fields = ["open", "high", "low", "close", "volume"]
        n = len(self._symbols)
        t = next(iter(aligned.values())).shape[0]

        raw_dict: dict[str, torch.Tensor] = {}
        for field in fields:
            rows = []
            for symbol in self._symbols:
                series = aligned[symbol][field].values
                rows.append(series)

            import numpy as np
            tensor = torch.tensor(np.array(rows), dtype=torch.float32)  # [N, T]
            raw_dict[field] = tensor

        # 加入时间戳字段，供 bar_time 属性和 K 线收盘检测使用
        import numpy as np
        time_rows = []
        for symbol in self._symbols:
            # aligned 的 index 是时间戳（Unix 秒，int64）
            time_rows.append(aligned[symbol].index.values.astype("int64"))
        raw_dict["time"] = torch.tensor(np.array(time_rows), dtype=torch.int64)  # [N, T]

        assert raw_dict["open"].shape == (n, t)
        return raw_dict

    @staticmethod
    def _compute_target_ret(open_tensor: torch.Tensor) -> torch.Tensor:
        """计算目标收益率张量。

        target_ret[n, t] = log(open[n, t+2] / open[n, t+1])，对 t ∈ [0, T-3]
        最后两个位置（t = T-2, T-1）设为 0（边界）。

        Args:
            open_tensor: 形状 [N, T] 的 open 价格张量（float32）。

        Returns:
            形状 [N, T] 的 target_ret 张量（float32）。
        """
        n, t = open_tensor.shape
        target = torch.zeros(n, t, dtype=torch.float32)

        if t >= 3:
            # open[t+2] 对应索引 2..T-1，open[t+1] 对应索引 1..T-2
            numerator   = open_tensor[:, 2:]    # [N, T-2]
            denominator = open_tensor[:, 1:-1]  # [N, T-2]

            # 防止除以零（价格理应 > 0，但防御性处理）
            safe_denom = denominator.clone()
            safe_denom[safe_denom == 0] = 1.0

            log_ret = torch.log(numerator / safe_denom)  # [N, T-2]
            target[:, :t - 2] = log_ret

        # 最后两个时间步已在初始化时设为 0（torch.zeros）

        return target
