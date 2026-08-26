"""实时行情数据源抽象层。

统一接口，供 realtime_manager 拉取 K 线并转换为 FactorNexus 特征引擎所需的
raw_dict（torch 张量 [1, T]，升序=最旧在前）。

参考 PA_Agent 的 DataSource 设计，但做成自包含、可选依赖优雅降级。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# 项目统一的周期字符串（各源在内部映射到自己的常量）
CANON_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M")


class DataSourceError(Exception):
    """数据源通用错误。"""


class DataSourceUnavailable(DataSourceError):
    """依赖缺失或无法连接（前端据此灰显该源）。"""


@dataclass
class Bar:
    """单根 K 线（ts = 开盘时间的 Unix 秒）。"""
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    # 可选附加字段（如期货持仓量 oi），所有既有调用保持兼容
    extra: dict | None = None


class DataSource(ABC):
    """K 线数据源统一接口。"""

    kind: str = ""
    label: str = ""

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """返回 (是否可用, 提示文案)。用于前端灰显与安装引导。"""

    @abstractmethod
    def supported_timeframes(self) -> list[str]:
        """返回该源支持的周期字符串列表（CANON_TIMEFRAMES 子集）。"""

    @abstractmethod
    def preset_symbols(self) -> list[str]:
        """返回预设/常用品种列表（不阻塞网络）。"""

    @abstractmethod
    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        """拉取最近 n 根已收盘 K 线，升序（最旧在前）。

        drop_forming=True 时剔除当前正在形成的 bar，使最后一根为「最后已收盘 bar」。
        """

    def connect(self) -> None:  # noqa: B027 - 可选
        """建立/复用连接（可选）。"""

    def disconnect(self) -> None:  # noqa: B027 - 可选
        """断开连接（可选）。"""


def bars_to_raw_dict(bars: list[Bar]):
    """将升序 Bar 列表转换为 FactorNexus 特征引擎所需的 raw_dict（torch [1, T]）。"""
    import torch

    if not bars:
        raise DataSourceError("空 K 线序列")

    def col(vals: list[float]):
        return torch.tensor([vals], dtype=torch.float32)

    return {
        "open": col([b.open for b in bars]),
        "high": col([b.high for b in bars]),
        "low": col([b.low for b in bars]),
        "close": col([b.close for b in bars]),
        "volume": col([max(b.volume, 0.0) for b in bars]),
        "time": col([float(b.ts) for b in bars]),
    }
