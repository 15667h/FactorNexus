"""
web/data_sources/code_map.py — 三源统一的证券代码映射（A股/指数/ETF/期货）

把用户输入的任意常见代码形式归一化为各源需要的代码：

  TencentSource / SinaSource(A股)  : sh600519 / sz000001 / sh000001 / sz399006
  SinaSource(期货)                 : RB0（主力连续）/ RB2510（具体合约）
  TongdaxinSource                  : 直接用 code_map 判定市场前缀后再转纯数字

规则（对齐通达信 _parse_market 的既有约定，避免三套规则漂移）：
  - 显式前缀：sh / sz / SH / SZ（A股、指数、ETF）
  - 无前缀纯数字：6/5/9 开头 → 沪市；0/3 开头 → 深市
  - 期货：字母+数字（RB0、IF0、au0、rb2510），保持原样，由 SinaSource 期货通道使用
"""
from __future__ import annotations

from dataclasses import dataclass

# ── 归一化结果 ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NormalizedCode:
    """统一代码视图。"""
    raw: str            # 用户原始输入
    market: str         # "sh" | "sz" | "futures"
    code: str           # 纯代码（无前缀）：600519 / RB0
    tencent: str        # 腾讯代码：sh600519 / sz000001
    sina: str           # 新浪 A股代码：sh600519 / sz000001
    sina_futures: str   # 新浪期货代码（仅期货）：RB0
    tdx_market: int     # 通达信市场号：1=上海 0=深圳（期货为 -1）
    tdx_code: str       # 通达信代码

    @property
    def is_index(self) -> bool:
        """是否为指数（沪 000xxx / 深 399xxx）。"""
        if self.market == "sh":
            return self.code.startswith("000")
        if self.market == "sz":
            return self.code.startswith("399")
        return False

    @property
    def is_futures(self) -> bool:
        return self.market == "futures"


# ── 解析 ──────────────────────────────────────────────────────────────────────

def normalize_code(symbol: str) -> NormalizedCode:
    """把任意常见代码形式归一化为统一视图。"""
    raw = symbol.strip()
    upper = raw.upper()

    # 显式前缀（优先于期货判定：SH/SZ 开头的是 A股/指数/ETF）
    if upper.startswith("SH") or upper.startswith("SZ"):
        market = "sh" if upper.startswith("SH") else "sz"
        pure = raw[2:].strip()
        return _build(raw, market, pure)

    # 期货：字母开头（RB0、IF0、AU0、rb2510）——此时已排除 SH/SZ 前缀
    if upper[:1].isalpha() and any(c.isdigit() for c in upper):
        return NormalizedCode(
            raw=raw, market="futures", code=upper,
            tencent="", sina="", sina_futures=upper,
            tdx_market=-1, tdx_code=upper,
        )

    # 无前缀：按首字符推断市场（对齐通达信规则）
    if pure := raw.strip():
        if pure[0] in ("6", "5", "9") or pure.startswith("11") or pure.startswith("13"):
            return _build(raw, "sh", pure)
        return _build(raw, "sz", pure)

    raise ValueError(f"无法解析证券代码: {symbol!r}")


def _build(raw: str, market: str, code: str) -> NormalizedCode:
    if not code.isdigit():
        raise ValueError(f"无效的证券代码: {raw!r}")
    return NormalizedCode(
        raw=raw, market=market, code=code,
        tencent=f"{market}{code}",
        sina=f"{market}{code}",
        sina_futures="",
        tdx_market=1 if market == "sh" else 0,
        tdx_code=code,
    )


# ── 常用品种预设 ──────────────────────────────────────────────────────────────

A_SHARE_PRESETS = [
    # 指数
    "sh000001", "sz399001", "sz399006", "sh000300", "sh000688", "sz899050",
    # 权重股
    "sh600519", "sz300750", "sz002594", "sh600036", "sh601318", "sz000858",
    # ETF
    "sh510300", "sh560860", "sz159919",
]

FUTURES_PRESETS = [
    # 主力连续（新浪期货代码）
    "RB0", "HC0", "I0", "JM0", "J0", "SF0", "SM0", "FG0", "ZC0", "MA0",
    "TA0", "RU0", "AU0", "AG0", "CU0", "AL0", "ZN0", "NI0", "SN0", "AO0",
    "IF0", "IC0", "IM0", "IH0",
]
