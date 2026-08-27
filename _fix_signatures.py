"""统一其余数据源的 fetch_bars 签名（加 adjust 参数，接口一致性）。"""
import pathlib
import re

for fn in ("domestic_futures_source.py", "mt5_source.py",
           "okx_source.py", "tradingview_source.py"):
    p = pathlib.Path("web/data_sources") / fn
    c = p.read_text(encoding="utf-8")
    # 多行签名：def fetch_bars(\n self, ..., drop_forming: bool = True,
    new = re.sub(
        r"(def fetch_bars\(\s*self, symbol: str, timeframe: str, n: int, "
        r"drop_forming: bool = True,\s*\))",
        r'\1\n        adjust: str = "raw",   # 接口统一（该源不实现复权，忽略）',
        c, count=1)
    if new != c:
        p.write_text(new, encoding="utf-8")
        print(fn, "OK(multi-line)")
        continue
    # 单行签名：def fetch_bars(self, symbol, timeframe, n, drop_forming=True)
    new = re.sub(
        r"(def fetch_bars\(\s*self, symbol: str, timeframe: str, n: int, "
        r"drop_forming: bool = True\s*\))",
        r'\1,\n                 adjust: str = "raw"),   # 接口统一',
        c, count=1)
    if new != c:
        p.write_text(new, encoding="utf-8")
        print(fn, "OK(single-line)")
        continue
    print(fn, "SKIP(no match)")
