"""诊断 Top-10 样本=0。"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from data_pipeline.store.kline_store import FactorStore, KlineStore

store, kstore = FactorStore("store"), KlineStore("store")
factors = store.list_factors()
per_sym = {}
for f in factors:
    sym = f["symbol"]
    rep = f.get("report") or {}
    inner = rep.get("meta") or {}
    kind = rep.get("kind", "")
    fdf = store.load(sym, f["hash"])
    if fdf is None or "factor" not in fdf.columns or fdf.empty:
        continue
    if kind == "highfreq":
        feat = rep.get("feature") or ""
        col = feat if feat else None
    else:
        col = f"{sym}_{f['hash'][:10]}"
    if not col:
        continue
    rankic = float(inner.get("cert_rankic", rep.get("cert_rankic", 0.0)))
    per_sym.setdefault(sym, []).append((col, rankic, fdf))

n_keep = 0
n_bucket = 0
n_kdf = 0
for sym, cands in per_sym.items():
    hf = [c for c in cands if c[0].startswith("hf_")]
    dl = [c for c in cands if not c[0].startswith("hf_")]
    dl.sort(key=lambda c: -abs(c[1]))
    dl = dl[:10]
    keep = hf + dl
    if not keep:
        print(f"{sym}: keep 为空! cands={len(cands)} hf={len(hf)} dl={len(dl)}")
        continue
    n_keep += 1
    kdf = kstore.load(sym, "1d")
    if kdf.empty:
        continue
    n_kdf += 1
    # 模拟 bucket
    bucket = {}
    ts_clean = kdf["ts"].values.astype(np.int64)
    for col, _r, fdf in keep:
        vals = fdf["factor"].values.astype(np.float64)
        if "ts" in fdf.columns:
            ts_arr = fdf["ts"].values.astype(np.int64)
            for t, v in zip(ts_arr, vals):
                if np.isfinite(v):
                    bucket.setdefault(int(t), {})[col] = float(v)
        else:
            n = min(len(vals), len(ts_clean))
            for t, v in zip(ts_clean[-n:], vals[-n:]):
                if np.isfinite(v):
                    bucket.setdefault(int(t), {})[col] = float(v)
    if bucket:
        n_bucket += 1

print(f"per_sym 股票数: {len(per_sym)}")
print(f"keep 非空: {n_keep}  有 K 线: {n_kdf}  bucket 非空: {n_bucket}")
# 样本 0 的真正检查点：第 2 段 for sym, bucket 循环的条件
# 用一个示例 sym 检查 label 逻辑
sym0 = next(iter(per_sym))
cands = per_sym[sym0]
hf = [c for c in cands if c[0].startswith("hf_")]
dl = [c for c in cands if not c[0].startswith("hf_")]
dl.sort(key=lambda c: -abs(c[1]))
dl = dl[:10]
print(f"\n示例 {sym0}: hf={len(hf)} dl={len(dl)} (Top-10 后)")
print(f"  dl 示例: {[(c[0], round(c[1],3)) for c in dl[:3]]}")
