"""验证特征选择：全量 vs Top-K 的列数与耗时。"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import time
from model_core.strategy_factory.dataset import build_dataset

t0 = time.time()
ds_all = build_dataset("store", horizon=5)
print(f"全量: 样本={ds_all.n_samples} 特征列={len(ds_all.feature_names)} "
      f"股票={ds_all.meta['n_symbols']} 耗时={time.time()-t0:.0f}s")
hf_cols = [c for c in ds_all.feature_names if c.startswith("hf_")]
print(f"高频共享列: {len(hf_cols)} 个（{hf_cols[:6]}...）")

t0 = time.time()
ds_k = build_dataset("store", horizon=5, top_factors_per_stock=10)
print(f"Top-10: 样本={ds_k.n_samples} 特征列={len(ds_k.feature_names)} "
      f"耗时={time.time()-t0:.0f}s")

# 稀疏度对比
import numpy as np
for name, ds in (("全量", ds_all), ("Top-10", ds_k)):
    if ds.n_samples == 0:
        continue
    X = ds.X.values
    sparsity = float(np.isnan(X).mean())
    print(f"{name} 稀疏度: {sparsity:.1%}（每行有效特征 ~{int(X.shape[1]*(1-sparsity))} 个）")
