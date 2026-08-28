"""临时验证：P2 修复配置（持有5日/Top20）下审计③是否显著。"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.portfolio_pipeline import build_panels
from scripts.audit_backtest import audit_random_returns

score, ret, _, _ = build_panels("store", horizon=5)
print(f"面板: {score.shape[1]} 只 x {score.shape[0]} 日")

# P2 修复配置：持有 5 日（匹配预测周期）+ Top20
r = audit_random_returns(score, ret, n_top=20, n_shuffle=50, seed=42, hold=5)
print(f"[持有5日/Top20] 通过={r['passed']}")
print(f"  原策略总收益 {r['real_total']:+.2%}, "
      f"shuffle {r['shuffle_mean']:+.2%}±{r['shuffle_std']:.2%}, "
      f"超随机分位 {r['pct_below_real']:.0%}")

# 对照：Top5 持有 5 日
r2 = audit_random_returns(score, ret, n_top=5, n_shuffle=50, seed=42, hold=5)
print(f"[持有5日/Top5 ] 通过={r2['passed']}")
print(f"  原策略总收益 {r2['real_total']:+.2%}, "
      f"超随机分位 {r2['pct_below_real']:.0%}")
