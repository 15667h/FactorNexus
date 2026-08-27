"""
scripts/factor_monitor.py — 因子监控终端工具（P17）

全库因子生命周期监控：IC 衰减轨迹、方向稳定性、失效预警。

用法：
    python scripts/factor_monitor.py               # 全库监控报告
    python scripts/factor_monitor.py --recent 60   # 实时段 60 根
    python scripts/factor_monitor.py --horizon 10  # 10 日预测周期
    python scripts/factor_monitor.py --detail      # 打印每个因子详情
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model_core.eval.factor_monitor import (  # noqa: E402
    monitor_all, save_monitor_report,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="因子监控：IC 衰减/方向/失效预警")
    ap.add_argument("--horizon", type=int, default=5, help="收益预测周期")
    ap.add_argument("--recent", type=int, default=120, help="实时段长度（根）")
    ap.add_argument("--detail", action="store_true", help="打印每个因子详情")
    ap.add_argument("--store-dir", default="store", help="存储根目录")
    args = ap.parse_args()

    rep = monitor_all(args.store_dir, horizon=args.horizon, recent=args.recent)
    print("═" * 62)
    print("  FactorNexus · 因子监控（P17）")
    print("═" * 62)
    print(f"因子总数: {rep['total']}  "
          f"健康: {rep['healthy']}  "
          f"预警: {rep['warning']}  "
          f"数据不足: {rep['stale']}  "
          f"错误: {rep['error']}")
    if rep["by_engine"]:
        parts = [f"{k}: {v['warning']}/{v['total']} 预警"
                 for k, v in sorted(rep["by_engine"].items())]
        print("按引擎:", "  ".join(parts))
    if rep["warnings"]:
        print("\n── 预警清单 ──")
        for w in rep["warnings"]:
            print(f"⚠ {w['symbol']} {w['hash'][:8]} "
                  f"({w.get('engine', '?')}/{w.get('kind', '?')}) "
                  f"认证RankIC={w['cert_rankic']:+.4f} "
                  f"实时RankIC={w['recent_rankic']:+.4f} p={w['recent_p']:.3f}")
            for a in w["alerts"]:
                print(f"    · {a}")
    if args.detail:
        print("\n── 全部因子详情 ──")
        from model_core.eval.factor_monitor import monitor_factor
        from data_pipeline.store.kline_store import FactorStore
        for f in FactorStore(args.store_dir).list_factors():
            d = monitor_factor(f["symbol"], f["hash"], args.store_dir,
                               horizon=args.horizon, recent=args.recent)
            track = d.get("decay_track") or []
            track_s = " ".join(f"{x:+.3f}" for x in track[-6:]) \
                if track else "—"
            print(f"{d['status']:8s} {d['symbol']} {d['hash'][:8]} "
                  f"认证={d['cert_rankic']:+.4f} 实时={d['recent_rankic']:+.4f} "
                  f"轨迹[{track_s}]")
    save_monitor_report(rep, Path(args.store_dir) / "meta" / "factor_monitor.json")
    print(f"\n[快照] → {args.store_dir}/meta/factor_monitor.json")


if __name__ == "__main__":
    main()
