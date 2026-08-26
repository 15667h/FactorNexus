"""
scripts/factor_backtest.py — 已入库因子浏览 + 因子回测（终端工具）

功能：
  1. 因子库列表：列出 FactorStore 全部已入库因子（编号 / 品种 / 引擎 / 公式 /
     IC / DSR / 五维 / Sharpe / PBO / 换手 / 入库时间），按 DSR 排序
  2. 因子详细体质：单因子全字段（五维五分量、DSR、PBO、CPCV、IC/ICIR/rankIC、
     夏普/索提诺、最大回撤、换手、n_trials、公式描述、染色体）
  3. 因子回测：因子序列与 K 线尾部对齐 → tanh 仓位 × 未来收益 − 换手成本 →
     绩效指标（总收益/年化/夏普/索提诺/最大回撤/Calmar/盈亏比/胜率/换手/IC）+
     终端 ASCII 资金曲线

用法：
    python scripts/factor_backtest.py                # 交互模式（列表 → 编号选择 → 回测）
    python scripts/factor_backtest.py --list         # 只列因子库
    python scripts/factor_backtest.py --factor 3     # 查看第 3 个因子的详细体质
    python scripts/factor_backtest.py --backtest sh600519 2f013419a1cf
                                                     # 直接回测指定因子
    python scripts/factor_backtest.py --top 5 --backtest-all   # 回测 DSR 前 5
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402


# ── 因子库读取 ──────────────────────────────────────────────────────────────

def load_factors(store_dir: str | Path = "store",
                 sort_by: str = "sharpe") -> list[dict]:
    """读取已入库因子列表。

    排序（--sort）：
      sharpe : 按回测 Sharpe 降序（默认；直观绩效口径）
      ic     : 按横截面认证 RankIC 降序（机构预测力标准；无 cert 时回退源股票 rankIC）
      dsr    : 按 DSR 降序（Bailey 修正后真实因子普遍≈0，仅供查看）

    每条: {symbol, hash, kind, engine, formula, vocab_version,
           describe, ic, rankic, icir, dsr, pbo, cpcv, five_dim,
           sharpe, max_dd, turnover, n_trials, mined_at, path}
    """
    from data_pipeline.store.kline_store import FactorStore
    fs = FactorStore(store_dir)
    items = []
    for meta in fs.list_factors():
        rep = meta.get("report") or {}
        five = rep.get("five_dim") or {}
        engine = ((rep.get("meta") or {}).get("engine")) or "?"
        items.append({
            "symbol": str(meta.get("symbol", "?")),
            "hash": str(meta.get("hash", "?")),
            "kind": "token" if len(meta.get("formula") or []) <= 8
                    and (rep.get("meta") or {}).get("engine") == "rl"
                    else "param",
            "engine": engine,
            "formula": list(meta.get("formula") or []),
            "vocab_version": str(meta.get("vocab_version", "")),
            "describe": str(rep.get("describe", "")),
            "direction": float((rep.get("meta") or {}).get("direction", 0.0)),
            "oos_rankic": float((rep.get("meta") or {}).get("oos_rankic", 0.0)),
            "oos_t": float((rep.get("meta") or {}).get("oos_t", 0.0)),
            "cert_mode": str((rep.get("meta") or {}).get("cert_mode", "")),
            "cert_rankic": float((rep.get("meta") or {}).get("cert_rankic", 0.0)),
            "cert_p": float((rep.get("meta") or {}).get("cert_p", 1.0)),
            "cert_stocks": int((rep.get("meta") or {}).get("cert_stocks", 0)),
            "cert_days": int((rep.get("meta") or {}).get("cert_days", 0)),
            "ic_decay": list((rep.get("meta") or {}).get("ic_decay", []) or []),
            "group_monotonicity": float((rep.get("meta") or {}).get(
                "group_monotonicity", 0.0)),
            "long_short_ret": float((rep.get("meta") or {}).get(
                "long_short_ret", 0.0)),
            "ic": float(rep.get("ic", 0.0)),
            "rankic": float(rep.get("rankic", 0.0)),
            "icir": float(rep.get("icir", 0.0)),
            "dsr": float(rep.get("dsr", 0.0)),
            "pbo": float(rep.get("pbo", 0.5)),
            "cpcv": rep.get("cpcv") or {},
            "five_dim": five,
            "sharpe": float(rep.get("sharpe", 0.0)),
            "max_dd": float(rep.get("max_dd", 0.0)),
            "turnover": float(rep.get("turnover", 0.0)),
            "n_trials": int(rep.get("n_trials", 0)),
            "mined_at": float(rep.get("mined_at", 0.0)),
            "path": str(meta.get("path", "")),
        })
    if sort_by == "ic":
        # 机构预测力标准：横截面认证 RankIC（无 cert 时回退源股票 rankIC）
        items.sort(key=lambda f: -(f.get("cert_rankic", 0.0)
                                   if f.get("cert_mode") else f.get("rankic", 0.0)))
    elif sort_by == "dsr":
        items.sort(key=lambda f: -f["dsr"])
    else:
        items.sort(key=lambda f: -f["sharpe"])
    return items


def _fmt_time(ts: float) -> str:
    if not ts:
        return "?"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _kind_label(kind: str) -> str:
    return {"param": "参数公式", "token": "token公式"}.get(kind, kind)


# ── 终端渲染 ────────────────────────────────────────────────────────────────

def format_factor_list(factors: list[dict]) -> str:
    """因子库列表表格（编号/品种/hash/引擎/类型/截面RankIC/五维/Sharpe/公式/认证）。"""
    lines = []
    header = (f"{'#':>3}  {'symbol':<10} {'hash':<8} {'引擎':<4} {'类型':<8} "
              f"{'IC':>7} {'五维':>5} {'Sharpe':>7}  公式/认证信息")
    lines.append(header)
    lines.append("-" * len(header))
    for i, f in enumerate(factors, 1):
        five = f["five_dim"].get("total", 0.0)
        desc = f["describe"][:30] if f["describe"] else f"{f['hash']}"
        # IC 列优先显示横截面认证 RankIC（机构范式），否则源股票 OOS IC
        if f.get("cert_mode"):
            ic_disp = f["cert_rankic"]
            cert_txt = f"{f['cert_stocks']}只×{f['cert_days']}日"
        else:
            ic_disp = f["ic"]
            cert_txt = "单标的OOS"
        lines.append(
            f"{i:>3}  {f['symbol']:<10} {f['hash'][:8]:<8} {f['engine']:<4} "
            f"{_kind_label(f['kind']):<8} "
            f"{ic_disp:>7.3f} {five:>5.2f} {f['sharpe']:>7.2f}  "
            f"{desc} [{cert_txt}]")
    lines.append("")
    lines.append(f"共 {len(factors)} 条已入库因子（IC 列为横截面认证 RankIC）")
    return "\n".join(lines)


def format_factor_profile(f: dict) -> str:
    """单个因子的完整体质（详细字段）。"""
    five = f["five_dim"]
    lines = [
        f"══ 因子 {f['symbol']} @ {f['hash']} ══",
        f"引擎: {f['engine']}    类型: {_kind_label(f['kind'])}    "
        f"词表: {f['vocab_version']}",
        f"公式: {f['describe'] or '(无描述)'}",
        f"染色体: {f['formula']}",
        "",
        f"── 预测力 ──",
        f"IC={f['ic']:+.4f}   rankIC={f['rankic']:+.4f}   ICIR={f['icir']:+.3f}",
        "",
        f"── 样本外认证（OOS，机构标准）──",
    ]
    if f.get("cert_mode"):
        lines.append(
            f"── 横截面认证（机构范式，{f['cert_mode']}）──\n"
            f"截面RankIC={f['cert_rankic']:+.4f}   块自助p={f['cert_p']:.3f}   "
            f"认证股票={f['cert_stocks']}只 × {f['cert_days']}交易日")
    elif f.get("oos_t") or f.get("direction"):
        lines.append(
            f"OOS_RankIC={f['oos_rankic']:+.4f}   OOS_t={f['oos_t']:+.2f}   "
            f"方向={'+1(做多)' if f.get('direction', 1.0) >= 0 else '-1(反向翻转做多)'}")
    else:
        lines.append("（旧因子无认证记录；方向按全样本 IC 符号翻转）")
    lines += [
        "",
        f"── 显著性（多重检验控制）──",
        f"DSR={f['dsr']:.3f}   PBO={f['pbo']:.3f}   n_trials={f['n_trials']}",
    ]
    cpcv = f["cpcv"]
    if isinstance(cpcv, dict) and cpcv:
        lines.append(f"CPCV: {json_safe(cpcv)[:160]}")
    lines += [
        "",
        f"── 绩效（tanh 仓位）──",
        f"Sharpe={f['sharpe']:.2f}   最大回撤={f['max_dd']:.4f}   "
        f"换手={f['turnover']:.3f}",
    ]
    decay = f.get("ic_decay") or []
    if decay:
        lines += [
            "",
            "── IC 衰减（lag 1..10 期，机构 E3）──",
            "  " + " ".join(f"{i + 1}:{v:+.3f}" for i, v in enumerate(decay)),
        ]
    if f.get("group_monotonicity") or f.get("long_short_ret"):
        lines += [
            "",
            f"── 分层回测（十分组，机构 E2）──",
            f"Q1-Q10 单调性(Spearman)={f['group_monotonicity']:+.3f}   "
            f"多空收益={f['long_short_ret']:+.4f}",
        ]
    lines += [
        "",
        f"── AlphaEval 五维 ──",
        f"预测力(PPS)={five.get('pps', 0):.3f}   时间稳定性={five.get('stability', 0):.3f}",
        f"鲁棒性={five.get('robustness', 0):.3f}   金融逻辑={five.get('logic', 0):.3f}",
        f"多样性={five.get('diversity', 0):.3f}   综合={five.get('total', 0):.3f}",
        f"入库时间: {_fmt_time(f['mined_at'])}",
    ]
    return "\n".join(lines)


def json_safe(cpcv: dict) -> str:
    import json
    try:
        return json.dumps(cpcv, ensure_ascii=False)[:200]
    except Exception:  # noqa: BLE001
        return str(cpcv)[:200]


# ── 因子回测 ────────────────────────────────────────────────────────────────

def _rankdata(a: np.ndarray) -> np.ndarray:
    """排名（并列取平均，与 scipy.stats.rankdata 同口径，纯 numpy 实现）。"""
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # 并列平均：对相同值分组的 rank 取均值
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    avg = np.zeros(len(counts), dtype=np.float64)
    for i in range(len(counts)):
        avg[i] = ranks[inv == i].mean()
    return avg[inv]


def _ffill_price(a: np.ndarray) -> np.ndarray:
    """非正/非有限价格前值填充（首日异常用 1.0 占位，避免除零）。"""
    a = a.copy()
    last = None
    for i in range(len(a)):
        if np.isfinite(a[i]) and a[i] > 0:
            last = float(a[i])
        else:
            a[i] = last if last is not None else 1.0
    return a


def _build_ret_from_close(close: np.ndarray, horizon: int) -> np.ndarray:
    """未来收益标签（与挖掘口径一致）：ret[t] = close[t+h]/close[t] - 1。"""
    T = len(close)
    ret = np.zeros(T)
    if T > horizon:
        ret[:T - horizon] = close[horizon:] / close[:-horizon] - 1.0
    return ret


def backtest_factor(factor: np.ndarray, close: np.ndarray,
                    horizon: int = 5, cost: float = 0.0003,
                    ppy: int = 244, direction: float = 1.0,
                    limit_filter: bool = True, slippage: float = 0.0,
                    limit_pct: float = 0.099) -> dict:
    """因子回测：因子与 K 线尾部对齐 → tanh 仓位 × 未来收益 − 换手成本 − 滑点。

    direction: 因子方向（-1 = 负 IC 反向因子，仓位翻转后做多）。
               机构标准：负 IC 因子信号翻转（SignalExporter 同口径）。
    limit_filter: 涨跌停不可成交限制（机构 B3，Qlib 中国模式 limit=9.9%）：
               涨停日（≥limit_pct）无法买入（多头仓位置 0）、跌停日（≤-limit_pct）
               无法卖出。limit_pct 按板块：主板 0.099，创业板/科创板 0.199。
    slippage: 单边滑点（额外成本，机构标准：成本 = 佣金 + 冲击成本）。
              每次建仓/平仓额外扣 slippage（比 cost 大一个量级，如 0.0005）。
    跳变日防御：|1日收益| > 21%（A股涨跌幅上限 + 容差）视为跨源混库/复权瑕疵
              （实测茅台 -85%/+590% 交替），该日收益置 0 且建仓日跳过（不交易、
              不支付换手成本），与挖掘侧 quality 清洗口径一致。

    Returns:
        {n, total_ret, annual_ret, annual_vol, sharpe, sortino, max_dd,
         calmar, profit_factor, win_rate, turnover, ic, icir, rankic, nav}
    """
    factor = np.asarray(factor, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    Lf = len(factor)
    if Lf < 30:
        raise ValueError(f"因子序列过短: {Lf}")
    if len(close) < Lf:
        raise ValueError(f"K线不足: {len(close)} < 因子 {Lf}")
    # 对齐：因子（入库时为 K 线尾部窗口计算）→ 取 K 线尾部 Lf 根
    close = close[-Lf:]
    # 数据防御：非正/非有限价格前值填充（停牌 0 价格等瑕疵，否则收益/净值爆掉）
    bad = ~np.isfinite(close) | (close <= 0)
    if bad.any():
        close = _ffill_price(close)
    ret = _build_ret_from_close(close, horizon)

    # 仓位 = tanh(因子)：入库因子已是因果 expanding zscore 序列（均值≈0、std≈1）。
    # 注意：不再做全样本 zscore（会引入双重标准化 + 全样本统计量的前视偏差）
    f = factor[-Lf:]
    pos = np.tanh(f) * float(direction)   # 负 IC 因子方向翻转（机构标准）

    # 执行时点（机构标准）：t 收盘产生信号 → t+1 才可成交（避免收盘价即时成交的前视）。
    # 持有期 = 预测期（horizon 根），非重叠调仓：每 horizon 根建仓一次、持有期间
    # 仓位固定，每日按 1 日收益 mark-to-market。
    # 历史 bug（2026-08-26 审计）：pnl[t] = pos[t-1]·ret5[t] 把未来 5 日收益在
    # 当天全部兑现且每日重建仓 → 时间重叠使收益虚高 ~5 倍并复利爆炸（曾出现 7e35%）。
    pos_exec = np.roll(pos, 1)
    pos_exec[0] = 0.0

    # 每日 1 日收益（mark-to-market；IC 仍用 h 日预测标签）
    ret1d = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    # 跳变日（混库/复权瑕疵）：收益置 0 + 建仓冻结（见下）
    jump_day = np.abs(ret1d) > 0.21
    ret1d = np.where(jump_day, 0.0, ret1d)

    # 涨跌停不可成交（机构 B3，Qlib 中国模式 limit_threshold）：
    # 涨停日无法买入（多头仓位置 0）、跌停日无法卖出（空头仓位置 0）；
    # 跳变日不参与涨跌停判定（价格本身不可信）
    if limit_filter and Lf > 1:
        r1 = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
        limit_up = (r1 >= limit_pct) & ~jump_day
        limit_down = (r1 <= -limit_pct) & ~jump_day
        pos_exec = pos_exec.copy()
        pos_exec[limit_up & (pos_exec > 0)] = 0.0
        pos_exec[limit_down & (pos_exec < 0)] = 0.0

    hold = max(int(horizon), 1)
    pnl = np.zeros(Lf)
    turnover_cost = np.zeros(Lf)
    slippage_cost = np.zeros(Lf)
    prev_pos = 0.0
    for e in range(1, Lf, hold):          # 非重叠建仓日
        seg_end = min(e + hold, Lf)
        if jump_day[e]:
            # 跳变日建仓冻结：不交易、保持原仓位、不付成本（价格不可信）
            w = prev_pos
        else:
            w = pos_exec[e]               # 建仓日执行的仓位（信号来自 e-1 收盘）
        pnl[e:seg_end] = w * ret1d[e:seg_end]
        turnover_cost[e] = abs(w - prev_pos) * cost
        slippage_cost[e] = abs(w - prev_pos) * slippage
        prev_pos = w
    pnl = pnl - turnover_cost - slippage_cost
    nav = np.cumprod(1.0 + pnl)

    # 绩效指标
    n = len(pnl)
    total_ret = float(nav[-1] - 1.0)
    daily = pnl[1:]  # 去掉首日（pos 从 0 起步的伪换手）
    mean_d, std_d = float(daily.mean()), float(daily.std())
    annual_ret = float((1.0 + mean_d) ** ppy - 1.0) if mean_d > -1 else -1.0
    annual_vol = float(std_d * math.sqrt(ppy))
    sharpe = float(mean_d / std_d * math.sqrt(ppy)) if std_d > 1e-12 else 0.0
    downside = daily[daily < 0]
    sortino = float(mean_d / (downside.std() + 1e-12) * math.sqrt(ppy)) \
        if downside.size > 1 else 0.0
    peak = np.maximum.accumulate(nav)
    max_dd = float(((peak - nav) / peak).max()) if peak[-1] > 0 else 0.0
    calmar = float(annual_ret / (max_dd + 1e-12)) if max_dd > 1e-9 else 0.0
    pos_sum = daily[daily > 0].sum()
    neg_sum = abs(daily[daily < 0].sum())
    profit_factor = float(pos_sum / neg_sum) if neg_sum > 1e-12 else float("inf")
    win_rate = float((daily > 0).mean())
    turnover = float(np.abs(np.diff(pos)).mean())

    # 相关性（全样本，与报告口径近似）
    ic = float(np.corrcoef(f, ret)[0, 1]) if f.std() > 1e-9 else 0.0
    icir = float(ic / (np.std(ret) + 1e-9)) if n > 2 else 0.0
    rankic = float(np.corrcoef(_rankdata(f), _rankdata(ret))[0, 1]) \
        if f.std() > 1e-9 else 0.0

    return {
        "n": n,
        "total_ret": total_ret,
        "annual_ret": annual_ret,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "turnover": turnover,
        "ic": ic,
        "icir": icir,
        "rankic": rankic,
        "nav": nav,
        "horizon": horizon,
        "cost": cost,
        "slippage": slippage,
        "limit_pct": limit_pct,
        "jump_days": int(jump_day.sum()),
    }


def ascii_equity_curve(nav: np.ndarray, width: int = 64, height: int = 16) -> str:
    """净值序列 → ASCII 资金曲线（终端文本图）。"""
    nav = np.asarray(nav, dtype=np.float64)
    n = len(nav)
    if n < 2:
        return "(数据不足，无法绘制)"
    lo, hi = float(nav.min()), float(nav.max())
    span = (hi - lo) or 1.0
    idx = np.linspace(0, n - 1, width).astype(int)
    rows: list[str] = []
    for r in range(height, 0, -1):
        level = lo + span * (r - 0.5) / height
        line = "".join("█" if nav[i] >= level else " " for i in idx)
        rows.append(f"{level:>10.3f} |{line}")
    axis = " " * 10 + "+" + "-" * width
    rows.append(axis)
    # x 轴刻度（首/中/尾）
    labels = f"{'首':>6} {nav[0]:.3f}     {'中':>4} {nav[n // 2]:.3f}     {'尾':>4} {nav[-1]:.3f}"
    rows.append(labels)
    return "\n".join(rows)


def format_backtest_report(b: dict) -> str:
    """回测绩效报告文本。"""
    extra = ""
    if b.get("slippage"):
        extra += f"   滑点={b['slippage']:.4f}"
    if b.get("jump_days"):
        extra += f"   跳变防御={b['jump_days']}日"
    return (
        f"── 回测绩效（窗口 {b['n']} 根，horizon={b['horizon']}，"
        f"单边成本={b['cost']:.4f}，涨跌停阈值={b.get('limit_pct', 0.099):.1%}）──\n"
        f"总收益={b['total_ret']:+.2%}   年化收益={b['annual_ret']:+.2%}   "
        f"年化波动={b['annual_vol']:.2%}\n"
        f"夏普={b['sharpe']:.2f}   索提诺={b['sortino']:.2f}   "
        f"最大回撤={b['max_dd']:.2%}   Calmar={b['calmar']:.2f}\n"
        f"盈亏比={b['profit_factor']:.2f}   胜率={b['win_rate']:.1%}   "
        f"换手={b['turnover']:.3f}{extra}\n"
        f"IC={b['ic']:+.4f}   ICIR={b['icir']:+.3f}   rankIC={b['rankic']:+.4f}"
    )


def run_factor_backtest(f: dict, store_dir: str | Path = "store",
                        horizon: int = 5, cost: float = 0.0003,
                        limit_filter: bool = True,
                        use_cert_direction: bool = False,
                        slippage: float = 0.0) -> dict:
    """加载因子序列 + K 线 → 回测。

    方向语义（2026-08-26 审计修复）：
      - 单标的回测 = 该因子在这只股票上的实际可用方向 → 默认按回测段
        实际 IC 符号翻转（个体信号方向，SignalExporter 同口径）。
      - 防前视（2026-08-26 强化）：方向只用回测段前一半估计（全样本 IC
        符号会混入后半段信息，方向选择本身即前视）。
      - 入库 direction（横截面认证 RankIC 符号）是"组合层普适方向"，
        单标的个体方向可能与截面方向相反；--use-cert-direction 可选使用。

    板块感知涨跌停（机构 B3）：创业板（sz30）/科创板（sh688）涨跌幅上限
    20%，主板 10%。旧实现统一 9.9% 会把创业板正常 15% 涨幅误判为涨停
    （无法买入）→ 系统性低估；现按代码前缀取 19.9% / 9.9%。
    """
    from data_pipeline.store.kline_store import FactorStore, KlineStore

    fdf = FactorStore(store_dir).load(f["symbol"], f["hash"])
    if fdf is None or "factor" not in fdf.columns or fdf.empty:
        raise FileNotFoundError(f"因子文件缺失: {f['symbol']}_{f['hash']}")
    factor = np.asarray(fdf["factor"].values, dtype=np.float64)

    df = KlineStore(store_dir).load(f["symbol"], "1d")
    if df.empty:
        raise FileNotFoundError(f"K线库无 {f['symbol']}_1d")
    close = df["close"].values.astype(np.float64)

    # 板块感知涨跌停阈值（创业板/科创板 20%，其余主板 10%）
    sym = str(f["symbol"])
    limit_pct = 0.199 if sym.startswith(("sh688", "sz30")) else 0.099

    # 方向：默认按本标的实际 IC 符号（单标的回测语义；仅用前半段防前视）
    if use_cert_direction and float(f.get("direction", 0.0)) != 0.0:
        direction = float(f["direction"])
    else:
        ret = _build_ret_from_close(close, horizon)
        n = min(len(factor), len(ret))
        x, y = factor[-n:], ret[-n:]
        half = n // 2
        x, y = x[:half], y[:half]          # 防前视：只允许用前半段定方向
        x, y = x - x.mean(), y - y.mean()
        sd = x.std() * y.std()
        ic_sign = np.sign((x * y).mean() / sd) if sd > 1e-12 else 1.0
        direction = 1.0 if ic_sign >= 0 else -1.0
    return backtest_factor(factor, close, horizon=horizon, cost=cost,
                           direction=direction, limit_filter=limit_filter,
                           slippage=slippage, limit_pct=limit_pct)


def resolve_selection(cmd: str, factors: list[dict]) -> tuple[dict | None, str | None]:
    """解析用户选择：编号（如 3）或 symbol+hash（如 sh600519 2f01）。

    Returns:
        (factor, None) 命中；或 (None, 错误提示)。
    """
    parts = cmd.split()
    if len(parts) == 1 and parts[0].isdigit():
        n = int(parts[0])
        if 1 <= n <= len(factors):
            return factors[n - 1], None
        return None, f"编号 {n} 超出范围（1-{len(factors)}）"
    if len(parts) == 2:
        matches = [f for f in factors
                   if f["symbol"] == parts[0] and f["hash"].startswith(parts[1])]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, f"命中 {len(matches)} 条，请用完整 hash"
        return None, f"未找到 {parts[0]} {parts[1]}"
    return None, "请输入编号或 symbol hash"


def resolve_backtest_selector(symbol: str, hash_prefix: str | None,
                              factors: list[dict]) -> tuple[dict | None, str | None]:
    """解析回测选择器（用户友好的因子定位方式）。

    - 纯数字（如 "5"）      → 列表编号（推荐，对应 --list 的第 N 个因子）
    - 只给 symbol（如 sh600519）→ 该品种唯一因子直接回测；多个时报错并列出编号
    - symbol + hash 前缀    → 精确匹配（次要；hash 可从 --list 复制）

    Returns: (factor, None) 或 (None, 错误提示)
    """
    if symbol.isdigit():
        n = int(symbol)
        if 1 <= n <= len(factors):
            return factors[n - 1], None
        return None, f"编号 {symbol} 超出范围（1-{len(factors)}），先用 --list 查看"
    matches = [f for f in factors if f["symbol"] == symbol]
    if not matches:
        return None, f"因子库中无 {symbol} 的因子（先用 --list 查看可用品种）"
    if hash_prefix:
        sub = [f for f in matches if f["hash"].startswith(hash_prefix)]
        if len(sub) == 1:
            return sub[0], None
        if len(sub) > 1:
            return None, f"hash 前缀 {hash_prefix} 命中 {len(sub)} 条，请用完整 hash 或编号"
        return None, f"{symbol} 下无 hash 前缀 {hash_prefix} 的因子（hash 见 --list）"
    if len(matches) == 1:
        return matches[0], None
    idx = [str(i) for i, f in enumerate(factors, 1) if f["symbol"] == symbol]
    return None, (f"{symbol} 有 {len(matches)} 个因子（编号: {', '.join(idx)}），"
                  f"请用 --backtest <编号> 选择，或追加 hash 前缀")


# ── 交互模式 ────────────────────────────────────────────────────────────────

def interactive(store_dir: str, horizon: int, cost: float,
                limit_filter: bool = True, slippage: float = 0.0) -> None:
    """交互式：列表 → 输入编号回测 → 回车继续 / q 退出。"""
    factors = load_factors(store_dir)
    if not factors:
        print("[因子库] 暂无已入库因子，先运行 python scripts/mine_full_market.py "
              "挖掘入库")
        return

    print(format_factor_list(factors))
    print()
    while True:
        try:
            cmd = input(
                "输入因子编号回测（如 3；也可输入 symbol hash；? 帮助；q 退出）: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出]")
            break
        if not cmd:
            continue
        if cmd.lower() == "q":
            print("[退出]")
            break
        if cmd == "?":
            print("  - 编号（1-N）：回测对应因子（推荐）\n"
                  "  - symbol：该品种唯一因子直接回测（如 sh600519）\n"
                  "  - symbol hash：精确指定（hash 见列表，如 sh600519 2f0134）\n"
                  "  - l：重新显示因子列表\n"
                  "  - q：退出")
            continue
        if cmd.lower() == "l":
            print(format_factor_list(factors))
            continue

        sel, err = resolve_selection(cmd, factors)
        if err:
            print(f"[无效] {err}")
            continue

        print()
        print(format_factor_profile(sel))
        try:
            bt = run_factor_backtest(sel, store_dir, horizon=horizon, cost=cost,
                                     limit_filter=limit_filter,
                                     slippage=slippage)
            print()
            print(format_backtest_report(bt))
            print()
            print(ascii_equity_curve(bt["nav"]))
        except Exception as exc:  # noqa: BLE001
            print(f"[回测失败] {type(exc).__name__}: {exc}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="已入库因子浏览 + 因子回测（终端工具）")
    ap.add_argument("--list", action="store_true", help="只显示因子库列表")
    ap.add_argument("--factor", type=int, default=0,
                    help="查看第 N 个因子的详细体质（不回测）")
    ap.add_argument("--backtest", nargs="+", metavar="选择器",
                    help="回测因子：编号（如 5）/ symbol（如 sh600519）/ "
                         "symbol+hash前缀（如 sh600519 2f0134）")
    ap.add_argument("--top", type=int, default=0,
                    help="只显示前 N 个因子（按 --sort 排序后）")
    ap.add_argument("--sort", default="sharpe", choices=["sharpe", "ic", "dsr"],
                    help="列表排序：sharpe=回测夏普降序(默认) / ic=横截面认证RankIC "
                         "(机构预测力标准) / dsr=DSR 降序")
    ap.add_argument("--backtest-all", action="store_true",
                    help="配合 --top：对前 N 个因子逐一回测")
    ap.add_argument("--horizon", type=int, default=5, help="收益预测周期（天）")
    ap.add_argument("--cost", type=float, default=0.0003, help="单边换手成本")
    ap.add_argument("--slippage", type=float, default=0.0,
                    help="单边滑点（机构标准建议 0.0005，成本=佣金+冲击）")
    ap.add_argument("--no-limit-filter", action="store_true",
                    help="关闭涨跌停不可成交限制（默认开启，机构 B3）")
    ap.add_argument("--use-cert-direction", action="store_true",
                    help="回测按入库截面方向翻转（默认按本标的实际 IC 符号翻转）")
    ap.add_argument("--store-dir", default="store", help="存储根目录")
    args = ap.parse_args()

    factors = load_factors(args.store_dir, sort_by=args.sort)
    if not factors:
        print("[因子库] 暂无已入库因子，先运行 python scripts/mine_full_market.py "
              "挖掘入库")
        return
    if args.top > 0:
        factors = factors[:args.top]

    if args.list:
        print(format_factor_list(factors))
        return
    if args.factor > 0:
        if not 1 <= args.factor <= len(factors):
            ap.error(f"编号超出范围（1-{len(factors)}）")
        print(format_factor_profile(factors[args.factor - 1]))
        return
    if args.backtest:
        sel = args.backtest
        if len(sel) == 1:
            f, err = resolve_backtest_selector(sel[0], None, factors)
        elif len(sel) == 2:
            f, err = resolve_backtest_selector(sel[0], sel[1], factors)
        else:
            err = "--backtest 接受 1-2 个参数：编号 | symbol | symbol hash"
            f = None
        if err:
            print(f"[错误] {err}")
            print("提示：先运行 --list 查看因子编号，再 --backtest <编号>")
            sys.exit(2)
        print(format_factor_profile(f))
        bt = run_factor_backtest(f, args.store_dir,
                                 horizon=args.horizon, cost=args.cost,
                                 limit_filter=not args.no_limit_filter,
                                 use_cert_direction=args.use_cert_direction,
                                 slippage=args.slippage)
        print()
        print(format_backtest_report(bt))
        print()
        print(ascii_equity_curve(bt["nav"]))
        return
    if args.backtest_all:
        for i, f in enumerate(factors, 1):
            print(f"\n######## 因子 {i}/{len(factors)}: {f['symbol']} {f['hash']} "
                  f"({f['engine']}) ########")
            print(format_factor_profile(f))
            try:
                bt = run_factor_backtest(f, args.store_dir,
                                         horizon=args.horizon, cost=args.cost,
                                         limit_filter=not args.no_limit_filter,
                                         use_cert_direction=args.use_cert_direction,
                                         slippage=args.slippage)
                print()
                print(format_backtest_report(bt))
            except Exception as exc:  # noqa: BLE001
                print(f"[回测失败] {type(exc).__name__}: {exc}")
        return

    # 默认：交互模式
    print(f"[因子库] {len(factors)} 条已入库因子（store-dir={args.store_dir}）\n")
    interactive(args.store_dir, args.horizon, args.cost,
                limit_filter=not args.no_limit_filter,
                slippage=args.slippage)


if __name__ == "__main__":
    main()
