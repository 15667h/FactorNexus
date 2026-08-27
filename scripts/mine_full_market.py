"""
scripts/mine_full_market.py — 全市场三引擎联动挖矿机（P6 核心）

把「全市场 5000+ 只 A股」作为一条矿脉：每次运行本文件，GP / RL / LLM
三个因子挖掘模型一起挖、互相联动（不再相互隔离），数据量大管饱，CUDA GPU 加速。

联动机制（三个模型互相喂知识，全部在本文件内闭环）：
  [联动①  LLM → GP 种子注入]  每批（--llm-batch 只）运行一次 LLM 多智能体挖掘，
                              产出的公式染色体注入该批每只股票的 GP 初始种群
                              （NSGA3FactorMiner 新增 init_chroms 支持），GP 从
                              LLM 的思路出发继续进化，而不是从零随机。
  [联动②  GP → LLM 发现日志]  GP 每批挖出的精英因子写入共享矿池；下一批 LLM 的
                              假设由「矿池发现日志」驱动（"参考 XX 已发现的因子
                              …改进其参数"），且矿池公式作为新颖性约束库，
                              防止 LLM 重复发明。
  [联动③  RL 跨品种精英迁移]  矿池中历史 RL 精英（token 公式）在每只股票训练前
                              预热进 AlphaEngine 精英回放池，RL 继承全市场知识；
                              RL 新发现继续写回矿池，供后续股票与 GP/LLM 使用。
  [联动④  三引擎候选统一裁决] 每只股票 GP/RL/LLM 的全部候选统一做五维 + DSR 评估，
                              批内相关性 >0.95 的拥挤因子丢弃（只留 DSR 高者），
                              达标因子写入 FactorStore 并回灌矿池。

数据量大管饱：
  - 股票池 = 新浪 Market_Center hs_a 节点全 A 股清单（沪深 ~5300 只，缓存本地）
  - 每只股票全历史日线（腾讯 fqkline 自动翻页），无本地数据自动并行回填
  - --bars 控制挖掘窗口（默认 2000 根 ≈ 8 年日线；0 = 全历史）

CUDA GPU：
  - 自动检测 CUDA（torch.cuda.is_available），RL 引擎（AlphaEngine/AlphaGPT）
    跑在 GPU 上；--device cpu 可强制回退。

用法：
    python scripts/mine_full_market.py                     # 全市场三引擎联动挖掘
    python scripts/mine_full_market.py --limit 5           # 先试 5 只
    python scripts/mine_full_market.py --skip-done         # 断点续跑（默认开启见 --help）
    python scripts/mine_full_market.py --engines gp,llm    # 只开 GP+LLM（快速模式）
    python scripts/mine_full_market.py --quick-gate 0.03   # 预选过滤（省时）

输出：
    store/meta/market_pool.json            联动矿池（三引擎知识交换中枢）
    store/meta/full_market_report.csv      全市场逐股三引擎挖掘汇总
    store/meta/full_market_progress.json   断点进度
    store/factors/…                        达标因子库（复用现有 FactorStore）
"""
from __future__ import annotations

import argparse
import csv
import heapq
import io
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_pipeline.store.kline_store import KlineStore, FactorStore  # noqa: E402

# 默认工作目录 = 项目根（KlineStore/FactorStore 默认 store/ 均相对根目录）
_PROJECT_ROOT = ROOT

# ── 联动矿池（MarketPool）：三引擎知识交换中枢 ────────────────────────────────

_POOL_FILE = "meta/market_pool.json"
_POOL_MAX = 2000          # 矿池容量上限（超出按 DSR 淘汰尾部）
_RL_ELITE_PREHEAT = 20    # RL 精英回放预热条数上限


def _pool_key(kind: str, formula: list) -> str:
    return f"{kind}:" + "-".join(str(int(t)) for t in formula)


class MarketPool:
    """共享矿池：三引擎产出的因子/公式在此去重、排序、互相引用。

    结构: {key: {key, kind(param|token), engine(gp|rl|llm), symbol,
                 formula:[int], desc:str, ic:float, dsr:float, total:float,
                 first_seen:str, last_seen:str, hits:int}}
    持久化: store/meta/market_pool.json（原子写 + 线程锁）
    """

    def __init__(self, store_dir: str | Path = "store", max_size: int = _POOL_MAX) -> None:
        self.root = Path(store_dir)
        self.path = self.root / _POOL_FILE
        self.max_size = max_size
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._load()

    # ── 持久化 ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._entries = data
            except (json.JSONDecodeError, OSError):
                self._entries = {}

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._entries, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)

    # ── 读写 ────────────────────────────────────────────────────────────

    def add(self, kind: str, engine: str, symbol: str, formula: list,
            desc: str, ic: float, dsr: float, total: float) -> str:
        """写入/更新一条矿池记录（按公式去重，命中次数+1）。返回 key。"""
        key = _pool_key(kind, formula)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            old = self._entries.get(key)
            if old is not None:
                old.update({
                    "engine": engine, "symbol": symbol, "desc": desc,
                    "ic": float(ic), "dsr": float(dsr), "total": float(total),
                    "last_seen": now, "hits": int(old.get("hits", 1)) + 1,
                })
            else:
                self._entries[key] = {
                    "key": key, "kind": kind, "engine": engine, "symbol": symbol,
                    "formula": [int(t) for t in formula], "desc": desc,
                    "ic": float(ic), "dsr": float(dsr), "total": float(total),
                    "first_seen": now, "last_seen": now, "hits": 1,
                }
            self._trim_locked()
        return key

    def _trim_locked(self) -> None:
        """容量控制：超出上限按 (dsr 降序, hits 降序) 淘汰尾部。"""
        if len(self._entries) <= self.max_size:
            return
        ranked = sorted(
            self._entries.values(),
            key=lambda e: (float(e.get("dsr", 0.0)), int(e.get("hits", 0))),
            reverse=True,
        )
        keep = {e["key"]: e for e in ranked[:self.max_size]}
        self._entries = keep

    def top(self, k: int = 10, kind: str | None = None) -> list[dict]:
        """按 (dsr, hits) 取矿池头部（联动引用用）。"""
        with self._lock:
            items = list(self._entries.values())
        if kind:
            items = [e for e in items if e.get("kind") == kind]
        items.sort(key=lambda e: (float(e.get("dsr", 0.0)), int(e.get("hits", 0))),
                   reverse=True)
        return items[:k]

    def param_formulas(self, k: int = 30) -> list:
        """矿池中 param 公式（转 ParamFormula），供 LLM 新颖性约束与 GP 种子。"""
        from model_core.formula_dsl import chrom_to_formula
        out = []
        for e in self.top(k, kind="param"):
            try:
                out.append(chrom_to_formula(e["formula"]))
            except Exception:  # noqa: BLE001
                continue
        return out

    def token_formulas(self, k: int = _RL_ELITE_PREHEAT) -> list[tuple[float, list[int]]]:
        """矿池中 token 公式（供 RL 精英回放预热）。返回 [(dsr, tokens)]。"""
        return [(float(e.get("dsr", 0.0)), [int(t) for t in e["formula"]])
                for e in self.top(k, kind="token")]

    def hypotheses_for_llm(self, k: int = 5) -> list[str]:
        """联动②：把矿池最近发现的精英因子改写成 LLM 假设（GP → LLM 反馈）。"""
        out = []
        for e in self.top(k, kind="param"):
            desc = str(e.get("desc", ""))[:40] or "因子"
            out.append(
                f"参考 {e.get('symbol', '?')} 已发现的因子 [{desc}] "
                f"(IC={float(e.get('ic', 0)):.3f} DSR={float(e.get('dsr', 0)):.2f})，"
                "改进其参数或结构以适配本股票，不要完全重复"
            )
        return out

    def size(self) -> int:
        with self._lock:
            return len(self._entries)


# ── 全 A 股清单（新浪 hs_a 节点）────────────────────────────────────────────

_UNIVERSE_CACHE = "meta/a_share_universe.json"
_MIN_UNIVERSE = 3000  # 全 A 股清单最低数量守卫（低于此值视为拉取异常）
_HS_A_API = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/"
             "json_v2.php/Market_Center.getHQNodeData")
_PAGE_SIZE = 100
_HTTP_TIMEOUT = 15.0


def fetch_universe(store_dir: str | Path = "store", refresh: bool = False) -> list[str]:
    """全 A 股清单：新浪 Market_Center node=hs_a 分页拉取（沪深，过滤北交所 bj）。

    缓存 store/meta/a_share_universe.json。返回 [sh600000, sz000001, ...]。
    """
    root = Path(store_dir)
    cache = root / _UNIVERSE_CACHE
    if not refresh and cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) >= _MIN_UNIVERSE:
                return [str(s) for s in data]
        except (json.JSONDecodeError, OSError):
            pass

    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://finance.sina.com.cn",
    })
    symbols: list[str] = []
    page = 1
    while True:
        resp = session.get(_HS_A_API, params={"page": page, "num": _PAGE_SIZE,
                                              "sort": "symbol", "asc": 1,
                                              "node": "hs_a"}, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            break
        for r in rows:
            sym = str(r.get("symbol", "")).strip()
            # 只保留沪深 A 股（sh/sz），过滤北交所 bj（腾讯/新浪 A股通道不支持）
            if sym.startswith(("sh", "sz")):
                symbols.append(sym)
        if len(rows) < _PAGE_SIZE:
            break
        page += 1
        time.sleep(0.25)  # 礼貌限速

    if len(symbols) < _MIN_UNIVERSE:
        raise RuntimeError(f"全 A 股清单拉取异常：仅 {len(symbols)} 只（<{_MIN_UNIVERSE}）")
    seen: set[str] = set()
    dedup = [s for s in symbols if not (s in seen or seen.add(s))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dedup, ensure_ascii=False), encoding="utf-8")
    tmp.replace(cache)
    return dedup


def _resolve_universe(symbols_file: str, limit: int, store_dir: str,
                      refresh_universe: bool) -> list[str]:
    """解析矿脉清单：--symbols-file > 全市场（新浪 hs_a）。"""
    if symbols_file:
        p = Path(symbols_file)
        if not p.exists():
            raise FileNotFoundError(f"股票清单文件不存在: {p}")
        out = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        out = fetch_universe(store_dir, refresh=refresh_universe)
    if limit > 0:
        out = out[:limit]
    return out


# ── token 公式可读描述（RL 产物）─────────────────────────────────────────────

def _decode_tokens(tokens: list[int]) -> str:
    try:
        from model_core.vocab import FORMULA_VOCAB
        names = FORMULA_VOCAB.token_names
        return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}"
                           for t in tokens)
    except Exception:  # noqa: BLE001
        return f"token{len(tokens)}"


def _oos_significance(f_os: np.ndarray, r_os: np.ndarray,
                      block: int = 20, n_boot: int = 500) -> tuple[float, float, float]:
    """OOS 段显著性：整体 Spearman + 块自助检验（机构时间序列显著性标准）。

    为什么不用滚动窗口秩相关：滚动 20 根窗口内的秩相关对"单调因子 × 短窗口趋势"
    极度敏感（窗口内样本非独立），会把价格水平类因子伪高估到 |RankIC| 0.5+。
    块自助保留序列自相关结构，p 值直接给出显著性。

    Returns: (rankic, t, p) —— t = real / bootstrap 标准误
    """
    from scipy.stats import spearmanr

    def _sr(a: np.ndarray, b: np.ndarray) -> float:
        r = spearmanr(a, b)
        stat = r.statistic if hasattr(r, "statistic") else r[0]
        return float(stat)

    T = len(f_os)
    if T < block * 4:
        return 0.0, 0.0, 1.0
    real = _sr(f_os, r_os)
    n_blocks = T // block
    rng = np.random.default_rng(0)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        bi = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([bi[j] * block + np.arange(block)
                              for j in range(n_blocks)])
        idx = idx[idx < T]
        boot[i] = _sr(f_os[idx], r_os[idx])
    sd = float(boot.std())
    t = float(real / sd) if sd > 1e-12 else 0.0
    # 中心化块自助 p 值：H0 下 bootstrap 分布平移到 0，
    # p = P(|boot - mean(boot)| >= |real|)。（历史 bug：未中心化 → p 恒 ≈0.5）
    p = float((np.abs(boot - boot.mean()) >= abs(real)).mean())
    return real, t, p


# ── LLM 调用（DeepSeek key 注入；无 key 自动降级规则化）──────────────────────

def _build_llm_call() -> object | None:
    """构造 llm_call（deepseek），无 key 时返回 None（引擎走规则化降级）。"""
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            return None
        from web.ai_providers import resolve_provider, chat_completions
        resolved = resolve_provider("deepseek", key)
        return lambda msgs: chat_completions(resolved, msgs, max_tokens=1024)
    except Exception:  # noqa: BLE001
        return None


# ── 单标的：三引擎联动挖掘 ──────────────────────────────────────────────────

_FS_LOCK = threading.Lock()  # FactorStore JSON index 并发写锁


class _RoutingConsole:
    """线程感知控制台代理：RL 训练日志按线程路由到私有缓冲，不刷屏。

    注意：tqdm 4.68.x 的 tqdm.write 默认写 sys.stdout（非 stderr），
    因此 stdout/stderr 都需代理。AlphaEngine 训练期间每步多行日志，
    全市场 5000 只 × 8 步 ≈ 4 万行，必须抑制；并发线程各自注册缓冲，
    代理按线程 id 路由，未注册线程的写入直通真实流（主线程进度不受影响）。
    """

    def __init__(self, real) -> None:
        self._real = real
        self._routes: dict[int, io.StringIO] = {}
        self._lock = threading.Lock()

    def register(self, buf: io.StringIO) -> None:
        with self._lock:
            self._routes[threading.get_ident()] = buf

    def unregister(self) -> None:
        with self._lock:
            self._routes.pop(threading.get_ident(), None)

    def write(self, s: str) -> int:
        with self._lock:
            buf = self._routes.get(threading.get_ident())
        if buf is not None:
            return buf.write(s)
        return self._real.write(s)

    def flush(self) -> None:
        try:
            self._real.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        return False  # 非交互 → tqdm 进度条自动 disable


_STDOUT_ROUTER: _RoutingConsole | None = None
_STDERR_ROUTER: _RoutingConsole | None = None


def _install_console_router() -> _RoutingConsole:
    """全局安装一次控制台路由代理（幂等）。返回 stdout 代理。"""
    global _STDOUT_ROUTER, _STDERR_ROUTER
    if _STDOUT_ROUTER is None:
        _STDOUT_ROUTER = _RoutingConsole(sys.stdout)
        _STDERR_ROUTER = _RoutingConsole(sys.stderr)
        sys.stdout = _STDOUT_ROUTER
        sys.stderr = _STDERR_ROUTER
    return _STDOUT_ROUTER


def _cleanup_rl_checkpoints(symbol: str) -> None:
    """RL 每轮训练都会落一个 checkpoint（8 步即触发），全市场会堆积磁盘，删之。"""
    try:
        for p in Path("checkpoints").glob(f"ckpt_{symbol}_*.pt"):
            p.unlink(missing_ok=True)
    except OSError:
        pass

# 预选公式（--quick-gate 开启时的快速筛选；覆盖动量/反转/微观结构）
_PRESELECT: tuple[tuple[str, int, object, str], ...] = (
    ("rsi14", 20, None, "Quantile"),
    ("macd_hist", 20, None, "Slope"),
    ("amt_vol_euclid", 20, None, "AC1"),
    ("vol_regime", 60, None, "Std"),
    ("ret20", 20, None, "Momentum"),
)


def _build_ret(close: np.ndarray, horizon: int) -> np.ndarray:
    T = len(close)
    ret = np.zeros(T)
    if T > horizon:
        ret[:T - horizon] = close[horizon:] / close[:-horizon] - 1.0
    return ret


def _quick_preselect(vm, ind: dict, ret: np.ndarray, gate: float) -> float:
    """预选层：固定公式最近 60 根的 |IC|，低于 gate 返回弱分。"""
    from model_core.formula_dsl import ParamFormula
    best = 0.0
    for a, w, sl, op in _PRESELECT:
        if a not in ind:
            continue
        f = ParamFormula(A=a, B=None, window=w, slice=sl, mask_field=None,
                         mask_rule=None, mode=1, mode1=op, mode2=None, B_shift_lag=0)
        try:
            fac = vm.execute(f)
            n = min(len(fac), len(ret))
            x, y = fac[-60:], ret[-60:]
            xm, ym = x - x.mean(), y - y.mean()
            sd = (xm ** 2).mean() ** 0.5 * (ym ** 2).mean() ** 0.5
            if sd > 1e-9:
                best = max(best, abs((xm * ym).mean() / sd))
        except Exception:  # noqa: BLE001
            continue
    return best


def mine_one(symbol: str, tf: str, cfg, ctx: dict) -> dict:
    """单只股票：GP（LLM 种子）+ RL（矿池精英预热，GPU）+ LLM（规则化）联动挖掘。

    ctx: {pool: MarketPool, llm_call, gp_seeds:[chrom], fs_done:set[str],
          fs_lock: threading.Lock}
    """
    t0 = time.time()
    engines = cfg.engines
    result = {"symbol": symbol, "status": "ok", "n_gp": 0, "n_llm": 0, "n_rl": 0,
              "n_accepted": 0, "best_dsr": 0.0, "best_ic": 0.0,
              "best_engine": "", "best_formula": "", "elapsed_s": 0.0}
    # 详细过程日志（--verbose 终端展示用）
    result["detail"] = {"data": {}, "gp": [], "llm": [], "rl_log": [],
                        "eval": []}
    try:
        store = KlineStore(cfg.store_dir)
        df = store.load(symbol, tf)

        # ── 数据量大管饱：无本地数据 → 全历史实拉（三源兜底 + 断联重试）──
        # 机构数据管道标准：腾讯(qfq复权) → 新浪(不复权) → 通达信(pytdx 不复权)，
        # 每源带重试退避；腾讯 501 / 新浪 456 均为 IP 级临时限流，等待后自动恢复
        if df.empty:
            if cfg.no_backfill:
                result["status"] = "no_data"
                return result
            from web.data_sources.factory import get_source
            bars = None
            last_err = ""
            # 机构 D1：腾讯默认后复权（hfq，Qlib 首日归一化后复权同构；
            # 前复权历史价被未来除权改写）；新浪/通达信为不复权（raw）
            # 通达信 3 次重试：批量并发时服务器限流返回空数组（源内已加空响应
            # 重连重试，这里再兜底一层）
            sources = [("tencent", 2, 1.0), ("sina", 2, 2.0),
                       ("tongdaxin", 3, 0.5)]
            for src_kind, n_try, base_wait in sources:
                if bars:
                    break
                try:
                    src = get_source(src_kind)
                    for attempt in range(n_try):
                        try:
                            bars = src.fetch_bars(
                                symbol, tf, n=100000, drop_forming=True,
                                adjust="hfq" if src_kind == "tencent" else "raw")
                            if bars:
                                result["detail"]["data"]["source_fallback"] = src_kind
                                break
                        except Exception as exc:  # noqa: BLE001
                            last_err = f"{type(exc).__name__}: {exc}"
                            time.sleep(base_wait * (1.0 + attempt))
                except Exception as exc:  # noqa: BLE001
                    last_err = f"{type(exc).__name__}: {exc}"
            if not bars:
                result["status"] = "no_data"
                result["best_formula"] = last_err[:80]
                return result
            import pandas as pd
            df = pd.DataFrame([{
                "ts": int(b.ts), "open": float(b.open), "high": float(b.high),
                "low": float(b.low), "close": float(b.close), "volume": float(b.volume),
            } for b in bars])
            # 通达信 volume 单位为手（1手=100股），统一为股（与腾讯/新浪一致）
            if result["detail"]["data"].get("source_fallback") == "tongdaxin":
                df = df.assign(volume=lambda d: d["volume"] * 100.0)
            store.update(symbol, tf, df,
                         source=result["detail"]["data"].get("source_fallback"),
                         adjust="hfq" if result["detail"]["data"].get(
                             "source_fallback") == "tencent" else "raw")

        if cfg.bars > 0 and len(df) > cfg.bars:
            df = df.iloc[-cfg.bars:]
        if len(df) < 120:
            result["status"] = "too_short"
            result["best_formula"] = f"仅 {len(df)} 根 K 线（<120）"
            return result

        # 机构级 D3：数据健康检查与清洗（重复日期剔除 + 跳变日标记）
        from data_pipeline.quality import check_series, clean_series, mask_jump_returns
        issues = check_series(df)
        df, jump_dates = clean_series(df)
        if len(df) < 120:
            result["status"] = "too_short"
            result["best_formula"] = f"清洗后仅 {len(df)} 根（<120）"
            return result
        result["detail"]["data"]["quality_issues"] = issues
        result["detail"]["data"]["jump_dates"] = len(jump_dates)
        # 混库硬门：跳变方向交替（qfq/不复权交替）→ 该标的 K 线不可信，拒绝挖掘
        from data_pipeline.quality import classify_jumps
        jkind, jinfo = classify_jumps(df)
        result["detail"]["data"]["jump_class"] = jkind
        if jkind == "mix":
            result["status"] = "dirty_data"
            result["best_formula"] = (
                f"K线疑似复权口径混库（{jinfo['n']} 次跳变，"
                f"{jinfo['alternations']} 次方向交替）——"
                f"删除 store/kline/{symbol}_1d.parquet 后重拉")
            return result

        from model_core.indicator_builder import build_indicators
        from model_core.param_vm import ParamVM
        from model_core.eval.report import build_factor_report

        ind = build_indicators(df)
        vm = ParamVM(ind)
        close = ind["close"]
        T = len(close)
        ret = _build_ret(close, cfg.horizon)
        # 跳变日收益标签置 0（数据瑕疵不产生伪收益）
        if jump_dates and "ts" in df.columns:
            ret = mask_jump_returns(ret, df["ts"].values, jump_dates)

        # 详细过程：数据量信息（量大管饱的直观证据）
        d = result["detail"]["data"]
        d["bars"] = len(df)
        if "ts" in df.columns and len(df) > 0:
            import datetime as _dt
            d["first"] = _dt.datetime.fromtimestamp(
                int(df["ts"].iloc[0])).strftime("%Y-%m-%d")
            d["last"] = _dt.datetime.fromtimestamp(
                int(df["ts"].iloc[-1])).strftime("%Y-%m-%d")
        d["horizon"] = cfg.horizon

        # 预选过滤（--quick-gate 开启时，省时模式）
        if cfg.quick_gate > 0:
            qic = _quick_preselect(vm, ind, ret, cfg.quick_gate)
            if qic < cfg.quick_gate:
                result["status"] = "filtered"
                result["best_ic"] = round(qic, 4)
                result["elapsed_s"] = round(time.time() - t0, 1)
                return result

        pool: MarketPool = ctx["pool"]
        candidates: list[dict] = []   # {chrom/formula, engine, kind, factor|None}
        library_factors: list[np.ndarray] = []

        # ── 机构三段式：训练段（挖掘选优）与 OOS 段（认证）严格分离 ──────
        # 选优引擎（GP/LLM/RL）只允许接触训练段收益，OOS 段不参与任何选优
        oos_n = max(int(T * cfg.oos_frac), 250)
        if oos_n >= T:
            oos_n = max(T // 2, 30)
        n_tr = T - oos_n
        if n_tr < 120:
            result["status"] = "too_short"
            return result
        ind_mine = {k: v[:n_tr] for k, v in ind.items()}
        ret_mine = ret[:n_tr]

        # ── 引擎一：GP（NSGA-III，初始种群注入 LLM/矿池种子 = 联动①）─────
        if "gp" in engines:
            from model_core.engines.gp_engine import NSGA3FactorMiner
            seeds = list(ctx.get("gp_seeds", []) or [])
            # 矿池 param 精英也作为 GP 种子（跨品种先验）
            for e in pool.top(4, kind="param"):
                seeds.append(e["formula"])
            miner = NSGA3FactorMiner(pop_size=cfg.pop, n_gen=cfg.gen, seed=cfg.seed)
            gp_results = miner.mine(ind_mine, ret_mine,
                                    init_chroms=seeds[:max(1, cfg.pop // 2)])
            n_trials = miner.history[-1]["n_eval"]
            for r in gp_results:
                candidates.append({"formula": r["formula"], "chrom": r["chrom"],
                                   "engine": "gp", "kind": "param",
                                   "n_trials": n_trials})
            result["n_gp"] = len(gp_results)
            # 详细过程：GP 非支配解（按 |IC| 排序，取前 8 条展示）
            for r in gp_results[:8]:
                objs = r["objectives"]
                result["detail"]["gp"].append({
                    "formula": r["formula"].describe(),
                    "abs_ic": round(float(objs[0]), 4),
                    "ic_win": round(float(objs[1]), 3),
                    "long_ret": round(float(objs[2]), 5),
                    "long_sharpe": round(float(objs[3]), 2),
                    "long_win": round(float(objs[4]), 3),
                })

        # ── 引擎二：RL（REINFORCE + token 公式；GPU；矿池精英预热 = 联动③）─
        if "rl" in engines:
            try:
                import torch
                from model_core.engine import AlphaEngine
                from model_core.config import ModelConfig as MC
                from model_core.feature_bridge import RLDataManager, execute_token_formula

                # 训练用训练段（前缀与全窗口一致 → 因果特征一致）；
                # 执行用全窗口（因子覆盖 OOS 段，供认证）
                rl_bars = min(cfg.rl_bars, n_tr) if cfg.rl_bars > 0 else n_tr
                df_tr = df.iloc[:n_tr]
                df_tr_rl = df_tr.iloc[-rl_bars:]
                dm_tr = RLDataManager(df_tr_rl, horizon=cfg.horizon)
                dm_full = RLDataManager(df, horizon=cfg.horizon)

                saved_batch = MC.BATCH_SIZE
                MC.BATCH_SIZE = cfg.rl_batch
                try:
                    eng = AlphaEngine(data_manager=dm_tr, target_symbol=symbol,
                                      n_folds=cfg.rl_folds)
                    # 联动③：矿池 token 精英 → RL 精英回放池预热（跨品种迁移）
                    preheat = pool.token_formulas(_RL_ELITE_PREHEAT)
                    if preheat:
                        eng._elite_pool = []
                        for sc, toks in preheat:
                            eng._elite_pool.append(
                                (max(float(sc), 0.05), eng._elite_counter, toks, 0))
                            eng._elite_counter += 1
                        heapq.heapify(eng._elite_pool)
                    # 压制 AlphaEngine 的每步刷屏日志，只保留关键行
                    # （tqdm.write 走 stdout；线程感知路由，并发互不干扰；
                    #   --verbose 时整段日志存入 detail 供终端完整回显）
                    router = _install_console_router()
                    buf = io.StringIO()
                    router.register(buf)
                    try:
                        eng.train(start_step=0, end_step=cfg.rl_steps,
                                  verbose_header=False)
                    finally:
                        router.unregister()
                    _cleanup_rl_checkpoints(symbol)
                finally:
                    MC.BATCH_SIZE = saved_batch

                # 详细过程：RL 完整训练日志（每步 IC/验证/熵/精英池）+ 预热信息
                rl_lines = [ln.rstrip() for ln in buf.getvalue().splitlines() if ln.strip()]
                result["detail"]["rl_log"] = rl_lines
                if preheat:
                    result["detail"]["rl_log"].insert(
                        0, f"[预热] 矿池 token 精英注入回放池 {len(preheat)} 条"
                           f"（跨品种迁移，首条={_decode_tokens(preheat[0][1])[:48]}）")

                if eng.best_formula is not None:
                    factor_t = execute_token_formula(eng.best_formula, dm_full)
                    factor = np.asarray(factor_t.detach().cpu().numpy(), dtype=np.float64)
                    # 因子对齐到全窗口：RL 因子长度 = T（与 GP/LLM 一致）
                    if len(factor) != T:
                        factor = factor[-T:]
                    # 关键信息回显（新最优/公式）
                    for line in buf.getvalue().splitlines():
                        if "[!]" in line or "新最优" in line:
                            print(f"      {line.strip()}", flush=True)
                    candidates.append({
                        "formula": [int(t) for t in eng.best_formula],
                        "chrom": [int(t) for t in eng.best_formula],
                        "engine": "rl", "kind": "token",
                        "factor": factor, "n_trials": cfg.rl_steps * cfg.rl_batch,
                    })
                    result["n_rl"] = 1
                    result["rl_best"] = float(eng.best_score)
            except Exception as exc:  # noqa: BLE001
                result["rl_error"] = f"{type(exc).__name__}: {exc}"

        # ── 引擎三：LLM 多智能体（批级注入假设，股票级规则化落地）──────────
        # 股票级用 llm_call=None（规则化降级，零 API 成本、快）；
        # 真 LLM 挖掘只在批级联动 _llm_batch_discovery 运行（每 llm_batch 只一次），
        # 其产出的种子/假设通过矿池 + gp_seeds 流入本股票。
        if "llm" in engines:
            try:
                from model_core.engines.llm_engine import LLMAgentMiner
                miner = LLMAgentMiner(
                    llm_call=None, max_rounds=cfg.llm_rounds,
                    seed=cfg.seed,
                    hypotheses=pool.hypotheses_for_llm(k=3) or None,
                )
                llm_results = miner.mine(
                    ind_mine, ret_mine, n_hypotheses=cfg.llm_hyp,
                    library_factors=library_factors or None,
                    existing_formulas=pool.param_formulas(30),
                )
                for r in llm_results:
                    candidates.append({
                        "formula": r.formula, "chrom": r.chrom,
                        "engine": "llm", "kind": "param",
                        "n_trials": max(cfg.llm_hyp * cfg.llm_rounds * 10, 20),
                    })
                    # 详细过程：假设 → 公式 → 对齐/新颖性/评估
                    result["detail"]["llm"].append({
                        "hypothesis": r.hypothesis[:80],
                        "formula": r.formula.describe(),
                        "alignment": round(r.alignment, 2),
                        "novelty": round(r.novelty, 2),
                        "rounds": r.rounds,
                        "rejected": r.rejected,
                        "reason": r.reason[:60],
                        "dsr": round((r.report or {}).get("dsr", 0.0), 3),
                        "ic": round((r.report or {}).get("ic", 0.0), 4),
                    })
                result["n_llm"] = len(llm_results)
            except Exception as exc:  # noqa: BLE001
                result["llm_error"] = f"{type(exc).__name__}: {exc}"

        if not candidates:
            result["status"] = "no_candidates"
            result["elapsed_s"] = round(time.time() - t0, 1)
            return result

        # ── 候选产出（机构范式：认证在批级横截面完成，此处只产出 + 辅助信息）──
        # 每个候选携带：公式（param chrom / token 序列）、源股票全窗口因子/收益/时间戳，
        # 以及源股票 OOS 五维（辅助展示，不作为门槛）
        cands_out: list[dict] = []
        for cand in candidates:
            try:
                if cand.get("engine") == "rl":
                    factor = np.asarray(cand["factor"], dtype=np.float64)
                else:
                    factor = vm.execute(cand["formula"])
                if len(factor) < 120:
                    continue
                ret_eval = ret[-len(factor):] if len(factor) != len(ret) else ret
                ts_arr = (df["ts"].values[-len(factor):].astype(np.int64)
                          if "ts" in df.columns else np.arange(len(factor)))
                desc = (cand["formula"].describe()
                        if cand.get("kind") == "param" else _decode_tokens(cand["formula"]))
                # 源股票 OOS 段五维（辅助信息，机构范式主认证在横截面）
                five_total = 0.0
                oos_n = max(int(len(ret_eval) * cfg.oos_frac), 250)
                if oos_n < len(ret_eval):
                    try:
                        rep = build_factor_report(
                            factor[-oos_n:], ret_eval[-oos_n:], cand["chrom"], desc,
                            n_trials=cand["n_trials"],
                            symbol=symbol, engine=cand["engine"])
                        five_total = float(rep.five_dim.total)
                    except Exception:  # noqa: BLE001
                        five_total = 0.0
                cands_out.append({
                    "source": symbol,
                    "engine": cand["engine"],
                    "kind": cand["kind"],
                    "chrom": [int(t) for t in cand["chrom"]],
                    "desc": desc,
                    "n_trials": cand["n_trials"],
                    "factor": factor,
                    "ret": ret_eval,
                    "ts": ts_arr,
                    "five_total": five_total,
                })
            except Exception:  # noqa: BLE001
                continue

        result["candidates_out"] = cands_out
        result["n_candidates"] = len(cands_out)
        if cands_out:
            result["status"] = "ok"
            best = max(cands_out, key=lambda c: c.get("five_total", 0.0))
            result["best_engine"] = best["engine"]
            result["best_formula"] = best["desc"][:80]
        else:
            result["status"] = "no_candidates"
    except Exception as exc:  # noqa: BLE001
        result["status"] = f"error: {type(exc).__name__}"
        result["best_formula"] = str(exc)[:80]
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


# ── 机构范式：批级横截面认证（对标 Qlib/华泰）────────────────────────────────

_MIN_CROSS_STOCKS = 30     # 有效截面股票数下限（低于则降级单标的；机构横截面需 ≥30 才稳定）
_MIN_CROSS_DAYS = 250      # 有效截面交易日下限（约 1 年）
_CROSS_BLOCK = 20          # 块自助块长（交易日）


def _certify_single_series(cand: dict, cfg) -> dict | None:
    """降级路径：源股票单标的 OOS 认证（股票池不足时使用）。

    门槛：|OOS RankIC| ≥ min_oos_rankic 且 块自助 p ≤ min_oos_p 且方向一致
    且 源股票 OOS 五维 ≥ 0.45。
    """
    factor, ret = cand["factor"], cand["ret"]
    oos_n = max(int(len(ret) * cfg.oos_frac), 250)
    if oos_n >= len(ret) or oos_n < 250:
        return None
    f_os, r_os = factor[-oos_n:], ret[-oos_n:]
    from model_core.eval.five_dim import _seq_rankic
    oos_rankic, oos_t, oos_p = _oos_significance(f_os, r_os)
    # nan 防护：常数/退化因子 spearmanr 返回 nan，比较全 False 会误放行
    if not np.isfinite(oos_rankic) or not np.isfinite(oos_p):
        return None
    ric_tr = _seq_rankic(factor[:-oos_n], ret[:-oos_n])
    tr_rankic = float(ric_tr.mean()) if len(ric_tr) else 0.0
    if (abs(oos_rankic) < cfg.min_oos_rankic or oos_p > cfg.min_oos_p
            or (oos_rankic * tr_rankic) <= 0.0 or cand.get("five_total", 0.0) < 0.45):
        return None
    out = dict(cand)
    out["cert"] = {"mode": "single_series", "rankic": oos_rankic,
                   "p": oos_p, "stocks": 1, "days": oos_n}
    return out


def _save_certified(c: dict, cfg, pool: MarketPool) -> str | None:
    """认证通过的候选 → 因子库 + 矿池（机构范式：横截面认证信息入库）。"""
    try:
        symbol = c["source"]
        factor = np.asarray(c["factor"], dtype=np.float64)
        ret_eval = np.asarray(c["ret"], dtype=np.float64)
        oos_n = max(int(len(ret_eval) * cfg.oos_frac), 250)
        if oos_n >= len(ret_eval):
            oos_n = max(len(ret_eval) // 2, 30)
        f_os, r_os = factor[-oos_n:], ret_eval[-oos_n:]
        import pandas as pd
        from model_core.eval.report import build_factor_report
        rep = build_factor_report(
            f_os, r_os, c["chrom"], c["desc"],
            n_trials=c["n_trials"],
            symbol=symbol, engine=c["engine"],
            cert_mode=c["cert"]["mode"],
            cert_rankic=round(float(c["cert"]["rankic"]), 4),
            cert_p=round(float(c["cert"]["p"]), 4),
            cert_stocks=int(c["cert"]["stocks"]),
            cert_days=int(c["cert"]["days"]),
        )
        direction = 1.0 if c["cert"]["rankic"] >= 0 else -1.0
        rep.meta["direction"] = direction
        fdf = pd.DataFrame({"factor": factor})
        fs = FactorStore(cfg.store_dir)
        with _FS_LOCK:
            fh = fs.save(symbol, c["chrom"], "param-v1", fdf,
                         report=rep.as_dict())
        pool.add(c["kind"], c["engine"], symbol, c["chrom"], c["desc"],
                 float(rep.ic), float(rep.dsr), float(rep.five_dim.total))
        return fh
    except Exception:  # noqa: BLE001
        return None


def _certify_batch(cands: list[dict], cert_symbols: list[str], tf: str,
                   cfg, pool: MarketPool, ctx: dict) -> tuple[list[dict], dict]:
    """机构范式认证：候选公式跨股票池执行 → 截面 RankIC + 块自助显著性。

    对每个候选：
      1. 在认证股票池（KlineStore 已有 A股 + 本批）上执行公式 → [N, T] 因子矩阵
      2. 每个共同交易日 t（有效股票 ≥8）：截面 RankIC = Spearman(因子截面, 收益截面)
      3. 截面 RankIC 序列 → 块自助（块长 20 交易日）→ p 值
      4. 门槛：有效交易日 ≥250 且 |mean_rankic| ≥ min_oos_rankic 且 p ≤ min_oos_p
         且 源股票 OOS 五维 ≥ 0.45
    股票池不足（有效截面 <8）→ 降级单标的 OOS 认证。

    Returns: (通过候选列表, 统计 dict)
    """
    stats = {"n_cands": len(cands), "n_cross": 0, "n_single": 0, "n_reject": 0,
             "n_stocks": 0, "n_days": 0}
    if not cands:
        return [], stats

    from data_pipeline.quality import clean_series
    from model_core.indicator_builder import build_indicators
    from model_core.param_vm import ParamVM

    # ── 1. 构建认证股票池的数据缓存（所有候选共享）────────────────────
    store = KlineStore(cfg.store_dir)
    # 股票池：KlineStore 已有 A 股 + 本批股票（去重）
    symbols = []
    seen_sym = set()
    for s in list(cert_symbols) + [c["source"] for c in cands]:
        if s not in seen_sym:
            seen_sym.add(s)
            symbols.append(s)
    for item in store.list_cached():
        code = item.get("code", "")
        if code.startswith(("sh", "sz")) and code not in seen_sym:
            seen_sym.add(code)
            symbols.append(code)

    data: dict[str, dict] = {}   # symbol -> {ind, ret, ts, df}
    for s in symbols:
        try:
            df = store.load(s, tf)
            if df.empty or len(df) < 250:
                continue
            if cfg.bars > 0 and len(df) > cfg.bars:
                df = df.iloc[-cfg.bars:]
            df, _jumps = clean_series(df)
            if len(df) < 250:
                continue
            ind = build_indicators(df)
            close = ind["close"]
            T = len(close)
            ret = _build_ret(close, cfg.horizon)
            ts = df["ts"].values.astype(np.int64)
            data[s] = {"ind": ind, "ret": ret, "ts": ts, "df": df}
        except Exception:  # noqa: BLE001
            continue
    stats["n_stocks"] = len(data)

    accepted: list[dict] = []
    for cand in cands:
        try:
            # ── 2. 跨股票执行公式 ─────────────────────────────────────
            series = []   # (ts, factor, ret)
            kind = cand["kind"]
            for s, d in data.items():
                try:
                    if kind == "param":
                        from model_core.formula_dsl import chrom_to_formula
                        vm = ParamVM(d["ind"])
                        f = vm.execute(chrom_to_formula(cand["chrom"]))
                    else:
                        from model_core.feature_bridge import RLDataManager, \
                            execute_token_formula
                        dm = RLDataManager(d["df"], horizon=cfg.horizon)
                        ft = execute_token_formula(cand["chrom"], dm)
                        f = np.asarray(ft.detach().cpu().numpy(), dtype=np.float64)
                    ret_v = d["ret"][-len(f):] if len(f) != len(d["ret"]) else d["ret"]
                    ts_v = d["ts"][-len(f):]
                    series.append((ts_v, f, ret_v))
                except Exception:  # noqa: BLE001
                    continue
            # ── 3. 截面 RankIC 序列 ───────────────────────────────────
            if len(series) < _MIN_CROSS_STOCKS:
                # 降级：单标的 OOS
                stats["n_single"] += 1
                c = _certify_single_series(cand, cfg)
                if c:
                    accepted.append(c)
                continue

            rankics, days = _cross_sectional_rankics(series)
            stats["n_days"] = max(stats["n_days"], days)
            if days < _MIN_CROSS_DAYS or len(rankics) < 20:
                stats["n_reject"] += 1
                continue
            mean_ric = float(np.mean(rankics))
            if not np.isfinite(mean_ric):
                stats["n_reject"] += 1
                continue
            # 时间分段方向一致性（机构稳健性检验）：认证期前半/后半的截面
            # RankIC 均值必须同号——小股票池上截面信号不稳定，跨时段翻转即拒绝
            half = len(rankics) // 2
            if half >= 30:
                m1, m2 = float(np.mean(rankics[:half])), float(np.mean(rankics[half:]))
                if m1 * m2 <= 0.0:
                    stats["n_reject"] += 1
                    continue
            # 中心化块自助 p 值（H0 下分布平移到 0；历史 bug：未中心化 → p 恒≈0.5）
            n_blocks = len(rankics) // _CROSS_BLOCK
            rng = np.random.default_rng(0)
            boot = np.empty(500)
            for i in range(500):
                bi = rng.integers(0, n_blocks, size=n_blocks)
                idx = np.concatenate([bi[j] * _CROSS_BLOCK + np.arange(_CROSS_BLOCK)
                                      for j in range(n_blocks)])
                idx = idx[idx < len(rankics)]
                boot[i] = float(np.mean(rankics[idx]))
            sd = float(boot.std())
            p = float((np.abs(boot - boot.mean()) >= abs(mean_ric)).mean())
            if not np.isfinite(p):
                stats["n_reject"] += 1
                continue
            if (abs(mean_ric) >= cfg.min_oos_rankic and p <= cfg.min_oos_p
                    and cand.get("five_total", 0.0) >= 0.45):
                out = dict(cand)
                out["cert"] = {"mode": "cross_sectional", "rankic": mean_ric,
                               "p": p, "stocks": len(series), "days": days}
                accepted.append(out)
                stats["n_cross"] += 1
            else:
                stats["n_reject"] += 1
        except Exception:  # noqa: BLE001
            stats["n_reject"] += 1
    return accepted, stats


def _cross_sectional_rankics(series: list) -> tuple[np.ndarray, int]:
    """多股票 (ts, factor, ret) → 截面 RankIC 序列（按共同交易日对齐）。

    每个交易日 t：对当日前 `有效股票` 的 (因子值, 未来收益) 计算 Spearman。
    返回 (rankic 序列, 有效交易日数)。
    """
    from scipy.stats import spearmanr

    # 收集日期并集
    day_map: dict[int, list] = {}   # ts -> [(f, r), ...]
    for ts_v, f, r in series:
        n = min(len(ts_v), len(f), len(r))
        for i in range(n):
            day_map.setdefault(int(ts_v[i]), []).append((float(f[i]), float(r[i])))
    days = sorted(day_map.keys())
    rankics = []
    for d in days:
        pairs = day_map[d]
        if len(pairs) < _MIN_CROSS_STOCKS:
            continue
        fs = np.array([p[0] for p in pairs])
        rs = np.array([p[1] for p in pairs])
        if fs.std() < 1e-12 or rs.std() < 1e-12:
            continue
        r = spearmanr(fs, rs)
        rankics.append(float(r.statistic if hasattr(r, "statistic") else r[0]))
    return np.array(rankics), len(rankics)


# ── 批级 LLM 联动（联动①②在批次间的落点）───────────────────────────────────

def _llm_batch_discovery(symbol: str, tf: str, cfg, pool: MarketPool,
                         llm_call) -> list[list[int]]:
    """批首运行一次真 LLM 挖掘：产出染色体种子注入本批 GP（联动①）。

    假设由矿池发现日志驱动（联动②）；公式受矿池新颖性约束。
    无 key 时走规则化降级（仍产出种子，保证每批都有 LLM 参与）。
    """
    seeds: list[list[int]] = []
    try:
        store = KlineStore(cfg.store_dir)
        df = store.load(symbol, tf)
        if df.empty or len(df) < 120:
            return seeds
        if cfg.bars > 0 and len(df) > cfg.bars:
            df = df.iloc[-cfg.bars:]
        from model_core.indicator_builder import build_indicators
        from model_core.engines.llm_engine import LLMAgentMiner
        ind = build_indicators(df)
        close = ind["close"]
        ret = _build_ret(close, cfg.horizon)
        miner = LLMAgentMiner(
            llm_call=llm_call, max_rounds=cfg.llm_rounds, seed=cfg.seed,
            hypotheses=pool.hypotheses_for_llm(k=5) or None,
        )
        results = miner.mine(ind, ret, n_hypotheses=cfg.llm_hyp,
                             library_factors=None,
                             existing_formulas=pool.param_formulas(50))
        for r in results:
            if r.report:
                pool.add("param", "llm", symbol, r.chrom,
                         r.formula.describe(),
                         float(r.report.get("ic", 0.0)),
                         float(r.report.get("dsr", 0.0)),
                         float((r.report.get("five_dim") or {}).get("total", 0.0)))
            seeds.append(r.chrom)
        print(f"  [LLM批联动] {symbol}: {len(results)} 个假设落地 → 注入本批 GP "
              f"种子 {len(seeds)} 条", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [LLM批联动跳过] {type(exc).__name__}: {exc}", flush=True)
    return seeds


# ── 主流程 ──────────────────────────────────────────────────────────────────

def _setup_device(device: str) -> str:
    """CUDA GPU 加速：RL 引擎设备选择。返回实际设备名。"""
    import torch
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            from model_core.config import ModelConfig
            ModelConfig.DEVICE = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            vram = (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)
            print(f"[GPU] {name}（显存 {vram:.1f} GB）→ RL 引擎跑 CUDA", flush=True)
            return "cuda"
    from model_core.config import ModelConfig
    ModelConfig.DEVICE = torch.device("cpu")
    print("[GPU] CUDA 不可用或已指定 CPU → RL 引擎跑 CPU", flush=True)
    return "cpu"


# ── 终端详细过程显示（--verbose）─────────────────────────────────────────────

def _print_mine_detail(r: dict) -> None:
    """把一只股票的完整挖掘过程打印到终端（数据量/GP 解/LLM 假设/RL 日志/裁决明细）。"""
    d = r.get("detail") or {}
    pad = "      "
    data = d.get("data") or {}
    if data:
        print(f"{pad}── 数据（机构D3健康检查）──", flush=True)
        print(f"{pad}窗口 {data.get('bars', '?')} 根 "
              f"[{data.get('first', '?')} .. {data.get('last', '?')}] "
              f"horizon={data.get('horizon', '?')} "
              f"跳变日={data.get('jump_dates', 0)}", flush=True)
        q = data.get("quality_issues") or []
        if q:
            print(f"{pad}  健康检查: {'; '.join(str(x) for x in q)}", flush=True)
        else:
            print(f"{pad}  健康检查: 通过", flush=True)

    gp = d.get("gp") or []
    if gp:
        print(f"{pad}── GP(NSGA-III) 非支配解 {len(gp)} 条 ──", flush=True)
        for i, g in enumerate(gp, 1):
            print(f"{pad}#{i} |IC|={g['abs_ic']:.3f} IC胜率={g['ic_win']:.2f} "
                  f"多头收益={g['long_ret']:.4f} 多头夏普={g['long_sharpe']:.2f} "
                  f"多头胜率={g['long_win']:.2f}", flush=True)
            print(f"{pad}    {g['formula']}", flush=True)

    llm = d.get("llm") or []
    if llm:
        print(f"{pad}── LLM 多智能体 {len(llm)} 条 ──", flush=True)
        for i, l in enumerate(llm, 1):
            print(f"{pad}#{i} [假设] {l['hypothesis']}", flush=True)
            print(f"{pad}   [公式] {l['formula']} "
                  f"(对齐={l['alignment']:.2f} 新颖={l['novelty']:.2f} "
                  f"DSR={l['dsr']:.2f} IC={l['ic']:.3f} 轮数={l['rounds']})", flush=True)

    rl_log = d.get("rl_log") or []
    if rl_log:
        print(f"{pad}── RL(REINFORCE, GPU) 训练日志 {len(rl_log)} 行 ──", flush=True)
        for line in rl_log:
            print(f"{pad}{line.strip()[:150]}", flush=True)

    ev = d.get("eval") or []
    if ev:
        print(f"{pad}── 统一裁决 {len(ev)} 候选（OOS认证: 块自助p值+方向一致+五维）──",
              flush=True)
        for e in ev:
            print(f"{pad}[{e['engine']}/{e['kind']}] {e['desc'][:52]} "
                  f"OOS_rankIC={e['oos_rankic']} OOS_p={e['oos_p']} "
                  f"DSR={e['dsr']} 五维={e['total']} Sharpe={e['sharpe']} "
                  f"PBO={e['pbo']} → {e['verdict']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="全市场三引擎(GP/RL/LLM)联动挖矿机 —— 每次运行全市场一起挖")
    ap.add_argument("--symbols-file", default="",
                    help="自定义股票清单文件（默认=全 A 股 5000+ 只，新浪 hs_a）")
    ap.add_argument("--refresh-universe", action="store_true",
                    help="强制刷新全 A 股清单缓存")
    ap.add_argument("--limit", type=int, default=0, help="矿脉中取前 N 只（0=全部）")
    ap.add_argument("--tf", default="1d", help="周期（腾讯支持 1d/1w/1M/60m 等）")
    ap.add_argument("--bars", type=int, default=2000,
                    help="挖掘窗口根数（0=全历史，量大管饱）")
    ap.add_argument("--rl-bars", type=int, default=800, help="RL 训练窗口根数")
    ap.add_argument("--horizon", type=int, default=5, help="收益预测周期")
    ap.add_argument("--workers", type=int, default=4, help="股票级并行线程数")
    ap.add_argument("--engines", default="gp,rl,llm",
                    help="启用引擎，逗号分隔: gp,rl,llm（默认三引擎全开）")
    ap.add_argument("--gen", type=int, default=8, help="GP 代数")
    ap.add_argument("--pop", type=int, default=48, help="GP 种群")
    ap.add_argument("--rl-steps", type=int, default=8, help="RL 训练步数")
    ap.add_argument("--rl-batch", type=int, default=64, help="RL 每步采样公式数")
    ap.add_argument("--rl-folds", type=int, default=3, help="RL walk-forward 折数")
    ap.add_argument("--llm-hyp", type=int, default=3, help="LLM 每批假设数")
    ap.add_argument("--llm-rounds", type=int, default=1, help="LLM 反馈轮数")
    ap.add_argument("--llm-batch", type=int, default=50,
                    help="批级 LLM 联动间隔（每 N 只股票注入一次真 LLM 假设）")
    ap.add_argument("--dsr-gate", type=float, default=0.0,
                    help="DSR 报告门槛（Bailey 修正口径后为报告项，默认不拦截；"
                         "主门槛是 OOS RankIC 显著性）")
    ap.add_argument("--oos-frac", type=float, default=0.25,
                    help="样本外认证段比例（机构三段式：挖掘→OOS 验证）")
    ap.add_argument("--min-oos-rankic", type=float, default=0.02,
                    help="OOS 段整体 RankIC 最低门槛")
    ap.add_argument("--min-oos-p", type=float, default=0.05,
                    help="OOS 段块自助检验 p 值上限（显著性门槛）")
    ap.add_argument("--crowd-corr", type=float, default=0.85,
                    help="因子库拥挤度去重阈值（与新因子相关性超过即丢弃）")
    ap.add_argument("--cert-batch", type=int, default=20,
                    help="横截面认证批大小（每 N 只股票完成后统一跨股票认证）")
    ap.add_argument("--quick-gate", type=float, default=0.0,
                    help=">0 时开启预选过滤（|IC| 低于此值的股票跳过三引擎深度挖掘）")
    ap.add_argument("--skip-done", action="store_true",
                    help="断点续跑：跳过已有因子/已完成的标的")
    ap.add_argument("--no-backfill", action="store_true",
                    help="不自动拉取缺失股票的 K 线（只挖本地已有数据）")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="计算设备（auto=有 CUDA 用 GPU）")
    ap.add_argument("--store-dir", default="store", help="数据/因子/矿池存储根目录")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true",
                    help="终端显示每只股票的详细挖掘过程（GP 解/LLM 假设/RL 日志/裁决明细）")
    args = ap.parse_args()

    engines = tuple(e.strip() for e in args.engines.split(",") if e.strip())
    if not engines:
        ap.error("--engines 至少指定一个引擎")

    device = _setup_device(args.device)
    llm_call = _build_llm_call() if "llm" in engines else None
    if llm_call:
        print("[LLM] DeepSeek key 已配置 → 批级 LLM 真挖掘", flush=True)
    else:
        print("[LLM] 无 key → 规则化降级模式（预置假设+模板公式）", flush=True)

    store_dir = args.store_dir
    universe = _resolve_universe(args.symbols_file, args.limit, store_dir,
                                 args.refresh_universe)
    print(f"[矿脉] 全市场 {len(universe)} 只 A股 × 引擎({','.join(engines)}) "
          f"workers={args.workers} device={device}", flush=True)

    # 断点续跑：跳过已入库因子/已完成标的
    done: set[str] = set()
    if args.skip_done:
        try:
            index = FactorStore(store_dir)._load_index()
            done |= {str(m["symbol"]) for m in index.values()}
        except Exception:  # noqa: BLE001
            pass
        prog = Path(store_dir) / "meta" / "full_market_progress.json"
        if prog.exists():
            try:
                data = json.loads(prog.read_text(encoding="utf-8"))
                done |= set(data.get("completed", []))
            except (json.JSONDecodeError, OSError):
                pass
        before = len(universe)
        universe = [s for s in universe if s not in done]
        print(f"[断点] 已挖 {len(done)} 只，本次待挖 {len(universe)}/{before} 只", flush=True)

    if not universe:
        print("[完成] 矿脉已全部挖完（断点续跑无待挖标的）", flush=True)
        return

    pool = MarketPool(store_dir)
    print(f"[矿池] 联动矿池加载完成：{pool.size()} 条历史精英（GP/RL/LLM 共享）",
          flush=True)

    # 批级 LLM 状态：当前批的 GP 种子（联动①），每 llm_batch 只刷新一次
    batch_counter = 0
    gp_seeds: list[list[int]] = []

    t_all = time.time()
    rows: list[dict] = []
    progress_path = Path(store_dir) / "meta" / "full_market_progress.json"
    completed: list[str] = []
    ctx = {"pool": pool, "llm_call": llm_call, "gp_seeds": gp_seeds}
    ctx_lock = threading.Lock()

    def _save_progress() -> None:
        try:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = progress_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(
                {"version": 1, "completed": completed,
                 "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                ensure_ascii=False), encoding="utf-8")
            tmp.replace(progress_path)
        except OSError:
            pass

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool_ex:
        futs = {pool_ex.submit(mine_one, s, args.tf, args, ctx): s for s in universe}
        done_cnt = 0
        pending_cands: list[dict] = []
        cert_counter = 0
        total_certified = 0
        for fut in as_completed(futs):
            s = futs[fut]
            done_cnt += 1
            r = fut.result()
            rows.append(r)
            # 断点续跑语义（2026-08-27 修复）：
            #   确定性状态（ok/none_accepted/too_short/dirty_data/filtered）→ 标记完成，
            #   下次 --skip-done 跳过；
            #   临时状态（no_data/error）→ 不标记，下次运行重试
            #   （数据源风控窗口内失败的正常股票不应被永久跳过）。
            if r.get("status") not in ("no_data", "error", "dirty_data"):
                completed.append(s)
            pending_cands.extend(r.get("candidates_out", []) or [])
            with ctx_lock:
                batch_counter += 1
                cert_counter += 1
                # 联动①②：每 llm_batch 只刷新一次批级 LLM 假设 → 下批 GP 种子
                if "llm" in engines and batch_counter % max(1, args.llm_batch) == 0:
                    ctx["gp_seeds"] = _llm_batch_discovery(s, args.tf, args, pool,
                                                           llm_call)
                # 机构范式：每 cert_batch 只做一次横截面认证（候选跨股票执行）
                if cert_counter % max(1, args.cert_batch) == 0 \
                        or done_cnt == len(universe):
                    cert_symbols = list(completed[-max(1, args.cert_batch):])
                    n_cert, st = _certify_batch(pending_cands, cert_symbols,
                                                args.tf, args, pool, ctx)
                    for c in n_cert:
                        total_certified += 1
                        _save_certified(c, cfg=args, pool=pool)
                    if n_cert or st["n_cands"]:
                        print(f"[认证] 批{cert_counter // max(1, args.cert_batch)}: "
                              f"候选{st['n_cands']} → 横截面通过{st['n_cross']} "
                              f"(降级单标的{st['n_single']} 拒绝{st['n_reject']}) "
                              f"股票池{st['n_stocks']}只 交易日{st['n_days']}",
                              flush=True)
                        for c in n_cert[:5]:
                            ck = c["cert"]
                            print(f"      [入池 {c['engine']}/{c['kind']}] "
                                  f"截面RankIC={ck['rankic']:+.4f} p={ck['p']:.3f} "
                                  f"({ck['mode']}, {ck['stocks']}只×{ck['days']}日) "
                                  f"五维={c['five_total']:.2f} | {c['desc'][:52]}",
                                  flush=True)
                    pending_cands = []
                if done_cnt % 20 == 0 or done_cnt == len(universe):
                    _save_progress()
                    pool.save()
            eta = (time.time() - t_all) / done_cnt * (len(universe) - done_cnt)
            status = r["status"]
            print(f"[{done_cnt}/{len(universe)}] {s} {status} "
                  f"GP={r['n_gp']} LLM={r['n_llm']} RL={r['n_rl']} "
                  f"候选={r.get('n_candidates', 0)} {r['elapsed_s']}s "
                  f"ETA={eta / 60:.1f}min", flush=True)
            if r.get("best_formula"):
                print(f"      [最佳 {r['best_engine']}] {r['best_formula']}", flush=True)
            if args.verbose:
                _print_mine_detail(r)

    pool.save()
    _save_progress()

    # 汇总 CSV
    out_dir = Path(store_dir) / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "full_market_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=[
            "symbol", "status", "n_gp", "n_llm", "n_rl", "n_accepted",
            "best_dsr", "best_ic", "best_engine", "best_formula", "elapsed_s"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in [
                "symbol", "status", "n_gp", "n_llm", "n_rl", "n_accepted",
                "best_dsr", "best_ic", "best_engine", "best_formula", "elapsed_s"]})

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n[汇总] {len(rows)} 只：挖掘成功 {n_ok}，横截面认证入库 {total_certified} 条，"
          f"矿池 {pool.size()} 条精英，总耗时 {(time.time() - t_all) / 60:.1f}min")
    print(f"[报告] → {csv_path}")
    print(f"[矿池] → {Path(store_dir) / _POOL_FILE}")
    print(f"[断点] → {progress_path}")


if __name__ == "__main__":
    main()
