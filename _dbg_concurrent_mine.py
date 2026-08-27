"""复现：4 worker 并发挖矿是否死锁（模拟真实 --workers 4 场景）。"""
import sys
sys.path.insert(0, ".")
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.mine_full_market import mine_one, MarketPool


class Cfg:
    engines = ("gp", "llm", "rl")
    tf = "1d"
    bars = 2000
    rl_bars = 200
    horizon = 5
    gen = 6
    pop = 48
    rl_steps = 6
    rl_batch = 48
    rl_folds = 2
    llm_hyp = 3
    llm_rounds = 1
    llm_batch = 50
    dsr_gate = 0.0
    oos_frac = 0.25
    min_oos_rankic = 0.02
    min_oos_p = 0.05
    crowd_corr = 0.85
    cert_batch = 20
    quick_gate = 0.0
    no_backfill = True
    seed = 42
    store_dir = "store"


def main():
    symbols = ["sh600026", "sh600027", "sh600028", "sh600029",
               "sh600030", "sh600031", "sh600032", "sh600033"]
    pool = MarketPool("store")
    ctx = {"pool": pool, "llm_call": None, "gp_seeds": []}
    print(f"4 worker 并发 {len(symbols)} 只（GP+RL+LLM）...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(mine_one, s, "1d", Cfg(), ctx): s for s in symbols}
        for fut in as_completed(futs):
            s = futs[fut]
            r = fut.result(timeout=300)
            print(f"  {s}: {r['status']} 耗时{r['elapsed_s']}s "
                  f"GP={r['n_gp']} RL={r['n_rl']}", flush=True)
    print(f"全部完成，总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
