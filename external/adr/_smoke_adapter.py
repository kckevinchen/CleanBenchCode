import sys, time
from pathlib import Path
ADR = Path(__file__).resolve().parent
sys.path.insert(0, str(ADR))
import adapter

DS = ADR.parents[1] / "external/rein/rein-benchmark/datasets"
small = ["141944852_Region",
         "012240e8708e14939670412f3b559e46f5926fb9.xlsx_8a91dfd8",
         "0c193a7d34b5d4fe53c7a0ce0e7aac59aa10137b.xlsx_8c6da0b3"]
methods = sys.argv[1:] or ["bigdansing","unified","relative_trust","holistic"]
print(f"{'method':16} {'table':44} {'rc_err':>6} {'dets':>5} {'sec':>6}")
print("-"*90)
for m in methods:
    for slug in small:
        ds = DS / slug
        if not (ds/"dirty.csv").exists():
            print(f"{m:16} {slug:44} MISSING"); continue
        t=time.time()
        try:
            r = adapter.run_method(m, ds, slug, timeout=240, force=True)
            err = "OK" if not r.error else r.error[:30]
            print(f"{m:16} {slug[:44]:44} {err:>6} {len(r.cells):>5} {time.time()-t:6.1f}", flush=True)
        except Exception as e:
            print(f"{m:16} {slug[:44]:44} EXC {type(e).__name__}:{e}", flush=True)
