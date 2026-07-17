import sys, time
from pathlib import Path
ADR = Path(__file__).resolve().parent
sys.path.insert(0, str(ADR))
import adapter
DS = ADR.parents[1] / "external/rein/rein-benchmark/datasets"
tables = ["114687617797507cfc62ade2bbd047f093f32002.xlsx_9a3524b0",
          "eaff6e4cbe7d0cf12d25b2e162a7f20dab459f43.xlsx_7afc0506",
          "005c05b7029cb909b653bd2942b5272c5030e130.xlsx_ed8fbac9",
          "047a17db9cd660bcc59982658992245f666403c7.xlsx_e6aa6cbe"]
methods = sys.argv[1:] or ["bigdansing","holistic"]
print(f"{'method':12} {'table':20} {'rules':>5} {'status':>10} {'dets':>5} {'sec':>6}")
print("-"*70)
for m in methods:
    for slug in tables:
        ds = DS/slug
        t=time.time()
        r = adapter.run_method(m, ds, slug, timeout=240, force=True)
        st = "OK" if not r.error else ("crash" if "failed rc" in r.error else r.error[:10])
        print(f"{m:12} {slug[:20]:20} {r.rules:>5} {st:>10} {len(r.cells):>5} {time.time()-t:6.1f}", flush=True)
