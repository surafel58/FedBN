"""Aggregate align_raw.json -> per-config mean±std, FULL per-seed table, and a
PASS/FAIL that is PAIRED and HEALTHY-SEED-scrutinized (a single FedAvg-collapse
seed must not prop up the verdict, as happened in the shrinkage gate).
"""

import json, os, sys, statistics as st
from collections import defaultdict

raw = json.load(open(sys.argv[1] if len(sys.argv) > 1
                     else os.path.expanduser("~/fedbn-experiment/align_raw.json")))
HEALTHY = 55.0   # FedAvg worst-client above this = a "healthy" (non-collapsed) seed


def ms(xs):
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)


seeds = sorted({r["seed"] for r in raw})
cfgs = []
for r in raw:
    if r["cfg"] not in cfgs:
        cfgs.append(r["cfg"])
W = {(r["cfg"], r["seed"]): r["worst"] for r in raw}
O = {(r["cfg"], r["seed"]): r["overall"] for r in raw}

print(f"=== seeds={len(seeds)}  worst-client (and overall) mean±std ===\n")
for c in cfgs:
    w = [W[(c, s)] for s in seeds if (c, s) in W]
    o = [O[(c, s)] for s in seeds if (c, s) in O]
    wm, ws = ms(w); om, oo = ms(o)
    print(f"  {c:20} worst {wm:6.2f} ± {ws:5.2f}    overall {om:6.2f} ± {oo:5.2f}")

print("\n=== FULL per-seed worst-client table ===")
print("seed  " + "  ".join(f"{c:>18}" for c in cfgs))
for s in seeds:
    print(f"{s:>4}  " + "  ".join(f"{W.get((c, s), float('nan')):>18.2f}" for c in cfgs))

fed = "FedAvg(a0)"
healthy = [s for s in seeds if W.get((fed, s), 0) >= HEALTHY]
collapsed = [s for s in seeds if s not in healthy]
print(f"\nHealthy seeds (FedAvg worst ≥ {HEALTHY}): {healthy}   collapsed: {collapsed}")

aligns = [c for c in cfgs if c.startswith("align")]
print("\n=== PAIRED Δ vs FedAvg (per-seed worst-client) ===")
print(f"{'config':14} " + "  ".join(f"s{s}" for s in seeds) +
      "   | mean(all)  mean(healthy)  #healthy-won")
best = None
for c in aligns:
    d = {s: W[(c, s)] - W[(fed, s)] for s in seeds if (c, s) in W}
    dh = [d[s] for s in healthy]
    hm, hs = ms(dh) if dh else (float("nan"), 0)
    won = sum(1 for s in healthy if d[s] > 0)
    allm = ms(list(d.values()))[0]
    print(f"{c:14} " + "  ".join(f"{d[s]:+5.1f}" for s in seeds) +
          f"   | {allm:+7.2f}   {hm:+7.2f}±{hs:.2f}   {won}/{len(healthy)}")
    robust = dh and (hm - hs) > 0 and won == len(healthy)
    if robust and (best is None or hm > best[1]):
        best = (c, hm)

print("\n================ VERDICT ================")
if best:
    print(f"PASS — {best[0]} beats FedAvg worst-client on ALL healthy seeds "
          f"(mean Δ_healthy=+{best[1]:.2f}, lower band > 0).")
    print("   -> the features lever has headroom; a deployable alignment method is justified.")
else:
    print("FAIL — no alignment α beats FedAvg robustly on the HEALTHY seeds")
    print("   (either Δ_healthy not consistently >0, or any apparent win is on collapsed seeds).")
    print("   Even this aggressive alignment CEILING does not lift shared-stats above FedAvg.")
