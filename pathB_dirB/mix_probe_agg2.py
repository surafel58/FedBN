"""PAIRED aggregation of mix_probe_raw.json.

The mix at every λ and FedAvg(λ=0) are evaluated on the SAME model per seed, so
seed-level model-quality variance (which is huge for worst-client) is shared and
must be cancelled by a PAIRED comparison: per seed compute Δ = worst(λ) − worst(λ=0),
then report mean±std of Δ over seeds. A genuine SHRINKAGE win needs an INTERIOR λ
(0<λ<1) whose paired Δ vs FedAvg is (a) robustly > 0 (lower band > 0) AND (b) >
the λ=1 naive paired Δ (i.e. shrinkage beats pure-local too).
"""

import json, os, sys, statistics as st
from collections import defaultdict

raw = json.load(open(sys.argv[1] if len(sys.argv) > 1
                     else os.path.expanduser("~/fedbn-experiment/mix_probe_raw.json")))


def ms(xs):
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)


# per-seed FedAvg (λ=0, model-only, regime/k-independent)
fed_seed = {}
by = defaultdict(dict)            # (regime,k,lam) -> {seed: worst}
ref = defaultdict(dict)
for r in raw:
    if r["tag"] == "mix":
        by[(r["regime"], str(r["k"]), str(r["lam"]))][r["seed"]] = r["worst"]
        if str(r["lam"]) == "0.0":
            fed_seed[r["seed"]] = r["worst"]
    else:
        ref[r["tag"]][r["seed"]] = r["worst"]

seeds = sorted(fed_seed)
fm, fs = ms([fed_seed[s] for s in seeds])
print(f"=== seeds={len(seeds)}  PAIRED Δworst-client vs FedAvg(λ0)  "
      f"(FedAvg abs = {fm:.2f} ± {fs:.2f}, high seed-variance -> paired needed) ===\n")
for t in ref:
    m, s = ms([ref[t][x] for x in ref[t]])
    print(f"  [ref] {t:22} abs {m:6.2f} ± {s:4.2f}")
print()


def deltas(regime, k, lam):
    d = [by[(regime, k, lam)][s] - fed_seed[s] for s in seeds if s in by[(regime, k, lam)]]
    return d


ks_order = ["4", "8", "16", "32", "64", "128", "256", "full"]
passes = []
for rg in [r for r in ["A", "B"] if any(k[0] == r for k in by)]:
    name = {"A": "SCARCE (random k)", "B": "BIASED (2-class k)"}[rg]
    print(f"########## Regime {rg}: {name}  —  Δ vs FedAvg (paired, mean±std) ##########")
    print(f"{'k':>5} {'Δnaive(λ1)':>15} {'Δbest-interior':>17} {'λi':>4} "
          f"{'int>0 robust':>13} {'int>naive':>11}")
    for k in [k for k in ks_order if any(kk == k and r == rg for (r, kk, l) in by)]:
        nd = deltas(rg, k, "1.0"); nm, ns = ms(nd)
        lam_d = {}
        for (r, kk, l), _ in by.items():
            if r == rg and kk == k and 0.0 < float(l) < 1.0:
                lam_d[l] = ms(deltas(rg, k, l))
        bi = max(lam_d, key=lambda l: lam_d[l][0])
        bm, bs = lam_d[bi]
        rob = (bm - bs) > 0          # consistently positive across seeds
        gtn = bm > nm                # interior beats pure-local (λ=1)
        if rob and gtn and bm > 0:
            passes.append((rg, k, bm, bs, bi))
        print(f"{k:>5} {nm:+6.2f}±{ns:<6.2f} {bm:+6.2f}±{bs:<7.2f} {bi:>4} "
              f"{('YES' if rob else 'no'):>13} {('YES' if gtn else 'no'):>11}")
    print()

print("================ VERDICT (paired) ================")
if passes:
    print("PASS — interior-λ shrinkage robustly improves worst-client over FedAvg (Δ>0,")
    print("       lower-band>0) AND beats pure-local (λ=1), i.e. genuine shrinkage headroom:")
    for rg, k, bm, bs, l in passes:
        print(f"   regime {rg}, k={k}: Δ=+{bm:.2f}±{bs:.2f} vs FedAvg  (λ*={l})")
    print("   -> build the shrinkage estimator (most promising: see the regime/k above).")
else:
    print("FAIL — no interior-λ shrinkage gives a robustly-positive paired Δ over FedAvg")
    print("       while also beating pure-local recompute. Shrinkage has no clear headroom.")
