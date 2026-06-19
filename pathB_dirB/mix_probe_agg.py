"""Aggregate mix_probe_raw.json -> the Path-B Direction-B verdict tables.

For each (regime, k): FedAvg(λ=0) | naive recompute(λ=1) | best-λ* mix | λ*(k) |
best INTERIOR-λ mix (0<λ<1) | does interior shrinkage beat FedAvg robustly?
Plus the λ*(k) shrinkage signature and the FedBN reference lines.
All numbers are worst-client mean±std over seeds.

KEY DISTINCTION (the gate the user specified): a λ*=1.0 "win" is just NAIVE
test-time recompute beating FedAvg (known; not a shrinkage contribution). A true
SHRINKAGE win requires an INTERIOR λ* (0<λ*<1) that beats FedAvg robustly AND
beats both endpoints (pure-global and pure-local). Only that counts as PASS.
"""

import json, os, sys, statistics as st
from collections import defaultdict

raw = json.load(open(sys.argv[1] if len(sys.argv) > 1
                     else os.path.expanduser("~/fedbn-experiment/mix_probe_raw.json")))


def ms(xs):
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)


W = defaultdict(list); O = defaultdict(list)
for r in raw:
    W[(r["tag"], r["regime"], str(r["k"]), str(r["lam"]))].append(r["worst"])
    O[(r["tag"], r["regime"], str(r["k"]), str(r["lam"]))].append(r["overall"])

fed = [v for (t, rg, k, l), vs in W.items() if t == "mix" and l == "0.0" for v in vs]
fedm, feds = ms(fed)
nseed = max(len(v) for v in W.values())
print(f"=== seeds={nseed}  worst-client mean±std  (FedAvg bar = {fedm:.2f} ± {feds:.2f}) ===\n")

for tag in ["fedbn_norecompute", "fedbn_fullrecompute"]:
    ks = [k for (t, rg, k, l) in W if t == tag]
    if ks:
        key = (tag, "-", ks[0], "-"); m, s = ms(W[key])
        print(f"  [ref] {tag:22} worst {m:6.2f} ± {s:4.2f}")
print()

regimes = sorted({rg for (t, rg, k, l) in W if t == "mix" and rg != "-"})
ks_order = ["4", "8", "16", "32", "64", "128", "256", "full"]
shrink_pass = []          # genuine interior-λ shrinkage wins (robust)
naive_pass = []           # λ*=1 wins (naive, not shrinkage)
for rg in regimes:
    name = {"A": "SCARCE (random k)", "B": "BIASED (2-class k)"}.get(rg, rg)
    print(f"########## Regime {rg}: {name} ##########")
    print(f"{'k':>5} {'naive(λ1)':>13} {'best-λ*':>14} {'λ*':>4} "
          f"{'best-INTERIOR':>16} {'λi':>4} {'interior beats':>16}")
    ks_here = [k for k in ks_order if any(t == "mix" and r == rg and kk == k
                                          for (t, r, kk, l) in W)]
    for k in ks_here:
        lam_means = {l: ms(vs) for (t, r, kk, l), vs in W.items()
                     if t == "mix" and r == rg and kk == k}
        nm, ns = lam_means.get("1.0", (float("nan"), 0))
        best_l = max(lam_means, key=lambda l: lam_means[l][0])
        bm, bs = lam_means[best_l]
        # interior = strictly between 0 and 1
        interior = {l: v for l, v in lam_means.items() if 0.0 < float(l) < 1.0}
        bi_l = max(interior, key=lambda l: interior[l][0]) if interior else None
        if bi_l is not None:
            bim, bis = interior[bi_l]
            end0 = lam_means["0.0"][0]; end1 = lam_means["1.0"][0]
            beats_fed = bim > fedm
            robust = beats_fed and (bim - bis) > (fedm + feds)
            beats_ends = bim > max(end0, end1)              # shrinkage > both endpoints
            tag = ("SHRINK-robust" if (robust and beats_ends)
                   else ("shrink>FedAvg" if beats_fed else ("shrink>ends" if beats_ends else "no")))
            if robust and beats_ends:
                shrink_pass.append((rg, k, bim, bi_l))
            istr = f"{bim:6.2f}±{bis:<5.2f}"
        else:
            bi_l, istr, tag = "-", "      -      ", "-"
        if best_l == "1.0" and bm > fedm and (bm - bs) > (fedm + feds):
            naive_pass.append((rg, k, bm))
        print(f"{k:>5} {nm:6.2f}±{ns:<5.2f} {bm:6.2f}±{bs:<5.2f} {best_l:>4} "
              f"{istr:>16} {str(bi_l):>4} {tag:>16}")
    sig = []
    for k in ks_here:
        lam_means = {l: ms(vs)[0] for (t, r, kk, l), vs in W.items()
                     if t == "mix" and r == rg and kk == k}
        sig.append(f"{k}:{max(lam_means, key=lam_means.get)}")
    print("  λ*(k) signature (incl. endpoints):", "  ".join(sig), "\n")

print("================ VERDICT ================")
if shrink_pass:
    print("PASS — genuine INTERIOR-λ shrinkage beats FedAvg ROBUSTLY and beats both endpoints:")
    for rg, k, m, l in shrink_pass:
        print(f"   regime {rg}, k={k}: interior mix {m:.2f} (λ={l}) vs FedAvg {fedm:.2f}")
    print("   -> shrinkage has headroom beyond naive recompute -> build the estimator.")
else:
    print("FAIL (for shrinkage) — no INTERIOR-λ shrinkage beats FedAvg robustly while also")
    print("   beating both endpoints. A shrinkage estimator would not clear FedAvg robustly.")
    if naive_pass:
        print("   (Note: λ*=1 'wins' exist = NAIVE test-time recompute beating FedAvg at higher k —")
        print("    a known result, NOT a shrinkage contribution:")
        for rg, k, m in naive_pass:
            print(f"      regime {rg}, k={k}: naive {m:.2f} > FedAvg {fedm:.2f})")
