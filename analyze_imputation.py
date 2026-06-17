"""Offline derisk: does SIMILARITY-weighted imputation of an unsampled client's
BN running stats approximate that client's OWN fresh stats (the oracle) BETTER
than a simple GLOBAL AVERAGE of recently-sampled clients?

Pure offline analysis over logged trajectories (no federated mechanism built).

Inputs (produced by a FedBN partial-participation run with bn-trace-full=true):
  bn_full/seed{S}/eval_p{pid}.csv : rows "round, <5632 floats>"  -- client pid's
      FRESH stats (running_mean||running_var, all 5 BN layers) under the CURRENT
      GLOBAL weights at that round. Logged for ALL clients every round.
  bn_full/seed{S}/fit.csv         : rows "round,pid" -- who was SAMPLED for fit.

For each UNSAMPLED client i at round t we form three quantities:
  oracle(i,t)      = F[t][i]                         <- TARGET (ground truth)
  global_avg(i,t)  = mean_j F[s_j][j]                <- trivial baseline
  sim(i,t)         = sum_j w_ij F[s_j][j]            <- similarity-weighted
where j ranges over clients sampled in window [t-K+1, t] (donors), s_j is j's
most recent sampling round in that window, and F[s_j][j] is the donor's stats
AS OF its sampling round (what the server would actually have cached).

NO LEAKAGE (the central correctness requirement):
  - i's similarity weights use i's LAST-KNOWN fingerprint F[s_i][i] with
    s_i < t (i was last sampled strictly before t, since it is unsampled at t).
  - oracle F[t][i] is used ONLY as the comparison target -- never to compute
    weights or either estimate. (Asserted below: s_i < t, and i excluded from
    its own donor set.)
"""

import glob
import os
import sys

import numpy as np

BN_ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else
                             "~/fedbn-experiment/bn_full")
SEEDS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                          else ["0", "1"])]
WINDOWS = [1, 3, 5]


def load_seed(seed):
    """Return F[(round,pid)]->vec and sampled[round]->set(pid)."""
    d = os.path.join(BN_ROOT, f"seed{seed}")
    F = {}
    for f in glob.glob(os.path.join(d, "eval_p*.csv")):
        pid = int(os.path.basename(f)[6:-4])
        for line in open(f):
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            rnd = int(parts[0])
            F[(rnd, pid)] = np.array(parts[1:], dtype=np.float64)
    sampled = {}
    for line in open(os.path.join(d, "fit.csv")):
        rnd, pid = line.strip().split(",")
        sampled.setdefault(int(rnd), set()).add(int(pid))
    return F, sampled


def rel_l2(target, est):
    """Relative L2 distance, scale-free."""
    return float(np.linalg.norm(target - est) / (np.linalg.norm(target) + 1e-12))


def sim_weights(i_fp, donor_fps):
    """Gaussian-kernel weights from STANDARDISED fingerprints (median-heuristic
    bandwidth). Standardisation (per-dim z-score over the candidate set) stops
    a few high-variance dims from dominating; weights are only used for mixing,
    the mixed VALUES stay in raw stat space."""
    X = np.vstack([i_fp] + donor_fps)              # (1+m, D)
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    Xs = (X - mu) / sd
    iz, dz = Xs[0], Xs[1:]
    dists = np.linalg.norm(dz - iz[None, :], axis=1)   # i vs each donor
    # median-heuristic bandwidth from pairwise donor distances
    if len(dz) >= 2:
        pw = [np.linalg.norm(dz[a] - dz[b])
              for a in range(len(dz)) for b in range(a + 1, len(dz))]
        sigma = np.median(pw) + 1e-8
    else:
        sigma = 1.0
    w = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    if w.sum() <= 0:
        w = np.ones_like(w)
    return w / w.sum()


def analyze(F, sampled, K):
    rounds = sorted({r for (r, _) in F})
    pids = sorted({p for (_, p) in F})
    recs = []
    for t in rounds:
        donors_all = {}                 # j -> most recent sampling round <= t in window
        for r in range(max(min(rounds), t - K + 1), t + 1):
            for j in sampled.get(r, ()):
                donors_all[j] = r       # later r overwrites -> most recent
        for i in pids:
            if i in sampled.get(t, set()):
                continue                # only impute for UNSAMPLED clients
            # i's last-known fingerprint: most recent sampling round < t
            prior = [r for r in sampled if r < t and i in sampled[r]]
            if not prior:
                continue                # no fingerprint yet -> skip
            s_i = max(prior)
            assert s_i < t              # NO LEAKAGE: never uses F[t][i]
            if (s_i, i) not in F or (t, i) not in F:
                continue
            donors = {j: s for j, s in donors_all.items() if j != i}  # exclude i
            if not donors:
                continue
            oracle = F[(t, i)]                                  # TARGET only
            donor_vals = [F[(s, j)] for j, s in donors.items() if (s, j) in F]
            donor_fps = donor_vals                              # donor fingerprint = its sampled stats
            if not donor_vals:
                continue
            global_avg = np.mean(donor_vals, axis=0)
            w = sim_weights(F[(s_i, i)], donor_fps)             # uses i's STALE fp
            sim_est = np.tensordot(w, np.vstack(donor_vals), axes=1)
            ent = float(-np.sum(w * np.log(w + 1e-12)))
            recs.append(dict(
                d_global=rel_l2(oracle, global_avg),
                d_sim=rel_l2(oracle, sim_est),
                d_staleself=rel_l2(oracle, F[(s_i, i)]),
                n_donors=len(donor_vals),
                w_entropy_ratio=ent / (np.log(len(donor_vals)) + 1e-12)
                if len(donor_vals) > 1 else 1.0,
                w_max=float(w.max()),
            ))
    return recs


def summarize(recs):
    dg = np.array([r["d_global"] for r in recs])
    ds = np.array([r["d_sim"] for r in recs])
    dself = np.array([r["d_staleself"] for r in recs])
    win = float(np.mean(ds < dg))
    return dict(
        n=len(recs),
        d_global=(dg.mean(), dg.std()),
        d_sim=(ds.mean(), ds.std()),
        d_staleself=dself.mean(),
        sim_win_frac=win,
        improve_pct=100 * (dg.mean() - ds.mean()) / (dg.mean() + 1e-12),
        w_entropy_ratio=np.mean([r["w_entropy_ratio"] for r in recs]),
        w_max=np.mean([r["w_max"] for r in recs]),
        avg_donors=np.mean([r["n_donors"] for r in recs]),
    )


if __name__ == "__main__":
    all_by_K = {K: [] for K in WINDOWS}
    for seed in SEEDS:
        F, sampled = load_seed(seed)
        for K in WINDOWS:
            all_by_K[K] += analyze(F, sampled, K)
    print(f"seeds={SEEDS}  (dist = relative-L2 to oracle; lower = better)\n")
    hdr = (f"{'K':>2} {'n':>5} {'dist_global':>18} {'dist_sim':>18} "
           f"{'sim<glob %':>10} {'improve%':>9} {'wEntRatio':>10} "
           f"{'wMax':>6} {'donors':>7} {'stale_self':>10}")
    print(hdr)
    for K in WINDOWS:
        s = summarize(all_by_K[K])
        print(f"{K:>2} {s['n']:>5} "
              f"{s['d_global'][0]:>8.4f}±{s['d_global'][1]:<8.4f} "
              f"{s['d_sim'][0]:>8.4f}±{s['d_sim'][1]:<8.4f} "
              f"{100*s['sim_win_frac']:>9.1f} {s['improve_pct']:>8.1f} "
              f"{s['w_entropy_ratio']:>10.3f} {s['w_max']:>6.2f} "
              f"{s['avg_donors']:>7.1f} {s['d_staleself']:>10.4f}")
    print("\nReading: wEntRatio=1.0 -> uniform weights (similarity adds nothing); "
          "lower -> more selective. improve% = how much closer sim is to oracle "
          "than global-avg. stale_self = dist from oracle to i's OWN stale stats "
          "(context).")
