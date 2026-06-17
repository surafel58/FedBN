# Phase 2, Step 2-derisk — does similarity-weighted stat imputation beat a global average?

**Offline analysis only — no mechanism built.** Step 1c confirmed an oracle (each client's OWN fresh
BN stats) has large headroom over FedAvg. The proposed mechanism would *impute* an unsampled client's
stats from **similar** recently-sampled clients. Central bet: **similarity-weighted imputation
approximates the client's own fresh stats (oracle) better than a plain global average of recent donors.**
If not, the mechanism collapses to the trivial global-stats config (which only reached FedAvg level).

Setup: FedBN partial-participation trajectory (20 clients, 10% participation, 30 rounds, seeds 0 & 1).
Logged each client's **fresh stats under current global weights** every round + the fit-sampling sequence.
Offline, for each **unsampled** client i at round t, compared two estimates of its stats against the
oracle target F[t][i]: (2) global-average of donors, (3) similarity-weighted (Gaussian kernel on
z-scored running_mean‖running_var fingerprints, donors = clients sampled in last K rounds).

**No leakage** (verified in code): i's similarity uses only its LAST-KNOWN fingerprint F[s_i][i] with
s_i < t (asserted); the oracle F[t][i] is the comparison target only; i is excluded from its own donors;
donors contribute their stats *as of their last sampling* (what a server would actually have cached).

## Results (seeds 0,1; n = 739 unsampled (client,round) cases; distance = relative-L2 to oracle, lower better)

| Window K | dist_global | dist_sim | sim closer than global | improve % | weight-entropy ratio | avg donors | (oracle vs i's own stale) |
|---|---|---|---|---|---|---|---|
| 1 | 0.235 ± 0.082 | 0.228 ± 0.087 | 89.6% | 3.2% | 0.995 | 2.0 | 0.229 |
| 3 | 0.225 ± 0.054 | 0.213 ± 0.056 | 94.2% | 5.3% | 0.994 | 5.0 | 0.229 |
| 5 | 0.230 ± 0.050 | 0.217 ± 0.051 | 98.4% | 5.4% | 0.994 | 7.6 | 0.229 |

(weight-entropy ratio: 1.0 = perfectly uniform weights → similarity adds nothing; lower = selective.)

## VERDICT: **central bet does NOT hold — do NOT build the similarity-imputation mechanism.**
Similarity-weighting wins the head-to-head in ~90–98% of cases, but:
1. **The margin is tiny** — only **3–5% relative** distance reduction (e.g. 0.225 → 0.213 at K=3). Not
   "clearly lower."
2. **The weights are essentially uniform** — entropy ratio **0.994–0.995** (1.0 = uniform). The BN
   fingerprints don't single out genuinely "more similar" donors; similarity ≈ global average by design.
3. **Both estimates are FAR from the oracle** (~0.22 relative-L2, i.e. ~22% error), and **no better than
   the client's own stale stats** (0.229). The oracle's power (Step 1c, 77% worst-client) comes from
   stats that are *exactly right* (distance 0); any cross-client imputation lands ~0.22 away — i.e. near
   the trivial/FedAvg level, nowhere near the oracle.

Larger windows K raise the win-fraction (more donors) but leave the margin (~5%) and near-uniform weights
unchanged → **K does not change the verdict.**

## Interpretation / implied pivot (NOT built)
The Step-1c headroom is real but **not reachable by imputing from other clients' stats**: a client's
fresh BN statistics are determined by ITS OWN data, and other clients' fingerprints carry little
predictive signal about it (near-uniform similarity, ~0.22 floor). Server-side cross-client similarity
imputation would collapse to the global-average / FedAvg level.

The oracle requires a client's own data — which points away from server-side imputation and toward a
**client-local / test-time mechanism**: a client recomputes its OWN BN running stats on its OWN data
before inference (a forward pass in BN-update mode), which is exactly the oracle and **is deployable for
on-device inference** (transductive / test-time BN adaptation), independent of whether it was sampled for
training. That — not cross-client similarity imputation — is the direction the evidence supports. (A
different/learned similarity signal is a possible but unpromising alternative, given the ~0.22 floor.)

## Artifacts / reproduce
- `phase2_step2derisk/analyze_imputation.py` (similarity + no-leakage), `phase2_step2derisk_results.txt`,
  `phase2_step2derisk/sampling/fit_seed{0,1}.csv`, `phase2_step2derisk_trace.patch` (the `bn-trace-full`
  instrumentation).
- Raw per-client fresh-stat vectors (~63 MB, `bn_full/`) are NOT committed; regenerate via
  `flwr run . --run-config "bn-local-mode='all' bn-trace-full=true seed=K fraction-fit=0.1 fraction-evaluate=1.0 eval-every=1 num-server-rounds=30"`
  then `python analyze_imputation.py <bn_full> 0,1`.

## Caveats
2 seeds, short scale, one (sensible) similarity function. The signal is directionally consistent
(sim reliably ~5% better) but too weak and weights too uniform to justify building; the dominant finding
(both imputations sit ~0.22 from the oracle, near FedAvg) is robust to these choices. No mechanism built;
this gate says the cross-client similarity-imputation direction does not clear the trivial fix.
