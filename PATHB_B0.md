# Path B, Step B0 — headroom gate for the STRICT zero-inference-data setting

**Diagnostic only — no mechanism built.** Decisive question: in a setting where clients have **no data at
inference** (so test-time BN recompute — which worked in derisk-B — is forbidden), is there ANY room above
FedAvg's worst-client, or is FedAvg already the ceiling?

The earlier Step-1c "oracle" recomputed each client's stats **at the eval round, under the current/final
weights** — exactly the forbidden test-time capability. To test the strict setting honestly, the generous
ceiling (#3) must compute ideal stats **once at a reference round, FREEZE them**, let the global model keep
training, and evaluate the **drifted final model** with those frozen stats and **no recompute**. We sweep the
freeze round to see whether the advantage survives staleness or collapses toward FedAvg.

Setup: Flower FedBN, Digits-Five, 20 clients, **10% participation**, 30 rounds, **5 seeds**. All configs
evaluated with **no per-client inference recompute** (except where noted). #3's recompute is the deliberate
"cheat" that obtains ideal stats — it uses client data, which the strict setting forbids, hence an upper bound.

## Implementations
- **#1 FedAvg** `bn-local-mode=none`: global model + global aggregated stats; no recompute.
- **#2 FedBN, no recompute** `bn-local-mode=all, oracle off`: each client's stale last-known local BN; no recompute.
- **#4 Shared-global-stats** `bn-local-mode=affine`: one globally-aggregated running-stats vector for all; γ/β local; no per-client recompute. (The Candidate-3 ceiling: shared stats without changing features.)
- **#3 frozen@R** `freeze-round=R`: at round R each client computes its ideal stats (oracle recompute on its
  own data), **freezes** them; training continues to round 30; the round-30 model is scored with the
  **frozen-at-R** stats and **no recompute** (scoring only at the final round; test set used only to score).
  `frozen@30` = compute on the final model = zero staleness (≡ test-time recompute on the deployed model = the
  *forbidden* capability); `frozen@5/@15` = genuinely frozen + stale.

## Results — mean ± std over 5 seeds (worst-client is the deciding metric)

| Config | Overall | **Worst-client** | Δ worst vs FedAvg |
|---|---|---|---|
| #2 FedBN, no recompute (stale local) | 81.75 ± 2.99 | 29.77 ± 23.34 | −39.2 |
| **#1 FedAvg (the bar)** | 83.91 ± 1.65 | **68.99 ± 2.77** | — |
| #4 Shared-global-stats (Candidate-3 ceiling) | 81.59 ± 5.53 | 64.13 ± 12.84 | **−4.87** |
| #3 frozen@5 (very stale) | 78.78 ± 4.41 | 65.57 ± 4.78 | **−3.42** |
| #3 frozen@15 (mid staleness) | 82.50 ± 1.51 | 70.19 ± 0.87 | **+1.20** |
| #3 frozen@30 (zero staleness ≡ recompute-on-final = FORBIDDEN) | 86.39 ± 0.75 | 76.51 ± 1.18 | +7.52 |

## VERDICT: **NO headroom — Path B via stats is DEAD.**
**Freezing the ideal stats (removing the eval-time recompute) collapses the advantage back to — and below —
FedAvg.** The advantage decays monotonically with staleness: frozen@30 **+7.5** (but this is zero-staleness =
recomputing on the deployed model = the test-time capability the strict setting forbids) → frozen@15 **+1.2**
(within FedAvg's ±2.8 noise) → frozen@5 **−3.4** (below FedAvg). The achievable **shared-stats** ceiling (#4,
Candidate 3) is **−4.9** (below FedAvg, high-variance). So in the strict zero-inference-data setting, **FedAvg
is effectively the ceiling for any stats-based intervention.**

The +12 "headroom" seen in Step 1c was **entirely an artifact of recompute-on-current-weights** (test-time
BN): the ideal stats only help while they track the *current* model; freeze them and let the model drift and
the benefit evaporates. This is the decisive evidence the gate was designed to find.

### Honesty notes
- #3 is a deliberately-optimistic upper bound (it uses client data to compute ideal stats — forbidden in the
  strict setting). Even granting that cheat, once the stats are **frozen** they don't beat FedAvg at realistic
  staleness. A real strict-setting mechanism (no client data ever; per-client stats not imputable, per
  derisk-A) would land at or below these frozen numbers → at or below FedAvg.
- frozen@15's +1.2 is within noise and is itself optimistic (ideal stats for *all* clients incl. unsampled,
  only 15 rounds stale). Not a basis for a method.
- #4 (shared stats) underperforming FedAvg is consistent with it being a degraded, single-vector version of
  what FedAvg already aggregates.

## Implication for direction
Stats are not the lever in the strict setting. If a method is to beat FedAvg with **zero inference data**, it
must change the **features / global representation** (so that the global model + global stats works better
for the worst client), not the BN statistics. That is a different (and harder) research question; whether to
pursue it is the open decision. Otherwise, the honest end-state across Phase 2 is:
- collapse is real (Step 1) and is a stale-running-stats problem;
- a personalization ceiling exists (Step 1c, +12) but is **only reachable by per-client, current-weight stats**;
- that is **unreachable** by server-side imputation (derisk-A) and, in the strict setting, by frozen/shared
  stats (this gate);
- it **is** reachable by client-local **test-time BN recompute** (derisk-B) — a known technique — but only
  when clients have inference data, which the strict setting excludes.

## Artifacts / reproduce
- `pathB_B0/histories/s{0..4}_{fedavg,fbn_nr,shared,fz5,fz15,fz30}.pkl`, `aggregate_strict.py`,
  `pathB_B0_results.txt`, `pathB_B0_freeze.patch` (the `freeze-round` mechanism).
- Run one frozen cell: `flwr run . --run-config "bn-local-mode='all' freeze-round=R seed=K fraction-fit=0.1 fraction-evaluate=1.0 eval-every=R num-server-rounds=30"` (R ∈ {5,15,30}; scores only at round 30 with frozen-at-R stats).

## Caveats
5 seeds, short scale (30 rounds, 20 clients, 10% participation). FedAvg/shared are high-variance at this scale;
the verdict rests on the monotonic staleness decay (frozen@30 +7.5 → @15 +1.2 → @5 −3.4) and that the only
clearly-beats-FedAvg point requires the forbidden recompute-on-deployed-model. No method built.
