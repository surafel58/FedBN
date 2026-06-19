# Path B, features-lever — headroom gate (Option 1: feature alignment)

**Diagnostic/ceiling only — no deployable mechanism built.** Deciding question: can reshaping the global
representation (an aggressive feature-alignment objective) make **shared global BN stats beat FedAvg's
worst-client** under partial participation, in the strict no-inference-recompute regime where stats
interventions died (B0: best shared stats trailed FedAvg by −4.9)?

Setup: 20 clients, Digits-Five, 10% participation, 30 rounds, 5 seeds. Alignment loss added to local
training: pull each client's per-channel **BN-input activation stats** (batch mean/var of inputs to bn1..bn5)
toward a COMMON target = the **global BN running stats broadcast with the model** (snapshotted per round; a
real, available quantity — no oracle, no test). Aggressive α sweep {0.1, 1, 10} = the CEILING. **Eval =
shared global stats, NO per-client recompute, no leakage** (α=0 ≡ FedAvg). A deployable method would land
below this ceiling.

## Results — worst-client mean ± std (5 seeds)

| Config | Worst-client | Overall |
|---|---|---|
| **FedAvg (α=0, shared stats)** — the bar | 60.86 ± 16.76 | 80.64 |
| align α=0.1 | 61.21 ± 15.23 | 81.65 |
| align α=1.0 | 60.85 ± 15.34 | 81.26 |
| align α=10.0 | **25.75 ± 21.87** | 34.42 |
| FedBN no-recompute (collapse ref) | 52.66 ± 12.34 | 83.38 |
| oracle recompute (unreachable ref) | 74.89 ± 1.19 | 86.69 |

## FULL per-seed worst-client table
| seed | FedAvg | α=0.1 | α=1.0 | α=10 | FedBN-nr | oracle |
|---|---|---|---|---|---|---|
| 0 | 69.55 | 70.75 | 67.65 | 7.90 | 65.25 | 73.25 |
| 1 | 67.95 | 68.25 | 69.65 | 51.85 | 33.15 | 75.40 |
| 2 | 64.70 | 64.15 | 62.90 | 53.20 | 43.65 | 73.70 |
| 3 | 74.20 | 71.70 | 73.15 | 7.90 | 63.25 | 76.00 |
| 4 | **27.90** (FedAvg collapsed) | 31.20 | 30.90 | 7.90 | 58.00 | 76.10 |

Healthy seeds (FedAvg worst ≥ 55): **{0,1,2,3}**; collapsed: {4}.

## Paired Δ vs FedAvg on HEALTHY seeds (the mandatory scrutiny)
| α | s0 | s1 | s2 | s3 | Δ_healthy mean±std | #healthy won |
|---|---|---|---|---|---|---|
| 0.1 | +1.2 | +0.3 | −0.6 | −2.5 | **−0.39 ± 1.37** | 2/4 |
| 1.0 | −1.9 | +1.7 | −1.8 | −1.0 | **−0.76 ± 1.46** | 1/4 |
| 10.0 | −61.6 | −16.1 | −11.5 | −66.3 | **−38.89 ± 25.19** | 0/4 |

## VERDICT: **FAIL — the features lever (feature alignment) has no headroom.**
- Mild alignment (α=0.1, 1.0) is **neutral on healthy seeds** (Δ ≈ 0, within ±1.5, wins 1–2/4) — it does
  NOT lift shared-stats above FedAvg.
- Strong alignment (α=10) is **catastrophic** (worst-client 7.9 ≈ random on several seeds): forcing all
  domains' activations to a common target destroys class discriminability.
- The only positive deltas are on **seed 4, where FedAvg itself collapsed** (27.9) — the exact single-seed
  artifact the shrinkage gate hit; the healthy-seed scrutiny correctly excludes it.
- This is the generous CEILING (aggressive α); a deployable alignment method would do worse. → dead.

## Interpretation
There is no sweet spot: weak alignment doesn't change the features enough to make one shared stat-set fit
everyone (shared stats stay ≈ FedAvg level), and strong alignment collapses the features (destroys accuracy).
Aligning feature *distributions* across feature-shifted domains trades away the very discriminability the
task needs. The feature-alignment route to "make shared stats beat FedAvg" does not exist here.

## Where this leaves Path B / Phase 2 (complete map)
Every lever has now been gated; **none robustly beats FedAvg without inference data:**
- stats — imputation (derisk-A) ✗ · frozen/shared strict (B0) ✗ · shrinkage (Dir-B) ✗
- features — alignment (this gate) ✗
- the ONLY thing that reliably helps worst-client is **test-time BN recompute when the client has inference
  data** (derisk-B) — a KNOWN technique, outside the strict setting.

**Net finding (robust, multiply-confirmed):** FedBN's partial-participation worst-client collapse is a
*test-time statistics* problem. With scarce/biased/zero inference data, neither stats- nor feature-alignment
interventions beat FedAvg; the deployable fix is the known test-time BN recompute (when inference data
exists). No novel training-time method in these directions clears FedAvg.

## Artifacts / reproduce
- `pathB_features/align_probe.py` (alignment harness), `align_agg.py` (per-seed + healthy-seed verdict),
  `align_full.json`, `pathB_features_results.txt`.
- Run: `python align_probe.py --seeds 0,1,2,3,4 --alphas 0.1,1.0,10.0 --rounds 30` then `python align_agg.py align_full.json`.

## Caveats
5 seeds, short scale; standalone-harness FedAvg seed-variance is high (one collapse seed), handled by the
paired healthy-seed analysis. Tested one alignment family (match BN-input mean/var to broadcast global). The
verdict (neutral at mild α, catastrophic at strong α, no win on healthy seeds) is unambiguous for this family;
a fundamentally different feature objective is a separate question, but Option-1 feature alignment is dead.
