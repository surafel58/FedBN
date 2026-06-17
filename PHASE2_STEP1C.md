# Phase 2, Step 1c — Multi-seed headroom confirmation (the deciding gate)

**Diagnostic only — no method built.** Settles whether a *personalized-fresh-stats* mechanism has room
above FedAvg, using the oracle (its upper bound) across 5 seeds. Step 1b's single seed was inconclusive
(oracle 69.9 < FedAvg 73.4, but configs swung ~10 pts run-to-run). This run resolves it.

Setup: Flower FedBN baseline, Digits-Five, 20 clients, **10% participation**, 30 rounds, **5 seeds (0–4)**.

## Results — mean ± std over 5 seeds (final round, 10% participation)

| Config | Overall acc | **Worst-client acc** |
|---|---|---|
| FedBN baseline (all BN local) | 82.03 ± 2.75 | 32.58 ± **22.57** |
| Stats-shared (global fresh stats, γ/β local) | 81.48 ± 2.94 | 64.16 ± 11.74 |
| **Oracle (personalized fresh stats, γ/β local)** | **86.61 ± 1.27** | **77.19 ± 1.13** |
| FedAvg (all shared) | 82.18 ± 3.10 | 64.93 ± 7.88 |

Key margins:
- **Oracle − FedAvg: overall +4.43, worst-client +12.25**
- Stats-shared − FedBN baseline: worst-client **+31.58** (overall −0.55, within noise)

## VERDICT: **HEADROOM CONFIRMED.**
**The oracle (best-case personalized-fresh stats) robustly beats FedAvg — worst-client 77.19 ± 1.13 vs
64.93 ± 7.88 (+12.25), with non-overlapping spreads (oracle low 76.1 > FedAvg high 72.8), and overall
86.61 vs 82.18 (+4.43). The margin far exceeds the seed noise. A personalized-fresh-stats mechanism is
justified — even an approximate version has ~12 pts of worst-client room above FedAvg to aim at.**

Why this overturns Step 1b: FedAvg's worst-client is **high-variance (±7.88)** and Step 1b drew a lucky
FedAvg / unlucky oracle seed. Across 5 seeds the ranking is stable and clear.

## Three findings worth keeping
1. **Headroom is real and large.** Oracle worst-client (77.2) sits ~12 pts above both FedAvg (64.9) and
   the trivial Stats-shared fix (64.2). The gain comes specifically from **personalizing** the fresh
   stats, not merely refreshing them: global-fresh (Stats-shared, 64.2) ≈ FedAvg (64.9) ≪ personalized-
   fresh (Oracle, 77.2).
2. **The oracle is also dramatically more stable** (worst-client std **1.13** vs FedAvg 7.88, baseline
   22.57). Personalized-fresh stats largely remove the sampling-luck dependence — every client, sampled
   or not, gets statistics matched to its own data. That stability is itself a strong argument for the
   direction.
3. **The trivial fix works but only to FedAvg level.** Stats-shared robustly rescues the collapse vs the
   FedBN baseline (+31.6 worst-client) but lands at ≈ FedAvg, i.e. it does NOT capture the personalization
   gain. So "just share the stats globally" is not the ceiling — personalization is where the contribution is.

## Implication for mechanism design (Phase 2 Step 2 — NOT built here)
Design a **deployable approximation of personalized-fresh BN statistics for unsampled clients** — the
oracle assumes touching every client's data each round, which a real cross-device system cannot. Targets
to approximate that 77% worst-client ceiling:
- similarity/neighbour-based **stat imputation** for clients not sampled this round,
- staleness-aware **stat interpolation** toward a current estimate,
- lightweight client-side **stat recomputation** when a client does participate, cached + decayed.
γ/β can stay personalized (FedBN-style); the lever is the running statistics. The bar to beat is FedAvg
(64.9 worst-client); the ceiling to chase is the oracle (77.2).

## Method note: seeding
The baseline had no seed control. Added a `seed` config that seeds server init / NumPy / Torch, the
per-seed data-partition shuffle, and per-client dataloaders (`phase2_step1c_seeding.patch`). Flower's
simulation retains some residual nondeterminism (client sampling / Ray worker order) that `seed` does not
fully pin — so the 5 runs per config are effectively **independent draws**, which is exactly what an
unpaired mean±std comparison needs. (Same-seed reproducibility is therefore approximate; this does not
affect the variance estimate or the verdict.)

## Artifacts / reproduce
- `phase2_step1c/histories/s{0..4}_{base,stats,oracle,fedavg}.pkl` (20 runs), `aggregate.py`,
  `aggregate_5seed.txt`, `phase2_step1c_seeding.patch`.
- Run one cell: `flwr run . --run-config "bn-local-mode='all' oracle-stats=true seed=K fraction-fit=0.1 eval-every=30 num-server-rounds=30"`
  (Stats-shared = `bn-local-mode='affine' oracle-stats=false`; FedAvg = `bn-local-mode='none'`;
  baseline = `bn-local-mode='all' oracle-stats=false`). `eval-every=30` → evaluate only at the final round.

## Caveats
Short scale (30 rounds, 20 clients, 10% participation) — about settling the *ranking*, not final absolute
numbers. The verdict (oracle ≫ FedAvg on worst-client, non-overlapping) is robust to this scale. No method
was built; this gate authorizes designing one.
