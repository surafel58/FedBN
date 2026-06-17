# Phase 2, Step 2-derisk-B — does TEST-TIME BN recompute recover the collapse (deployably)?

**Confirmation/diagnostic only — NO novel mechanism built.** Server-side similarity imputation was ruled
out (derisk-A: a client's correct BN stats can't be imputed from others). But the oracle's power comes from
each client using its OWN fresh stats, and at INFERENCE a client legitimately has its own data/device. So
"recompute your own BN running stats on your own data before predicting" (**test-time / transductive BN
adaptation**) is deployable at inference. This run confirms whether that recovers the worst-client gap, and
how much own-data it needs.

> **Honesty / novelty:** test-time BN adaptation is a KNOWN technique — this is **not** a novel mechanism.
> The potential contribution is the **finding/characterization**: that this known, deployable test-time
> recompute resolves FedBN's documented partial-participation collapse that no training-time method (and no
> cross-client imputation, derisk-A) addresses — and that it is cheap (tens of samples suffice).

Setup: Flower FedBN baseline, Digits-Five, 20 clients, **10% participation**, 30 rounds, **5 seeds**.
"Test-time recompute" = at evaluation each client resets BN, does a `momentum=None`, `no_grad`,
`train()`-mode forward pass over its OWN data (inputs only, **no labels**), then predicts on its test set.
Variants by adaptation budget: **1 batch (32 samples)**, **4 batches (128)**, **full partition (≈oracle)**.

**No test leakage (verified in code):** recompute iterates `self.trainloader` (own train split), uses only
`data[0]` inputs (labels never touched), `no_grad`, BN-update mode; `test()` runs in `eval()`+`no_grad`
(does not update stats) and is the ONLY use of the test set (accuracy scoring).

## Results — mean ± std over 5 seeds (10% participation, 30 rounds)

| Config | Overall | **Worst-client** | Δ worst vs FedAvg |
|---|---|---|---|
| FedBN baseline (no adaptation) | 81.45 ± 2.30 | 29.37 ± **22.02** | — |
| FedAvg | 81.02 ± 3.75 | 63.18 ± **10.95** | — (bar) |
| **Test-time BN, 32 samples** | 84.23 ± 1.12 | 73.62 ± **1.25** | **+10.44** |
| **Test-time BN, 128 samples** | 86.19 ± 0.50 | 75.23 ± **2.00** | **+12.05** |
| **Test-time BN, full (≈oracle)** | 84.94 ± 0.98 | 77.07 ± **1.42** | **+13.89** |

Recovery vs full recompute: 32 samples = −3.4, 128 samples = −1.8 (of worst-client).

## VERDICT
**Yes — test-time BN recompute recovers the partial-participation collapse, and it is deployable.**
- **Full recompute reaches oracle level** (77.07 ± 1.42), matching Step 1c's oracle (77.19 ± 1.13) — confirming
  the Step-1c "oracle" was always just test-time recompute on a client's own data (deployable at inference).
- **It robustly beats FedAvg** on worst-client: +10.4 (32 samples) to +13.9 (full) on the mean, and — crucially —
  with **far lower variance** (test-time std ~1–2 vs FedAvg 10.95, baseline 22.02). Every seed's test-time
  worst-client sits ~72–77; FedAvg swings 52–74. So test-time BN is reliably ≥ FedAvg and dramatically more stable.
  (Caveat: FedAvg's huge variance means its luckiest seeds reach ~74, so 1-std bands touch — the robustness is in
  test-time BN's *consistency*, not a fully separated band against high-variance FedAvg.)
- **Cheap adaptation already works:** just **32 samples** lifts worst-client from 29 → 73.6 (≈ FedAvg, far more
  stable); **128 samples** → 75.2 (within ~1.8 of the oracle). More own-data → closer to the 77 ceiling.

## Deployability — stated plainly
Each client, at inference, performs **one extra forward pass over a small amount of its OWN data (32–128
samples shown sufficient; full ≈ a few hundred for the ceiling) to refresh BN running stats before
predicting.** It is cheap, on-device, and **label-free** (BN stats use inputs only). It IS an extra
inference-time step, and it assumes the client has some of its own (unlabeled) data available at inference —
reasonable for cross-device, where the device generates its own data. It changes nothing about training or
the communicated model; it is purely a local inference-time statistic refresh.

## Where this leaves Phase 2 (the honest picture across derisks)
- **The collapse is real** (Step 1: worst-client 32 vs FedAvg ~65; driver = stale running stats).
- **A personalization ceiling exists** (Step 1c oracle: 77 worst-client, +12–14 over FedAvg).
- **It is NOT reachable by a training-time / server-side mechanism** that imputes from other clients
  (derisk-A: cross-client similarity ≈ global average, both ~0.22 from oracle).
- **It IS reachable, deployably, by test-time BN adaptation** (this run) — but that is a KNOWN technique.

So the defensible contribution is **a characterization, not a new method**: "FedBN's partial-participation
worst-client collapse is a *test-time statistics* problem, not a training/aggregation problem — it is left
unsolved by training-time fixes and cross-client imputation, but resolved cheaply and deployably by client-
local test-time BN recompute (≥FedAvg from 32 samples; oracle-level from full)." If a *novel* method is still
wanted, the open room is elsewhere (e.g. reaching the 77 ceiling with even less data, or a setting where
clients lack inference-time data) — not in re-deriving test-time BN.

## Artifacts / reproduce
- `phase2_step2deriskB/histories/s{0..4}_{base,fedavg,tt1,tt4,oracle}.pkl`, `aggregate_ttbn.py`,
  `phase2_step2deriskB_results.txt`, `phase2_step2deriskB_testtimebn.patch` (the `adapt-max-batches` knob).
- Run one cell: `flwr run . --run-config "bn-local-mode='all' oracle-stats=true adapt-max-batches=N seed=K fraction-fit=0.1 eval-every=30 num-server-rounds=30"`
  (N=1→32 samples, 4→128, 0→full; FedAvg = `bn-local-mode='none' oracle-stats=false`).

## Caveats
5 seeds, short scale (30 rounds, 20 clients, 10% participation). FedAvg is high-variance at this scale, so the
"beats FedAvg" claim rests on test-time BN's much higher mean AND much lower variance rather than fully
separated 1-std bands. No method built; test-time BN adaptation is a known technique reported here as a
characterization of the FedBN partial-participation problem.
