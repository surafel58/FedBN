# Phase 1 — Reproduce FedBN & confirm the partial-participation gap

**Goal (Phase 1 only):** reproduce FedBN, confirm it beats FedAvg under full-participation
feature shift, and confirm the *documented* failure of FedBN under cross-device **partial
participation** is real and runnable — **before building anything**.

**Verdict: confirmed on all three counts.** FedBN ≥ FedAvg at full participation, and FedBN
**collapses** under 10%-per-round client sampling (worst-client accuracy falls to near-random),
while FedAvg is barely affected. The gap is real and runnable in our hands.

---

## TL;DR results

### Step 3 — Full participation, raw `med-air/FedBN` repo (cross-silo, 5 clients, 50 rounds)
FedBN beats FedAvg on **every** domain; biggest gains on the most feature-shifted domains.

| Domain | FedBN | FedAvg | Δ |
|---|---|---|---|
| MNIST | 96.20 | 95.19 | +1.01 |
| SVHN | 69.28 | 62.53 | **+6.75** |
| USPS | 96.83 | 95.43 | +1.40 |
| SynthDigits | 81.14 | 80.03 | +1.11 |
| MNIST-M | 76.69 | 70.80 | **+5.89** |
| **Average** | **84.03** | **80.80** | **+3.23** |

Paper reports avg FedBN 88.4 / FedAvg 86.5. Our absolutes run ~4 pts low only because this was a
50-round smoke (paper uses 100); the qualitative claim reproduces exactly.

### Step 4 — Partial participation, Flower FedBN baseline (20 clients = 4/domain, 30 rounds)
All-client federated evaluation; "worst-client" = lowest single-client test accuracy of the 20.

| Setting | Overall acc | Worst-client |
|---|---|---|
| FedBN, full participation (fraction-fit=1.0) | **88.75** | **79.55** |
| FedBN, **10% participation** (fraction-fit=0.1) | **71.99** | **12.43** |
| FedAvg, full participation | 87.54 | 75.35 |
| FedAvg, **10% participation** | 86.51 | 71.76 |

**Reading the table:**
- **Full participation:** FedBN ≥ FedAvg (overall +1.2, worst-client +4.2). Paper claim holds.
- **Going full → partial:** FedBN drops **−16.8** overall and **−67.1** worst-client (collapse).
  FedAvg drops only **−1.0** / **−3.6** (robust).
- **Under partial participation the ranking REVERSES:** FedAvg (86.5) now beats FedBN (72.0), and
  the worst-client gap is enormous — FedAvg 71.8 vs FedBN **12.4** (≈ random for 10-class).

---

## Why FedBN collapses (mechanism, confirmed in code)
FedBN keeps each client's **BatchNorm parameters local** — they are *never* sent to / averaged by
the server. In Flower this is implemented by persisting the BN state in each client's
`context.state` and reloading it whenever that client is next sampled
(`fedbn/client_app.py: FedBNFlowerClient`).

Under 10%-per-round sampling, most clients are selected only rarely, so their local BN statistics
go **stale** relative to the global feature extractor (the non-BN weights) that keeps drifting
every round. At evaluation, a rarely/never-sampled client pairs the *current* global weights with
its *outdated* (or initial mean=0/var=1) BN stats → mismatch → near-random output. That is exactly
the documented cross-device failure. FedAvg has no local state: every client — sampled or not — is
handed one coherent global model (BN included), so it degrades only mildly.

---

## What was run, where, and why two codebases

| | Step 3 (full participation) | Step 4 (partial participation) |
|---|---|---|
| Codebase | `med-air/FedBN` (original ICLR'21 repo) | Flower official `baselines/fedbn` |
| Why | The headline reproduction | Original repo is **hard-wired to full participation** (no client sampling); Flower's baseline has the *same* model/data **plus native client sampling** (`fraction_fit`) |
| Model | 6-layer CNN, 14,219,210 params | identical CNN, 14,219,210 params |
| Data | Digits-Five (MNIST/SVHN/USPS/SynthDigits/MNIST-M), authors' pre-processed partitions | same |

Both use the same pre-processed `digit_dataset.zip` from the FedBN authors.

## Harness edits made to the Flower baseline (Step 4)
All are instrumentation/efficiency only — **no change to FL algorithms**. Full diff in
`FedBN/flower_fedbn_step4.patch` (also `phase1_artifacts/`).
1. `strategy.py` — aggregator also emits `overall_accuracy`, `min_client_accuracy`,
   `num_clients_reported` (needed for worst-client).
2. `dataset.py` — per-worker cache of built dataloaders (the baseline rebuilt *all* clients'
   loaders on *every* `client_fn` call → 16 GB host OOM at scale; cache fixes it).
3. `server_app.py` — `PeriodicEvalFedAvg`: run the (expensive) all-client evaluation every 5
   rounds + final round only. Training untouched.
4. `pyproject.toml` — 20 clients, GPU client-resources, `fraction-evaluate=1.0`, `eval-every=5`.
   Also pinned `click==8.1.7` (flwr 1.17 CLI is incompatible with click 8.4).

---

## Caveats / honesty notes
- **20 clients, not ~100.** The Flower baseline caps at 50 clients (a multiple of the 5 datasets),
  and this 16 GB / L4 node OOMs above ~20 with all-client evaluation. We kept the scientifically
  decisive quantity — the **10% participation fraction** (2 of 20 sampled/round) — which is what
  drives BN staleness; the absolute client count does not change the mechanism. Lifting to 100
  would need a bigger box and a baseline edit (Phase 2 if wanted).
- **Smoke scale, not paper scale:** Step 3 = 50 rounds (paper 100); Step 4 = 30 rounds, 1 seed.
  Trends are large and unambiguous; absolute numbers are indicative, not final.
- **FedWon not included** — it has no public code repo (Sony AI). Optional in the brief.
- Step 4 used Flower's standard federated evaluation (per-client test sets per domain); Step 3 used
  the original repo's own eval loop. Numbers across the two steps are therefore not 1:1 comparable
  (different harnesses), but each is internally consistent.

## Compute
GCP `fl-experiment` VM (g2-standard-4, **NVIDIA L4 23 GB**, 16 GB RAM), zone
`northamerica-northeast2-b`. Step 3 ≈ 38 min (both modes concurrent); Step 4 matrix ≈ 60 min
(4 cells sequential, 2 clients in parallel on GPU). **VM stopped after the runs.**

## Reproduce
- Original full-participation: `cd FedBN/federated && python fed_digits.py --mode fedbn|fedavg --iters 50`
- Flower partial-participation (after `pip install -e .` of the baseline + `pip install click==8.1.7`,
  applying `flower_fedbn_step4.patch`):
  `flwr run . --run-config "algorithm-name='FedBN' fraction-fit=0.1 num-server-rounds=30"`
  (full participation = `fraction-fit=1.0`; FedAvg = `algorithm-name='FedAvg'`).
- Artifacts: `phase1_artifacts/` (4 result histories `step4_histories/*.pkl`, raw logs,
  `parse_history.py`, the patch).

## Status / next
Phase 1 complete. **Do not** build a method yet (per brief). Phase-2 candidates if we proceed:
scale clients toward 100 on a larger node, add seeds/rounds for publication-grade numbers, and
adopt the per-client metric + harness patterns from our existing `fl_experiment`.
