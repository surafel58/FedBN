# Phase 2, Step 1 — BN-component diagnostic (which part of BatchNorm drives FedBN's collapse?)

**Diagnostic only — no new method.** Goal: isolate *which* BatchNorm component's staleness causes
FedBN's worst-client collapse under partial participation, so the eventual fix targets the real cause.

Setup: identical to the Step-4 collapse setup — Flower FedBN baseline, Digits-Five, **20 clients,
10% participation (fraction-fit=0.1), 30 rounds, single seed**, all-client federated evaluation.
Only the **BN sharing rule** changes across variants.

## Variant definitions (confirmed before running)
BatchNorm = **affine** (`weight`=γ, `bias`=β) + **running statistics** (`running_mean`, `running_var`,
`num_batches_tracked`). One predicate `is_bn_local(key, mode)` decides which BN tensors stay local
(personalised, persisted in client state) vs shared (FedAvg-aggregated). conv/fc weights are always shared.

| Variant | mode | BN kept LOCAL | BN SHARED |
|---|---|---|---|
| FedBN baseline | `all` | all 25 BN tensors | none |
| Stats-shared | `affine` | affine γ/β (10) | running stats (15) |
| Affine-shared | `stats` | running stats (15) | affine γ/β (10) |
| FedAvg | `none` | none | all 25 |

## Results (10% participation, 30 rounds)

| Variant | What becomes "fresh" (shared) | Overall acc | **Worst-client** |
|---|---|---|---|
| **FedBN baseline** (all BN local) | — (both stale) | 65.7 | **16.7** |
| **Affine-shared** (share γ/β, stats local) | affine only | 81.9 | 51.7 |
| **FedAvg** (share everything) | both | 82.5 | 63.5 |
| **Stats-shared** (share running stats, γ/β local) | running stats only | 82.6 | **68.6** |

Worst-client ranking: baseline 16.7 ≪ affine-shared 51.7 < FedAvg 63.5 < **stats-shared 68.6**.

## VERDICT (one line)
**Stale BatchNorm running statistics are the primary driver of the collapse: sharing them alone
restores worst-client from 16.7% → 68.6% (≥ FedAvg) while keeping affine personalised; sharing the
affine alone (stats still stale) only partly recovers it (→ 51.7%).**

Notable: **Stats-shared is the best cell overall** — it matches FedAvg's robustness *and* beats its
worst-client (68.6 vs 63.5) while *retaining personalised affine* (FedBN's feature-shift benefit).
That is a strong hint for the mechanism direction: **keep running statistics fresh/global, personalise γ/β.**

## Staleness instrumentation (baseline `all`, made visible)
Per-client norms logged each eval round (`bn_trace/all.csv`). The least-sampled client was **partition 14
(SynthDigits), trained 0× in 30 rounds**; the most-sampled was **partition 8 (USPS), trained 8×**.

| | round → | conv1.weight (GLOBAL) | bn1.running_mean | bn1.running_var | γ | β |
|---|---|---|---|---|---|---|
| **Rarely (p14, 0×)** | 5→30 | 4.641 → 4.661 *(drifts)* | **0.000 (frozen)** | **8.000 (frozen)** | 8.00 | 0.00 |
| **Frequently (p8, 8×)** | 5→30 | 4.641 → 4.661 *(drifts)* | 0.0 → ~1.92 *(tracks data)* | 8.0 → ~1.36 *(tracks data)* | 8.00 | 0→0.005 |

Reading it: the **global feature extractor (conv1.weight) keeps drifting every round for both clients**,
but the rarely-sampled client's **running stats stay frozen at initialisation** (mean 0, var "8" = √64 for
64 unit-variance channels) because it is never retrained → its BN normalises with stats that match
neither its own data nor the drifted global weights → near-random output (the worst-client collapse).
The frequently-sampled client's running stats track its real data. **Affine barely moves even when
trained** (γ stays ≈8.0, β ≈0), which is why affine staleness is the *secondary*, not primary, cause.

## Implication for mechanism design (Phase 2+, NOT built here)
Target the **running statistics**: keep them fresh/synchronised across clients (e.g. aggregate/sync BN
stats globally, or estimate them in a participation-robust way) while **personalising the affine γ/β**.
The "Stats-shared" cell shows this direction already recovers — and slightly exceeds — FedAvg robustness
without giving up FedBN's personalised affine. Candidate mechanisms to weigh next: global-stat sync /
imputation for unsampled clients, staleness-aware stat interpolation, or recomputing stats at eval.

## Artifacts / reproduce
- `phase2_step1/histories/*.pkl` (4 variants), `phase2_step1/bn_trace/*.csv`, `phase2_step1/log_diag_*.out`.
- Variant code: `phase2_step1_variants.patch` (`is_bn_local` in `utils.py`, generalised client in
  `client_app.py`, mode wiring in `server_app.py`/`pyproject.toml`).
- Run: `flwr run . --run-config "bn-local-mode='all|affine|stats|none' fraction-fit=0.1 num-server-rounds=30"`

## Caveats
Single seed, short scale (30 rounds, 20 clients), 10% participation — a diagnostic, not final numbers.
FedAvg's worst-client here (63.5) is lower than Step-4's (71.8) due to seed/sampling variance; the
*relative* pattern (which component recovers the collapse) is the robust takeaway. The least-sampled
client happened to be trained 0× this seed, giving the cleanest possible staleness illustration.
