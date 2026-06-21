# Part B — FedBN Partial-Participation Gap: Full Investigation Report

*Thesis-grade handoff. Every number below is transcribed from a committed file in this
repo (`surafel58/FedBN`); each table cites its source file and the commit that added it.
Values not present in the repo are marked **NOT FOUND IN REPO**. No new experiments were
run to produce this report (read-only).*

> **Critical reading note — the FedAvg bar is experiment-local.** The "FedAvg worst-client"
> number differs across experiments because of (a) single-seed vs 5-seed runs, (b) two
> different harnesses — the **Flower** `baselines/fedbn` harness (Phase 1, Steps 1/1b/1c,
> de-risk A/B, B0) vs a **standalone PyTorch** harness (Direction-B shrinkage, feature
> alignment), and (c) genuine high seed-variance of a min-over-clients statistic. Therefore
> **only within-experiment comparisons (Δ vs that experiment's own FedAvg) are valid**;
> absolute worst-client numbers are **not** comparable across sections. This is called out
> again where it matters.

---

## 1. Problem and motivation

**The gap.** FedBN (Li, Jiang, Zhang, Kamp, Dou, *"FedBN: Federated Learning on Non-IID
Features via Local Batch Normalization"*, ICLR 2021 — cited in the repo `README.md`, commit
`2fa38ad`) keeps each client's BatchNorm layers **local/personalized** and never aggregates
them. This wins under feature-shift non-IID but is documented to be fragile under **partial
client participation** (cross-device sampling). The target was a method that keeps FedBN-style
**personalized normalization** *and* is robust to **partial participation**.

- Documented partial-participation weakness reference: **Ramezani-Kebrya et al., TMLR 2023**
  — *provided by the user as the motivating citation; this reference is **NOT FOUND IN REPO***
  (the repo's own citations are FedBN and FedWon). Treat as external literature, not a repo
  artifact.
- **Pivot from the concept-drift method:** stated motivation is that feature-shift is a regime
  where gains over FedAvg are *conceptually* achievable (unlike label drift). The narrative
  rationale itself is **NOT FOUND IN REPO**, but its concrete precondition **is verified**:
  the original concept-drift model `class CifarCNN` (`FedCCFA/utils/models.py:39`) contains
  **0 BatchNorm layers** (`grep -c BatchNorm` = 0 on that file). FedBN requires BatchNorm,
  which is why a separate workspace/model/dataset was needed (Section 2).

**The two bars every candidate had to clear:** (1) **novel** (not a known technique), and
(2) **beats baselines robustly** — multi-seed, with the gain not inside the seed noise.

---

## 2. Setup (verified from code)

| Item | Value | Source |
|---|---|---|
| Model (reproduction) | `DigitModel`: 3 conv + 2 fc, **5 BatchNorm layers** (`bn1,bn2,bn3`=BatchNorm2d; `bn4,bn5`=BatchNorm1d) | `nets/models.py:7–43` (commit `2fa38ad`); confirmed by grep |
| Model (partial-participation runs) | "identical CNN, **14,219,210 params**", 5 BN layers (Flower `CNNModel`) | `SUMMARY.md` (commit `510fbe1`). *Flower `CNNModel` source is **NOT IN REPO** — it lives in the Flower baseline on the VM; only its patches are committed.* |
| Dataset | Digits-Five feature-shift benchmark: MNIST, SVHN, USPS, SynthDigits, MNIST-M (authors' pre-processed partitions) | `SUMMARY.md` (`510fbe1`); `nets/models.py`, `fedbn/dataset.py` (patched) |
| Clients | **20** (4 per domain) | `SUMMARY.md`, `PHASE2_STEP1.md` |
| Participation | **10% per round** (`fraction-fit=0.1`, 2 of 20 sampled/round) | `SUMMARY.md`, all Phase-2 reports |
| Rounds | **30** (smoke scale) for all partial-participation runs; Step 3 reproduction used 50 | `SUMMARY.md` and Phase-2 reports |
| Original model has no BN | `CifarCNN` (`FedCCFA/utils/models.py:39`): **0 BatchNorm** | verified by grep (this report) |

**Codebases (confirmed, `SUMMARY.md` §"What was run, where, and why two codebases", `510fbe1`):**
- **`med-air/FedBN`** (original ICLR'21 repo) — used for the full-participation reproduction
  (Step 3). It is **hard-wired to full participation (no client sampling)**.
- **Flower official `baselines/fedbn`** — used for all partial-participation runs, *because*
  the original repo has no client sampling and Flower's baseline has the same model/data
  **plus** native `fraction_fit` sampling.

**Seed counts per experiment** (from each report/results file):

| Experiment | Seeds | Source |
|---|---|---|
| Phase 1 Step 3 (reproduction) | 1 | `SUMMARY.md` |
| Phase 1 Step 4 (full vs partial) | 1 | `SUMMARY.md` ("Step 4 = 30 rounds, 1 seed") |
| Step 1 (4-variant diagnostic) | 1 | `PHASE2_STEP1.md` |
| Step 1b (oracle headroom) | 1 | `PHASE2_STEP1B.md` |
| Step 1c (multi-seed headroom) | **5** | `phase2_step1c/aggregate_5seed.txt` |
| De-risk A (similarity imputation) | **2** (seeds 0,1) | `phase2_step2derisk/phase2_step2derisk_results.txt` (`seeds=[0,1]`) |
| De-risk B (test-time BN) | **5** | `phase2_step2deriskB/phase2_step2deriskB_results.txt` |
| B0 (frozen/strict) | **5** | `pathB_B0/pathB_B0_results.txt` |
| Direction-B (shrinkage) | **5** | `pathB_dirB/pathB_dirB_results.txt` |
| Feature-alignment | **5** | `pathB_features/pathB_features_results.txt` |

---

## 3. Phase 1 — gap confirmation

**Step 3 — full-participation reproduction** (med-air repo, cross-silo, **5 clients, 50 rounds,
1 seed**). Source: `SUMMARY.md` (commit `510fbe1`); raw logs `phase1_artifacts/digits_fedbn.out`,
`digits_fedavg.out`.

| Domain | FedBN | FedAvg | Δ |
|---|---|---|---|
| MNIST | 96.20 | 95.19 | +1.01 |
| SVHN | 69.28 | 62.53 | +6.75 |
| USPS | 96.83 | 95.43 | +1.40 |
| SynthDigits | 81.14 | 80.03 | +1.11 |
| MNIST-M | 76.69 | 70.80 | +5.89 |
| **Average** | **84.03** | **80.80** | **+3.23** |

(Paper reference, per `SUMMARY.md`: avg FedBN 88.4 / FedAvg 86.5 — our smoke is ~4 pts low at
50 vs paper's 100 rounds; qualitative claim reproduces.)

**Step 4 — full vs partial participation** (Flower baseline, **20 clients, 30 rounds, 1 seed**;
worst-client = lowest single-client test accuracy of the 20). Source: `SUMMARY.md` (`510fbe1`);
histories `phase1_artifacts/step4_histories/{fedbn,fedavg}_{full,partial}.pkl`.

| Setting | Overall | Worst-client |
|---|---|---|
| FedBN, full participation | 88.75 | 79.55 |
| FedBN, **10% participation** | 71.99 | **12.43** |
| FedAvg, full participation | 87.54 | 75.35 |
| FedAvg, **10% participation** | 86.51 | 71.76 |

**The collapse:** FedBN worst-client **79.55 → 12.43** going full→partial (≈ random for 10-class),
overall −16.8; FedAvg barely moves (worst −3.6, overall −1.0). Under partial participation the
ranking **reverses**: FedAvg (86.5 overall / 71.8 worst) now beats FedBN (72.0 / 12.4).
*(All four numbers are single-seed.)*

---

## 4. Phase 2 Step 1 — BN-component diagnostic

Which part of BatchNorm drives the collapse? BN = **affine** (γ,β) + **running statistics**
(running_mean, running_var, num_batches_tracked). **20 clients, 10% participation, 30 rounds,
single seed.** Source: `PHASE2_STEP1.md` (commit `59cd050`); histories `phase2_step1/histories/diag_*.pkl`;
staleness trace `phase2_step1/bn_trace/{all,affine,stats}.csv`.

| Variant | Shared (fresh) | Overall | Worst-client |
|---|---|---|---|
| FedBN baseline (all BN local) | — (both stale) | 65.7 | **16.7** |
| Affine-shared (share γ/β, stats local) | affine only | 81.9 | 51.7 |
| FedAvg (share everything) | both | 82.5 | 63.5 |
| Stats-shared (share running stats, γ/β local) | running stats only | 82.6 | **68.6** |

**Verdict (per report):** stale **running statistics** are the primary driver — sharing them
alone restores worst-client 16.7 → 68.6 (≥ FedAvg 63.5) while keeping affine personalized;
sharing affine alone only partly recovers (→ 51.7).

**Staleness trace** (`phase2_step1/bn_trace/all.csv`, single seed): least-sampled = partition 14
(SynthDigits, trained 0×/30); most-sampled = partition 8 (USPS, 8×). Rarely-sampled client:
`conv1.weight` norm drifts 4.641→4.661 while `bn1.running_mean` stays **0.000 (frozen at init)**
and `bn1.running_var` stays **8.000 (frozen)**; γ≈8.00, β≈0.00 throughout. Frequently-sampled
client: running_mean 0.0→~1.92, running_var 8.0→~1.36 (tracks data); affine barely moves.

> **⚠ The "trivial fix" problem.** Stats-shared (68.6) **already beats FedAvg (63.5)** here and
> is a *known* idea (FBN/SiloBN-style sharing of BN statistics). So the novelty bar is not
> "beat collapsed-FedBN" — a novel method must beat **stats-shared / FedAvg**, not just the
> collapse. This reframed every subsequent gate. *(These are single-seed numbers; Step 1c
> re-measures the stats-shared-vs-FedAvg relationship at 5 seeds — see §5, where stats-shared
> ≈ FedAvg, not clearly above.)*

---

## 5. Phase 2 Step 1b / 1c — headroom gate

**Step 1b (single seed, the misleading negative).** Source: `PHASE2_STEP1B.md` (commit `a2c0f90`).

| Config | Overall | Worst-client |
|---|---|---|
| Stats-shared (global fresh) | 81.26 | 63.89 |
| Oracle (personalized fresh, recompute on own data) | 84.76 | 69.91 |
| FedAvg | 85.65 | **73.38** |

At one seed the oracle (69.91) **lost** to FedAvg (73.38) → "headroom not established."

**Step 1c (5 seeds — HEADROOM CONFIRMED).** Source: `phase2_step1c/aggregate_5seed.txt`
(committed in `fd32e22`); report `PHASE2_STEP1C.md`.

| Config | Overall (mean±std) | Worst-client (mean±std) |
|---|---|---|
| FedBN baseline | 82.03 ± 2.75 | 32.58 ± 22.57 |
| Stats-shared | 81.48 ± 2.94 | 64.16 ± 11.74 |
| Oracle (pers-fresh) | 86.61 ± 1.27 | **77.19 ± 1.13** |
| FedAvg | 82.18 ± 3.10 | 64.93 ± 7.88 |

Reported deltas (same file): **Oracle − FedAvg = +12.25 worst-client** (+4.43 overall);
Stats-shared − FedBN baseline = +31.58 worst-client (−0.55 overall). *(All ±std are over
seeds 0–4 from `aggregate_5seed.txt`.)*

**Key insight (per report):** the +12 gain is from **personalizing** the stats, not merely
refreshing them — global-fresh (Stats-shared, 64.16) ≈ FedAvg (64.93) ≪ personalized-fresh
oracle (77.19). **Lesson:** single-seed go/no-go is noise here — Step 1b's negative was an
unlucky draw, overturned by 5 seeds (the same configs swung ~10 pts run-to-run).

> Note the stats-shared-vs-FedAvg relationship flipped between Step 1 (single seed: stats-shared
> 68.6 > FedAvg 63.5) and Step 1c (5 seeds: stats-shared 64.16 ≈ FedAvg 64.93). At 5 seeds
> stats-shared does **not** robustly beat FedAvg — reinforcing that the only robust headroom is
> the *personalized* oracle.

---

## 6. The gated mechanism attempts

### 6a. Similarity imputation (de-risk A) — **FAIL**
Can an unsampled client's BN stats be imputed from *similar* recently-sampled clients better than
a plain global average? Offline analysis, **2 seeds (0,1), n=739 (client,round) cases**; distance =
relative-L2 to the oracle stats (lower = better). Source:
`phase2_step2derisk/phase2_step2derisk_results.txt` (commit `429817e`); report `PHASE2_STEP2DERISK.md`.

| Window K | dist_global | dist_sim | sim<glob % | improve % | weight-entropy ratio | avg donors |
|---|---|---|---|---|---|---|
| 1 | 0.2351 ± 0.0821 | 0.2276 ± 0.0866 | 89.6 | 3.2 | 0.995 | 2.0 |
| 3 | 0.2253 ± 0.0538 | 0.2133 ± 0.0560 | 94.2 | 5.3 | 0.994 | 5.0 |
| 5 | 0.2296 ± 0.0501 | 0.2173 ± 0.0514 | 98.4 | 5.4 | 0.994 | 7.6 |

(`stale_self` = dist from oracle to the client's own stale stats = 0.2287, same file.)
**Why it failed:** the weight-entropy ratio ≈ **0.99 (near-uniform)** → BN fingerprints don't
single out genuinely "similar" donors, so similarity ≈ global-average (only 3–5% closer), and
**both** sit ~0.22 from the oracle (no better than the client's own stale stats). A client's
correct stats cannot be reconstructed from other clients.

### 6b. Test-time BN recompute (de-risk B) — **WORKS, but known technique**
Each client recomputes its own BN running stats on k of its own samples at inference (no test-label
leakage; train/adaptation split only), then predicts. **5 seeds.** Source:
`phase2_step2deriskB/phase2_step2deriskB_results.txt` (commit `75b376c`); report `PHASE2_STEP2DERISKB.md`.

| Config | Overall (mean±std) | Worst-client (mean±std) | Δ worst vs FedAvg |
|---|---|---|---|
| FedBN baseline (no recompute) | 81.45 ± 2.30 | 29.37 ± 22.02 | — |
| FedAvg | 81.02 ± 3.75 | 63.18 ± 10.95 | (bar) |
| Test-time BN, 32 samples | 84.23 ± 1.12 | 73.62 ± 1.25 | **+10.44** |
| Test-time BN, 128 samples | 86.19 ± 0.50 | 75.23 ± 2.00 | **+12.05** |
| Oracle (full recompute) | 84.94 ± 0.98 | 77.07 ± 1.42 | **+13.89** |

Deltas are from the same file. **It works (+10–14 pp worst-client, low variance).** Honest
caveats (per report): test-time BN adaptation is a **known technique** (fails the novelty bar),
**requires inference-time data**, and FedAvg's worst-client is high-variance (±10.95) so the
robustness is in test-time BN's *consistency* (std ~1–2) and much higher mean, not in fully
non-overlapping 1-std bands.

### 6c. Frozen / strict-setting stats (B0) — **FAIL**
Strict zero-inference-data setting: compute ideal stats once at round R, **freeze**, evaluate the
drifted final model with shared/frozen stats and **no recompute**. **5 seeds.** Source:
`pathB_B0/pathB_B0_results.txt` (commit `e03209d`); report `PATHB_B0.md`.

| Config | Overall (mean±std) | Worst-client (mean±std) | Δ worst vs FedAvg |
|---|---|---|---|
| #2 FedBN, no recompute (stale local) | 81.75 ± 2.99 | 29.77 ± 23.34 | −39.2 |
| #1 FedAvg (shared global stats) — bar | 83.91 ± 1.65 | 68.99 ± 2.77 | — |
| #4 Shared-global-stats (Candidate-3 ceiling) | 81.59 ± 5.53 | 64.13 ± 12.84 | **−4.87** |
| #3 frozen@5 (stale) | 78.78 ± 4.41 | 65.57 ± 4.78 | **−3.42** |
| #3 frozen@15 (mid) | 82.50 ± 1.51 | 70.19 ± 0.87 | **+1.20** |
| #3 frozen@30 (no staleness ≡ recompute-on-final) | 86.39 ± 0.75 | 76.51 ± 1.18 | **+7.52** |

Deltas from the same file. **Why it failed:** the advantage decays **monotonically with staleness**
(+7.52 at frozen@30 → +1.20 at frozen@15 → −3.42 at frozen@5). The only point clearly above FedAvg
(frozen@30, +7.52) is **zero-staleness = recompute on the deployed model** — exactly the test-time
capability the strict setting forbids. The achievable shared-stats ceiling (#4) **trails FedAvg by
−4.87**. So the Step-1c +12 oracle headroom was an artifact of recompute-on-current-weights, not of
having good frozen stats.

### 6d. Shrinkage estimator (Direction B) — **FAIL (single-seed artifact)**
Interpolate scarce/biased local stats with the global prior: `stats(λ)=λ·local_k+(1−λ)·global` on
the FedAvg model (λ=0≡FedAvg, λ=1≡naive recompute). **5 seeds, standalone harness, PAIRED per-seed
analysis.** Source: `pathB_dirB/pathB_dirB_results.txt` and `mix_full.json` (commit `fda71f6`);
report `PATHB_DIRB.md`.

FedAvg absolute worst-client = **59.17 ± 14.55** (high variance → paired analysis required).
Regime B (biased) paired Δ vs FedAvg, best interior λ (same file):

| k | Δ naive (λ1) | Δ best-interior (λ*) | int Δ>0 robust? | int > naive? |
|---|---|---|---|---|
| 4 | −5.68 ± 18.88 | +1.43 ± 2.32 (λ=0.2) | no | yes |
| 16 | +2.96 ± 17.06 | +3.76 ± 9.78 (λ=0.6) | no | yes |
| 64 | +4.82 ± 17.62 | +5.52 ± 12.86 (λ=0.7) | no | yes |
| 256 | +5.41 ± 17.18 | +6.06 ± 12.69 (λ=0.7) | no | yes |

The **λ\*(k) shrinkage signature is textbook** (interior λ* rising with k: 0.2→0.4→0.6→0.6→0.7→0.8→
0.7→1.0, `PATHB_DIRB.md`) and interior > naive on the *mean* — **promising on the surface**. But the
**per-seed breakdown** (Regime B, k=64, λ*=0.7; `PATHB_DIRB.md`) kills it:

| seed | FedAvg worst | shrinkage Δ |
|---|---|---|
| 0 | 69.2 | −0.5 |
| 1 | 68.6 | −6.4 |
| 2 | 62.2 | +2.3 |
| 3 | 65.5 | +1.7 |
| 4 | **30.5** (FedAvg collapsed) | **+30.5** |

Per-seed FedAvg/naive-full (`pathB_dirB_results.txt`): FedAvg = {69.2, 68.6, 62.2, 65.5, **30.5**};
naive-full = {75.0, 72.5, 73.7, 73.7, 76.4}. **Why it failed:** the positive *mean* is **entirely
seed 4**, where FedAvg itself collapsed (30.5) and *any* recompute rescues it — and naive (λ=1, 76.4)
does so as well or better. On the 4 healthy seeds, interior shrinkage is ≈0 or negative
({−0.5, −6.4, +2.3, +1.7}). No robust interior headroom.

### 6e. Feature-alignment (features lever) — **FAIL**
Add a loss pulling each client's BN-input activation stats toward the broadcast global BN stats
during local training; evaluate with shared global stats (strict, no recompute). α=0≡FedAvg.
**5 seeds, standalone harness.** Source: `pathB_features/pathB_features_results.txt` and
`align_full.json` (commit `2f92805`); report `PATHB_FEATURES.md`.

| Config | Overall (mean±std) | Worst-client (mean±std) |
|---|---|---|
| FedAvg (α=0) — bar | 80.64 ± 9.00 | 60.86 ± 16.76 |
| align α=0.1 | 81.65 ± 7.12 | 61.21 ± 15.23 |
| align α=1.0 | 81.26 ± 7.49 | 60.85 ± 15.34 |
| align α=10.0 | 34.42 ± 30.57 | **25.75 ± 21.87** |
| FedBN no-recompute (ref) | 83.38 ± 1.21 | 52.66 ± 12.34 |
| oracle recompute (ref) | 86.69 ± 1.15 | 74.89 ± 1.19 |

Per-seed worst-client and healthy-seed paired Δ (same file; healthy seeds = {0,1,2,3} where FedAvg
worst ≥ 55; seed 4 = FedAvg collapse at 27.90):

| α | Δ_healthy (mean±std) | #healthy won |
|---|---|---|
| 0.1 | −0.39 ± 1.37 | 2/4 |
| 1.0 | −0.76 ± 1.46 | 1/4 |
| 10.0 | −38.89 ± 25.19 | 0/4 |

**Why it failed:** mild alignment (α=0.1, 1.0) is **neutral** on healthy seeds (Δ≈0, within ±1.5);
strong alignment (α=10) is **catastrophic** (worst-client 25.75; 7.90 ≈ random on several seeds) —
forcing feature-shifted domains' activations to a common target destroys class discriminability,
the opposite of what per-domain BN buys. The only positive deltas are on the FedAvg-collapse seed
(excluded by the healthy-seed scrutiny).

---

## 7. The complete two-bar scorecard

| Lever | Beats baselines robustly? | Novel? | Status | Source |
|---|---|---|---|---|
| Similarity imputation | No (≈ global-avg; ~0.22 from oracle) | yes | FAIL | de-risk A (`429817e`) |
| Frozen / strict-setting stats | No (shared-stats −4.87 vs FedAvg; frozen decays to ≤ FedAvg) | yes | FAIL | B0 (`e03209d`) |
| Shrinkage estimator | No (gain = 1 collapse seed; ≈0 on healthy seeds) | yes | FAIL | Direction-B (`fda71f6`) |
| Feature-alignment | No (neutral at mild α, catastrophic at strong α) | yes | FAIL | features (`2f92805`) |
| Test-time BN recompute | **Yes** (+10–14 pp worst-client, 5 seeds) | **No** (known technique) | gain ✓ / novelty ✗ | de-risk B (`75b376c`) |

**No direction cleared both bars.**

---

## 8. The unified finding (the characterization)

FedBN's partial-participation worst-client collapse is fundamentally a **test-time statistics
problem**. With scarce/biased/zero inference data, **neither stats nor feature-alignment
interventions beat FedAvg** (de-risk A, B0, Direction-B, feature-alignment all FAIL). The only
thing that reliably lifts worst-client is the **known test-time BN recompute** — and only **when
the client has inference data** (de-risk B, +10–14 pp).

**Mechanistic throughline:** a client's correct normalization depends on **its own current data**.
You cannot (i) **reconstruct** it from other clients (similarity imputation: BN fingerprints don't
identify similar donors — de-risk A), (ii) **freeze** it (it goes stale as the global model drifts —
B0), (iii) **shrink** toward a global prior to recover it from few samples (no robust interior gain —
Direction-B), or (iv) **align it away** by reshaping features (destroys discriminability —
feature-alignment). All four failures reduce to the same root, and the one method that works is the
one that uses the client's own data at inference.

---

## 9. What was achieved vs not (honest accounting)

- **Real metric gain:** test-time BN recompute, **+10–14 pp worst-client** (de-risk B, 5 seeds) —
  but a **known technique** → clears the gain bar, fails the novelty bar.
- **Novel metric gain:** **none** — every novel mechanism tied or lost to FedAvg on healthy seeds.
- **Both bars simultaneously:** **zero** directions.
- **The contribution, framed as a thesis:** the **characterization** (the collapse is a test-time
  statistics problem; a systematic map of why four distinct novel levers all fail) **plus** the
  deployable **test-time-BN recommendation** — **not** a novel accuracy gain.

---

## 10. Honest-framing checklist (for the write-up)

- **Smoke scale, not full scale:** 20 clients, 30 rounds, Digits-Five — `SUMMARY.md` explicitly flags
  this ("Trends are large and unambiguous; absolute numbers are indicative, not final"). Step 3 used
  5 clients / 50 rounds.
- **Single-seed go/no-go are diagnostics, not claims:** Step 1, 1b, and Phase-1 Step 4 are single-seed.
  Step 1b's single-seed *negative* was overturned by the 5-seed Step 1c — single-seed results are
  inside the ~10-pt seed noise here.
- **Test-time BN robustness is in *consistency*, not separated means:** FedAvg worst-client is
  high-variance (±10.95 in de-risk B; ±14.55 / ±16.76 in the standalone-harness Direction-B / feature
  experiments); test-time BN's case rests on its low variance (std ~1–2) and higher mean, with 1-std
  bands not fully separated from high-variance FedAvg.
- **Per-seed scrutiny is a methodological strength to highlight:** it caught a *false positive* — the
  shrinkage estimator's positive **mean** (Direction-B) was entirely one FedAvg-collapse seed (seed 4);
  on healthy seeds the effect was ≈0/negative. The healthy-seed paired analysis (also applied in the
  feature-alignment gate) is what prevented over-claiming.
- **Experiment-local FedAvg bar / two harnesses:** absolute worst-client numbers are not comparable
  across sections (Flower vs standalone harness, single vs 5 seeds); only within-experiment deltas are.

---

## 11. Data provenance

| Report file | Commit | Date | Backing result files (also in repo) |
|---|---|---|---|
| `SUMMARY.md` | `510fbe1` | 2026-06-16 | `phase1_artifacts/step4_histories/*.pkl`, `digits_{fedbn,fedavg}.out` |
| `PHASE2_STEP1.md` | `59cd050` | 2026-06-17 | `phase2_step1/histories/diag_*.pkl`, `phase2_step1/bn_trace/*.csv` |
| `PHASE2_STEP1B.md` | `a2c0f90` | 2026-06-17 | `phase2_step1b/histories/*` *(histories present; **no separate results.txt** — numbers are in the report)* |
| `PHASE2_STEP1C.md` | `fd32e22` | 2026-06-17 | **`phase2_step1c/aggregate_5seed.txt`** |
| `PHASE2_STEP2DERISK.md` | `429817e` | 2026-06-17 | **`phase2_step2derisk/phase2_step2derisk_results.txt`**, `sampling/fit_seed{0,1}.csv` |
| `PHASE2_STEP2DERISKB.md` | `75b376c` | 2026-06-18 | **`phase2_step2deriskB/phase2_step2deriskB_results.txt`** |
| `PATHB_B0.md` | `e03209d` | 2026-06-18 | **`pathB_B0/pathB_B0_results.txt`**, `pathB_B0/histories/*.pkl` |
| `PATHB_DIRB.md` | `fda71f6` | 2026-06-19 | **`pathB_dirB/pathB_dirB_results.txt`**, `pathB_dirB/mix_full.json` |
| `PATHB_FEATURES.md` | `2f92805` | 2026-06-20 | **`pathB_features/pathB_features_results.txt`**, `pathB_features/align_full.json` |

**Marked NOT FOUND IN REPO:** the Ramezani-Kebrya et al. (TMLR 2023) citation (external, user-provided);
the concept-drift→feature-shift *pivot rationale* narrative (motivation, not a repo number — but the
CifarCNN-no-BN precondition is verified in `FedCCFA/utils/models.py:39`); the Flower `CNNModel` source
code (the 14,219,210-param model — only its param count in `SUMMARY.md` and its patches are committed;
the class source lives in the Flower baseline on the VM, not in this repo); raw per-(seed,domain) tables
for Phase 1 Step 4 beyond the four (overall, worst-client) values reported in `SUMMARY.md`.
