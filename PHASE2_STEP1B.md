# Phase 2, Step 1b — Headroom / oracle diagnostic (is there room above the trivial global-stats fix?)

**Diagnostic only — no method built.** Question: Step 1 showed the trivial "stats-shared" config
(share BN running stats globally, keep γ/β local) already fixes the collapse. Is there **headroom**
above it for a *personalized*-fresh-stats mechanism, or does the trivial global-stats fix already
capture the gain? We measure the **ceiling** of personalized-fresh stats with an oracle.

Setup: Flower FedBN baseline, Digits-Five, 20 clients, **10% participation (fraction-fit=0.1)**,
30 rounds, **single seed**.

## Configs
| Config | BN running stats | BN affine γ/β | conv/fc |
|---|---|---|---|
| **Stats-shared** | global (FedAvg-aggregated, "fresh") | local | global |
| **Oracle (personalized-fresh)** | **recomputed per client on its OWN train data w/ current weights** | local (FedBN) | global |
| **FedAvg** | global | global | global |

**Oracle** (`_recompute_bn_stats_on_local_data`, eval-time, all 20 clients incl. never-sampled):
reset BN stats → `momentum=None` (exact cumulative) → `train()` + `torch.no_grad()` forward pass over
the client's **training** data (never the test set) → restore → `eval()` + test. No gradient step, no
test leakage. This is the best a personalization mechanism could do (every client gets stats that are
both fresh and personalized). γ/β kept FedBN-local, isolating *stats personalization* vs global stats.

## Results (10% participation, 30 rounds, single seed)
| Config | Overall | Worst-client | Δ overall vs Stats-shared | Δ worst vs Stats-shared |
|---|---|---|---|---|
| Stats-shared (global fresh) | 81.26 | 63.89 | — (ref) | — (ref) |
| **Oracle (personalized fresh)** | 84.76 | 69.91 | **+3.50** | **+6.02** |
| FedAvg (all shared) | 85.65 | **73.38** | +4.39 | +9.49 |

## VERDICT (one line)
**Headroom for a personalized-fresh-stats mechanism is NOT reliably established at this scale: the
oracle beats the trivial global-stats config nominally (+3.5 overall / +6.0 worst-client), but it does
NOT beat plain FedAvg (FedAvg 73.4 vs oracle 69.9 worst-client), and all margins are within the
single-seed run-to-run variance we observe — so this does not justify designing a novel mechanism yet.**

## Why this is cautionary (read before acting)
1. **The oracle's own ceiling is below FedAvg this seed.** Ranking: FedAvg (85.7/73.4) > Oracle
   (84.8/69.9) > Stats-shared (81.3/63.9). Since the oracle is the *upper bound* for "personalized-fresh
   stats + FedBN-local γ/β", a deployable approximation of it would land **at or below** that ceiling —
   i.e. at or below FedAvg. A mechanism whose ceiling barely matches the FedAvg baseline is hard to
   justify as a novel contribution.
2. **Margins are within single-seed noise.** Worst-client swings ~10 pts run-to-run at this scale: the
   *same* configs gave, across Step 1 vs Step 1b, Stats-shared 68.6 → 63.9 and FedAvg 63.5 → 73.4
   (their ranking even flipped). The oracle's +6.0 over stats-shared is comparable to that noise.
3. **Affine, not just stats, may be the real lever.** The oracle keeps γ/β LOCAL, so never-sampled
   clients use *initial* γ/β; FedAvg (global γ/β) beat the oracle. This hints the local-affine design
   itself hurts rarely-sampled clients under partial participation — consistent with Step 1, where
   sharing affine also helped (16.7 → 51.7).

## Recommendation (next decision, NOT executed here)
Before designing any mechanism, **run multiple seeds** (≥3-5) on these three configs (+ FedBN baseline)
to establish whether (a) oracle reliably > stats-shared, and crucially (b) oracle reliably > FedAvg.
If the oracle does not robustly beat FedAvg across seeds, the honest finding is that **global stat
sharing / FedAvg already captures the available gain**, and the "personalized-fresh-stats" direction
lacks headroom — we should rethink (e.g. target affine personalization, or a different problem framing)
rather than build it. If the oracle *does* robustly clear FedAvg across seeds, then a personalized-fresh
mechanism (e.g. similarity-based stat imputation for unsampled clients) is justified.

## Artifacts / reproduce
- `phase2_step1b/histories/{hr_statsshared,hr_oracle,hr_fedavg}.pkl`, logs, `parse_history.py`.
- Oracle code: `phase2_step1b_oracle.patch` (the `_recompute_bn_stats_on_local_data` method + the
  `oracle-stats` flag in `client_app.py`/`pyproject.toml`).
- Run: `flwr run . --run-config "bn-local-mode='all' oracle-stats=true fraction-fit=0.1 num-server-rounds=30"`
  (Stats-shared = `bn-local-mode='affine' oracle-stats=false`; FedAvg = `bn-local-mode='none'`).

## Caveats
Single seed, short scale (the brief scoped it so). The cross-run variance noted above is precisely why
the verdict is "not reliably established" rather than a clean yes/no — multi-seed is the needed
tie-breaker. No method was built.
