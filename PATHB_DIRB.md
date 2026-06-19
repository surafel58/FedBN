# Path B, Direction B — shrinkage-estimator headroom gate (combined Step 1+2)

**Diagnostic only — no estimator built.** Deciding question: is there headroom for a SHRINKAGE
estimator (interpolate a client's scarce/biased local BN stats with a global BN prior) to beat **FedAvg**
worst-client — not just naive recompute? Bar that matters = **FedAvg**.

Design 1 (confirmed): base = FedAvg model (partial-participation, 20 clients, 10% participation, 30 rounds,
Digits-Five). `stats(λ) = λ·local_k + (1−λ)·FedAvg_global` on the FedAvg model → **λ=0 ≡ FedAvg**, **λ=1 ≡
naive test-time recompute at k**. λ* hand-picked = the family's oracle ceiling. 5 seeds. Regimes:
A=SCARCE (k random samples), B=BIASED (k from only 2 classes). No test leakage (stats from the client's
TRAIN split, inputs only; test set only for scoring; verified in `mix_probe.py`).

## Why the analysis must be PAIRED
FedAvg worst-client varies wildly by seed: **{69.2, 68.6, 62.2, 65.5, 30.5}** → mean 59.2 ± **14.6**.
Since the mix and FedAvg(λ=0) use the SAME model per seed, that model-quality variance is shared and must
be cancelled by a **paired** per-seed Δ = worst(λ) − worst(λ=0). (Unpaired mean±std is meaningless here.)

## Results — paired Δ worst-client vs FedAvg (mean±std over 5 seeds)

| Regime | k | Δnaive(λ1) | Δbest-interior (λ*) | int Δ robust? (lower band>0) | int>naive? |
|---|---|---|---|---|---|
| A scarce | 4 | −6.7±16.6 | +1.9±5.5 (0.4) | no | yes |
| A scarce | 16 | +10.9±15.4 | +10.8±15.0 (0.9) | no | no |
| A scarce | full | +15.1±15.6 | +14.7±15.0 (0.9) | no | no |
| B biased | 4 | −5.7±18.9 | +1.4±2.3 (0.2) | no | yes |
| B biased | 64 | +4.8±17.6 | +5.5±12.9 (0.7) | no | yes |
| B biased | 256 | +5.4±17.2 | +6.1±12.7 (0.7) | no | yes |
| B biased | full | +15.1±15.6 | +14.7±15.0 (0.9) | no | no |

Regime B shows the **textbook shrinkage λ*(k) signature** (interior λ* rising with k: 0.2→0.4→0.6→0.6→0.7
→0.8→0.7→1.0) and interior > naive on the *mean* — but **no row is robust** (every lower band crosses 0).

## The decisive per-seed breakdown (Regime B, k=64, λ*=0.7)
| seed | FedAvg worst | mix Δ vs FedAvg |
|---|---|---|
| 0 | 69.2 | **−0.5** |
| 1 | 68.6 | **−6.4** |
| 2 | 62.2 | +2.3 |
| 3 | 65.5 | +1.7 |
| 4 | **30.5** (FedAvg collapsed) | **+30.5** |

The positive *mean* Δ is **entirely one pathological seed (seed 4)** where FedAvg itself broke (a client
collapsed to 30.5). On that seed, *any* recompute rescues it — and **naive recompute (λ=1) does it as well
or better** (seed4 naive-full = 76.4). On the **4 healthy-FedAvg seeds (0–3), the interior-shrinkage Δ is
≈0 or negative** ({−0.5, −6.4, +2.3, +1.7}). Shrinkage does **not** help when FedAvg works.

## VERDICT: **FAIL — shrinkage has no robust headroom over FedAvg.**
- No (regime, k) gives an interior-λ shrinkage that beats FedAvg robustly (lower band > 0) while also
  beating pure-local recompute.
- The apparent mean gain is a **single-seed artifact** (FedAvg-collapse on seed 4), which **naive
  test-time recompute already addresses** (and ≥ shrinkage there). On healthy seeds, shrinkage ≤ FedAvg.
- In the **scarce** regime, the best interior λ* sits at ~0.9 (≈ pure-local) for all useful k → shrinkage
  ≈ naive recompute, no genuine interior benefit. The biased regime shows the right *shape* but the effect
  is within seed noise and driven by the FedAvg-failure seed.
- Hand-picked-λ* is an **oracle ceiling**; a deployable estimator would do worse. Even the ceiling doesn't
  robustly clear FedAvg → a real shrinkage estimator won't either.

## Implication
Shrinkage does not pass the gate → **do not build the shrinkage estimator.** This is consistent with the
whole Phase-2 arc: the only intervention that reliably lifts worst-client is **test-time BN recompute when
the client has inference data** (derisk-B, a known technique). With scarce/biased/zero inference data,
no stats-based method (imputation [derisk-A], frozen/shared [B0], shrinkage [here]) robustly beats FedAvg.
The remaining lever, if any, is the **features/representation**, not the BN statistics.

## Artifacts / reproduce
- `pathB_dirB/mix_probe.py` (cached harness), `mix_probe_agg.py` (unpaired), `mix_probe_agg2.py` (PAIRED —
  the correct one), `mix_full.json` (890 records: 5 seeds × 8 k × 11 λ × 2 regimes + refs),
  `pathB_dirB_results.txt`.
- Run: `python mix_probe.py --seeds 0,1,2,3,4 --ks 4,8,16,32,64,128,256,full --lams 0,...,1.0`
  then `python mix_probe_agg2.py mix_full.json`.

## Caveats
5 seeds, short scale; the standalone training harness has higher FedAvg seed-variance (±14.6) than the
Flower setup (±~8), which inflates the worst-client noise. But the verdict does not hinge on the noise: the
per-seed breakdown shows the gain is one FedAvg-failure seed (naive ≥ shrinkage there) and ≈0/negative on
healthy seeds — so shrinkage fails the gate structurally, not just for lack of statistical power.
