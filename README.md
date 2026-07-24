# vineppo_replication — 8 rollouts agree with 16 rollouts (they must); the number that matters is the within-problem r ≈ 0.62

VinePPO (Kazemnejad et al., arXiv 2410.01679) claims PPO's learned value network is near chance at step-level credit assignment on math reasoning, while Monte-Carlo rollouts from the same policy are near ceiling. This repo runs that comparison locally for $0. For a partial GSM8K chain-of-thought, gold V\* is the fraction of 16 independent continuations (Qwen2.5-1.5B-Instruct 4-bit, MLX) that reach the correct answer; three estimators compete: a **simulator** (8 disjoint rollouts), a **verbalized one-pass judge** (P("Yes") from the logits of "will this end correctly?"), and a **probe** (ridge/MLP on the prefix's last hidden state, trained directly on gold V\*).

Read the design before the numbers. The simulator and the "gold standard" are the *same estimator at two sample sizes* — same policy, temperature, prompt, and answer parser — so their headline correlation is a reliability coefficient (an estimator against its higher-sample twin) and cannot fail. And both one-pass baselines were too weak for their low scores to indict one-pass critics in general: the judge is degenerate and the probe is trained on 42 examples in 1536 dimensions. Net: **this is a cheap local study whose setup is consistent with VinePPO's claim — rollout-based value estimates carry real step-level signal, and the one-pass baselines here couldn't read value — but it is not an independent confirmation: the simulator side can't fail by construction, and the critic side was starved.**

## Results

23 problems, 69 prefixes — 3 per problem, so clustered, not independent (informative slice: screened for 0 < p(success) < 1). Single run, rollouts unseeded, one probe split, no repeats.

| Measurement | Value | What it shows / what it can't show |
|---|---|---|
| Simulator (8 rollouts) vs gold V\* (16 rollouts), pooled | r ≈ 0.878 (Spearman 0.87) | **Circular by construction.** Given the observed spread of success probabilities, sampling noise alone predicts r ≈ 0.913 between two disjoint rollout sets of the same policy; observed 0.878. It measures rollout-count reliability, not estimator quality, and cannot fail. Also 87.5% of V\* variance is between-problem, so the pooled r is mostly problem difficulty. |
| Simulator vs V\*, within-problem (difficulty removed) | r ≈ 0.62 (difficulty component alone: r ≈ 0.82) | **The most meaningful positive number here.** After centering out problem difficulty, 8 rollouts still track which *step* of a chain raised or lowered the success probability — genuine credit-assignment signal. Still same-estimator-vs-twin, so it bounds reliability, not accuracy against an external truth. |
| Hidden-state probe vs V\* | r ≈ 0.06 | **Underpowered, not "≈ chance."** Fisher CI (n = 69): [−0.18, 0.29]; cluster-adjusted (n_eff = 23, ICC ≈ 0.88): [−0.36, 0.46] — not distinguishable from weakly informative. The probe is Ridge(α = 10) on frozen 1536-dim states trained on 42 prefixes; with n ≪ d, r ≈ 0 is expected even if value were perfectly decodable. Says nothing about VinePPO's fine-tuned value net, which trains a full network on vastly more data. |
| Verbalized one-pass judge vs V\* | r ≈ −0.23 | Problem-level bootstrap CI [−0.52, 0.20] includes 0, and the judge is degenerate (median P(Yes) = 0.07 — it says "no" to nearly everything). Shows this prompt failed, not that verbalized value estimation fails. |
| Within-chain prefix ordering (mean Spearman) | simulator 0.45; verbalized 0.01; probe 0.03 | Directionally matches the rows above, but each chain has only 3 prefixes, so per-chain Spearman is extremely coarse. |
| Branch ranking: pick the better next step (gold gap ≥ 0.15) | simulator 0.83; verbalized 0.67; probe 0.67 (chance 0.50) | n = 6 pairs — anecdotal. |
| Probe capacity check (analyze2.py, 3-fold) | ridge r ≈ 0.02; MLP (256→64) r ≈ −0.03 | MLP does not beat ridge, but with ~46 training prefixes per fold neither model could learn anything, so this cannot distinguish "value isn't encoded" from "the probe was starved." |
| Cost | 242k generated tokens, ~51 tok/s, ≈ 80 min on an M4 Mac (MLX), $0 | |

## Run

```bash
pip install mlx-lm datasets numpy scipy scikit-learn
python3 run.py --smoke    # tiny pipeline check first
python3 run.py            # full run, ≈ 80 min on an M4; resumable
python3 analyze.py        # headline metrics table
python3 analyze2.py       # failure-mechanism re-analysis (no new generation)
```

## Layout

```
run.py                  screen GSM8K problems, build step prefixes, compute V* (16 rollouts),
                        simulator (8 rollouts), verbalized P(Yes), hidden states; append-only/resumable
analyze.py              metrics: correlation with V*, within-chain ordering, branch-point ranking
analyze2.py             re-analysis of saved data: signal vs remaining steps, difficulty-vs-step
                        variance decomposition, MLP-vs-ridge probe capacity
results/data.jsonl      per-prefix records incl. 1536-dim last-token hidden states (for probe extensions)
results/rank.jsonl      branch-point next-step candidate pairs with per-candidate values
results/full_run.log    console log of the full run
results/*_smoke.jsonl   smoke-test outputs
```

## Limitations

The central one: the experiment is asymmetric. The simulator is compared against a higher-sample copy of itself (shared policy, temperature, prompt, parser), so its side of the ledger cannot come out badly; the one-pass side fields a degenerate judge and an n ≪ d probe, so it could hardly come out well. That asymmetry is why this repo is a consistency check on VinePPO, not a test of it. Distinguishing "value is not one-pass readable" from "these one-pass readers were too weak" would need a trained value head (or at least a probe with hundreds of problems), and V\* from a source other than the policy's own rollouts.

Everything is also small and single-shot: one run, unseeded rollouts, one probe train/test split, no repeats; 23 problems and 69 prefixes with 3 prefixes per problem (effective n is closer to 23 than 69 for pooled statistics, ICC ≈ 0.88); 6 usable ranking pairs. One model (1.5B, 4-bit) on one dataset, and screening keeps only problems the policy sometimes solves.

## Context

The setup is consistent with the broader critic-supervision picture — critics work as predictors of grounded outcomes when they simulate the future, and the one-pass estimates tried here read nothing — but on this design the first half is guaranteed and the second half is underpowered, so treat it as a cheap directional data point beside VinePPO, not independent evidence for it.

Released under the MIT License.
