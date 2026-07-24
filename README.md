# vineppo_replication — you can't read a reasoning step's value off the text in one pass; you have to simulate

VinePPO (Kazemnejad et al., arXiv 2410.01679) claims PPO's learned value network is near chance at step-level credit assignment on math reasoning, while Monte-Carlo rollouts from the same policy are near ceiling. This repo replicates that measurement locally for $0. The question, plainly: for a partial chain-of-thought, is V(prefix) = P(correct final answer | prefix) *readable* — from the text or the hidden state, in one forward pass — or does it only exist by *running the future*? Ground truth V\* is the fraction of 16 independent continuations (Qwen2.5-1.5B-Instruct 4-bit, MLX) that reach the correct GSM8K answer. Three estimators compete: a **simulator** (8 disjoint rollouts), a **verbalized one-pass judge** (P("Yes") from the logits of "will this end correctly?"), and a **probe** (ridge/MLP on the prefix's last hidden state, trained directly on gold V\* — more per-state supervision than a real PPO value head ever gets).

## Results

23 problems, 69 prefixes (informative slice: screened for 0 < p(success) < 1).

| Experiment | Finding |
|---|---|
| Value prediction vs gold V\* | simulator r ≈ 0.88 (Spearman 0.87) — ≈ ceiling for 8-vs-16-rollout noise; verbalized r ≈ −0.23; hidden-state probe r ≈ 0.06 — ≈ chance |
| Within-chain prefix ordering (mean Spearman) | simulator 0.45; verbalized 0.01; probe 0.03 |
| Branch ranking: pick the better next step (gold gap ≥ 0.15, n = 6 pairs) | simulator 0.83 vs chance 0.50; verbalized 0.67; probe 0.67 |
| Failure mechanism (analyze2.py) | simulator tracks both problem difficulty (r = 0.82) and within-problem step signal (r = 0.62); one-pass methods ≈ 0 on both — not even the difficulty shortcut |
| Probe capacity | out-of-fold ridge r ≈ 0.02, MLP (256→64) r ≈ −0.03 — MLP does not beat ridge, so the failure is encoding, not linear readout |
| Cost | 242k generated tokens, ~51 tok/s, ≈ 80 min on an M4 Mac (MLX), $0 |

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

Small: 23 problems, 69 prefixes, and only 6 usable ranking pairs, so the branch-ranking numbers are anecdotal. One model (1.5B, 4-bit) on one dataset. The "critic" is a verbalized judge plus a frozen-feature probe, not a PPO-trained value head — though the probe gets gold V\* labels, which is strictly more supervision, so its failure is the stronger result. V\* is itself a 16-rollout estimate, so the simulator's r ≈ 0.88 is roughly the sampling-noise ceiling, not a claim of a perfect estimator. Screening keeps only problems the policy sometimes solves, so all numbers are on that informative slice.

## Context

Consistent with the broader critic-supervision picture: critics work as predictors of grounded outcomes when they get to simulate the future, and fail (and become hackable) when used as one-pass estimates or optimization targets.

Released under the MIT License.
