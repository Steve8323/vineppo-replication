# vineppo_replication — 6 rollouts agree with 12 rollouts (they must), and the within-problem r ≈ 0.62 is a reliability ceiling, not accuracy

VinePPO (Kazemnejad et al., arXiv 2410.01679) claims PPO's learned value network is near chance at step-level credit assignment on math reasoning, while Monte-Carlo rollouts from the same policy are near ceiling. This repo runs that comparison locally for $0. For a partial GSM8K chain-of-thought, gold V\* is the fraction of 12 independent continuations (Qwen2.5-1.5B-Instruct 4-bit, MLX) that reach the correct answer; three estimators compete: a **simulator** (6 disjoint rollouts), a **verbalized one-pass judge** (P("Yes") from the logits of "will this end correctly?"), and a **probe** (ridge/MLP on the prefix's last hidden state, trained directly on gold V\*).

Read the design before the numbers. The simulator and the "gold standard" are the *same estimator at two sample sizes* — same policy, temperature, prompt, and answer parser — so their headline correlation is a reliability coefficient (an estimator against its higher-sample twin) and cannot fail. The one-pass judge is degenerate. The probe was originally suspected of being starved (42 examples, 1536 dims), but the 2026-07-24 follow-up removed that excuse: with cluster CV, tuned ridge, an MLP, and a learning curve, the value-probe null holds while the *same* frozen states decode prefix length at r ≈ 0.75 — so the states are usable, the value just isn't linearly there (modest readability r ≲ 0.3 stays unexcluded at 23 problems). Net: **this is a cheap local study whose setup is consistent with VinePPO's claim — rollout-based value estimates carry real step-level signal, and one-pass value reading here fails for real — but it is not an independent confirmation: the simulator side can't fail by construction, and its within-problem r ≈ 0.62 turns out to sit at the rollout-count reliability ceiling rather than measuring accuracy.**

## Results

23 problems, 69 prefixes — 3 per problem, so clustered, not independent (informative slice: screened for 0 < p(success) < 1). Single run, rollouts unseeded, one probe split, no repeats.

| Measurement | Value | What it shows / what it can't show |
|---|---|---|
| Simulator (6 rollouts) vs gold V\* (12 rollouts), pooled | r ≈ 0.878 (Spearman 0.87) | **Circular by construction.** Given the observed spread of success probabilities, sampling noise alone predicts r ≈ 0.913 between two disjoint rollout sets of the same policy; observed 0.878. It measures rollout-count reliability, not estimator quality, and cannot fail. Also 87.5% of V\* variance is between-problem, so the pooled r is mostly problem difficulty. |
| Simulator vs V\*, within-problem (difficulty removed) | r ≈ 0.62 (difficulty component alone: r ≈ 0.82) | **Reliability ceiling, not accuracy** (revised 2026-07-24). After centering out problem difficulty, 6 rollouts track which *step* of a chain raised or lowered the success probability. But a noise-only twin of the gold estimator — pure sampling noise, no real signal — predicts within-problem r ≈ 0.69 here, so 0.62 is at (below) that rollout-count-limited ceiling; it demonstrates reliability, not accuracy against an external truth. And simulator vs probe are statistically indistinguishable within-problem at this n. |
| Hidden-state probe vs V\* | r ≈ 0.06 | **Real null, not starvation** (revised 2026-07-24). The original worry was n ≪ d starvation. The follow-up ran the proper test — problem-level cluster CV, tuned ridge, a small MLP, a learning curve over training-set size — and the probe still cannot beat a mean-only predictor: ΔMSE = +0.0045 [+0.0006, +0.0096] (positive = worse than the mean). The *same* frozen states, *same* pipeline, decode prefix length at r ≈ 0.75, so the states are usable and the probe isn't starved — the linear value signal genuinely isn't there. Modest readability (r ≲ 0.3) stays unexcluded at 23 problems. Still says nothing about VinePPO's fine-tuned value net, which trains a full network on vastly more data. |
| Verbalized one-pass judge vs V\* | r ≈ −0.2 | Null under problem-level bootstrap (CI includes 0), and the judge is degenerate (median P(Yes) = 0.07 — it says "no" to nearly everything). Shows this prompt failed, not that verbalized value estimation fails in general. |
| Within-chain prefix ordering (mean Spearman) | simulator 0.45; verbalized 0.01; probe 0.03 | Directionally matches the rows above, but each chain has only 3 prefixes, so per-chain Spearman is extremely coarse. |
| Branch ranking: pick the better next step (gold gap ≥ 0.15) | simulator 0.83; verbalized 0.67; probe 0.67 (chance 0.50) | n = 6 pairs — anecdotal. |
| Probe capacity check (analyze2.py, 3-fold) | ridge r ≈ 0.02; MLP (256→64) r ≈ −0.03 | MLP does not beat ridge. The 2026-07-24 follow-up settled the "encoded vs starved" ambiguity this row left open: with cluster CV and a positive control that passes (prefix length decodes at r ≈ 0.75), the value null is real. |
| Cost | 242k generated tokens, ~51 tok/s, ≈ 80 min on an M4 Mac (MLX), $0 | |

## Follow-up experiments (2026-07-24)

A follow-up sprint ran the analyses the first pass had left open, all on the *saved* data — no new generation. Five findings; they revise the rows above rather than add new claims. (These runs live in a private research archive; only their results are summarized here.)

- **The probe null is real, not starvation.** Rebuilt the probe the right way: problem-level cluster CV (train/test never share a problem), a tuned ridge, a small MLP, and a learning curve over training-set size. The probe still cannot beat a mean-only predictor of V\* — ΔMSE = +0.0045 [+0.0006, +0.0096], i.e. slightly *worse* than the mean, CI above 0. The excuse is gone because the positive control passes: the *identical* pipeline decodes prefix length from the same frozen states at r ≈ 0.75. The states are usable; the value simply isn't linearly in them. Modest readability (r ≲ 0.3) remains unexcluded at 23 problems.

- **The simulator's within-problem r ≈ 0.62 is at its reliability ceiling.** Built a noise-only twin of the gold estimator — same rollout counts, but success draws replaced by pure sampling noise around each prefix's difficulty, so *by construction there is no real step signal*. It predicts within-problem r ≈ 0.69. So 0.62 doesn't demonstrate accurate credit assignment; it demonstrates rollout-count-limited reliability. Within-problem, simulator and probe are statistically indistinguishable at this n — the gap in the pooled numbers is difficulty, which the simulator gets for free by being a rollout of the same policy.

- **Two budget scales, from rollout subsampling.** Subsampling the 12 gold rollouts and re-fitting: **problem difficulty** saturates almost immediately (needs λ ≈ 1.15 rollouts — 1–2 rollouts already pin how hard a problem is), while **step-level value** needs an order of magnitude more (λ ≈ 7.4 [3.6, 25.7] rollouts, ≈ 1000 tokens). Reading: a cheap single-rollout value estimate is mostly reading problem difficulty, not doing credit assignment — the step signal only stabilizes around ~7 rollouts.

- **Verbalized step-value judging stays null** (r ≈ −0.2) under problem-level bootstrap — same conclusion as before, now with the clustered resampling done correctly.

Net effect on the story: the positive half (rollouts carry step signal) is now bounded to a *reliability* statement, and the negative half (one-pass reading fails here) is now a *real* null with a passing positive control rather than an underpowered one. Both halves got more honest; neither flipped.

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
run.py                  screen GSM8K problems, build step prefixes, compute V* (12 rollouts),
                        simulator (6 rollouts), verbalized P(Yes), hidden states; append-only/resumable
analyze.py              metrics: correlation with V*, within-chain ordering, branch-point ranking
analyze2.py             re-analysis of saved data: signal vs remaining steps, difficulty-vs-step
                        variance decomposition, MLP-vs-ridge probe capacity
results/data.jsonl      per-prefix records incl. 1536-dim last-token hidden states (for probe extensions)
results/rank.jsonl      branch-point next-step candidate pairs with per-candidate values
results/full_run.log    console log of the full run
results/*_smoke.jsonl   smoke-test outputs
```

## Limitations

The central one: the experiment is asymmetric. The simulator is compared against a higher-sample copy of itself (shared policy, temperature, prompt, parser), so its side of the ledger cannot come out badly — and the follow-up showed its within-problem r ≈ 0.62 sits at the rollout-count reliability ceiling, so even that number reports reliability, not accuracy. The one-pass side fields a degenerate judge (still null under problem-level bootstrap) and a probe whose null the follow-up upgraded from "maybe starved" to real, using a positive control that passes on the same states. What this repo still cannot do is test VinePPO's own construct: distinguishing "value is not one-pass readable at all" from "not linearly readable from a 1.5B's frozen states at 23 problems" (modest r ≲ 0.3 stays unexcluded) would need a trained value head, more problems, and V\* from a source other than the policy's own rollouts.

Everything is also small and single-shot: one run, unseeded rollouts, one probe train/test split, no repeats; 23 problems and 69 prefixes with 3 prefixes per problem (effective n is closer to 23 than 69 for pooled statistics, ICC ≈ 0.88); 6 usable ranking pairs. One model (1.5B, 4-bit) on one dataset, and screening keeps only problems the policy sometimes solves.

## Context

The setup is consistent with the broader critic-supervision picture — critics work as predictors of grounded outcomes when they simulate the future, and the one-pass estimates tried here read nothing. After the 2026-07-24 follow-up the two halves are sharper: the simulator half is a *reliability* result (its within-problem r sits at the rollout-count ceiling, not an accuracy floor), and the one-pass half is a *real* null (the value-probe fails while a prefix-length positive control passes on the same states). Still treat it as a cheap directional data point beside VinePPO, not independent evidence for it — the design can't touch a trained value head or a non-rollout V\*.

Released under the MIT License.
