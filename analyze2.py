#!/usr/bin/env python3
"""Failure-mechanism re-analysis of saved run-1 data (no new generation).

E1: one-pass accuracy vs REMAINING computation (steps left to the answer).
    H-C (computation) predicts: near the end -> better; far -> chance.
E2: variance decomposition — is the one-pass signal just problem difficulty?
    H-D (shortcut) predicts: correlation with problem-mean V* >> correlation
    with within-problem-centered V*.
Also: MLP probe vs ridge on the same saved hidden states (H-B readout, capacity axis).
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

OUT = Path(__file__).parent / "results"

data = [json.loads(l) for l in (OUT / "data.jsonl").open() if l.strip()]
rows = []
for rec in data:
    n_steps = len([l for l in rec["base_sol"].split("\n") if l.strip()
                   and not l.strip().lower().startswith("the answer is")])
    for p in rec["prefixes"]:
        rows.append({"pid": rec["id"], "k": p["k"], "remaining": n_steps - p["k"],
                     "v_gold": p["v_gold"], "v_sim": p["v_sim"], "v_verb": p["v_verb"],
                     "hidden": p["hidden"]})

v_gold = np.array([r["v_gold"] for r in rows])
pids = sorted({r["pid"] for r in rows})
print(f"{len(rows)} prefixes, {len(pids)} problems, "
      f"remaining-steps range {min(r['remaining'] for r in rows)}..{max(r['remaining'] for r in rows)}")

# -------- probes (ridge + MLP), problem-level split, out-of-fold predictions --
rng = np.random.RandomState(0)
H = np.array([r["hidden"] for r in rows])
fold = {pid: i % 3 for i, pid in enumerate(rng.permutation(pids))}
v_ridge = np.zeros(len(rows)); v_mlp = np.zeros(len(rows))
for f in range(3):
    te = np.array([fold[r["pid"]] == f for r in rows]); tr = ~te
    v_ridge[te] = Ridge(alpha=10.0).fit(H[tr], v_gold[tr]).predict(H[te])
    m = MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=2000, random_state=0,
                     alpha=1e-3).fit(H[tr], v_gold[tr])
    v_mlp[te] = m.predict(H[te])
v_ridge = np.clip(v_ridge, 0, 1); v_mlp = np.clip(v_mlp, 0, 1)
for r, a, b in zip(rows, v_ridge, v_mlp):
    r["v_probe"] = float(a); r["v_mlp"] = float(b)

def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return stats.pearsonr(x, y)[0]

METHODS = [("simulator", "v_sim"), ("verbalized", "v_verb"),
           ("probe-ridge", "v_probe"), ("probe-MLP", "v_mlp")]

# -------- E1: accuracy vs remaining computation ------------------------------
print("\n== E1: correlation with V* by REMAINING steps (H-C signature) ==")
bins = [(0, 0, "0 (last step)"), (1, 1, "1"), (2, 3, "2-3"), (4, 99, "4+")]
hdr = f"{'remaining':<14}" + "".join(f"{n:>13}" for n, _ in METHODS) + f"{'n':>5}"
print(hdr)
for lo, hi, label in bins:
    sel = [i for i, r in enumerate(rows) if lo <= r["remaining"] <= hi]
    if len(sel) < 5:
        print(f"{label:<14}" + " " * 13 * len(METHODS) + f"{len(sel):>5}  (skip)")
        continue
    line = f"{label:<14}"
    for _, key in METHODS:
        line += f"{corr([rows[i][key] for i in sel], v_gold[sel]):>13.3f}"
    print(line + f"{len(sel):>5}")

# -------- E2: shortcut decomposition ------------------------------------------
print("\n== E2: problem-difficulty vs within-problem signal (H-D signature) ==")
pmean = {pid: v_gold[[i for i, r in enumerate(rows) if r["pid"] == pid]].mean()
         for pid in pids}
between = np.array([pmean[r["pid"]] for r in rows])          # problem difficulty
within = v_gold - between                                     # step progress
print(f"V* variance split: between-problem {between.std()**2:.3f} / "
      f"within-problem {within.std()**2:.3f}")
print(f"{'method':<13}{'corr w/ difficulty':>20}{'corr w/ step-signal':>21}")
for name, key in METHODS:
    v = np.array([r[key] for r in rows], float)
    vc = v - np.array([np.mean([rows[i][key] for i, r2 in enumerate(rows)
                                 if r2['pid'] == r['pid']]) for r in rows])
    print(f"{name:<13}{corr(v, between):>20.3f}{corr(vc, within):>21.3f}")

# -------- probe capacity summary ----------------------------------------------
print("\n== probe capacity (H-B, readout axis) ==")
print(f"ridge  vs V*: r={corr(v_ridge, v_gold):.3f}")
print(f"MLP    vs V*: r={corr(v_mlp, v_gold):.3f}   "
      "(if MLP >> ridge, failure was linear readout, not encoding)")
