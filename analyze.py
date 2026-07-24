#!/usr/bin/env python3
"""Metrics for the VinePPO value-prediction replication. Reads results/*.jsonl."""

import argparse, json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

OUT = Path(__file__).parent / "results"


def load(f):
    return [json.loads(l) for l in f.open() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tol", type=float, default=0.1, help="within-tol accuracy band")
    ap.add_argument("--gap", type=float, default=0.15, help="min gold gap for ranking pairs")
    args = ap.parse_args()
    suf = "_smoke" if args.smoke else ""
    data = load(OUT / f"data{suf}.jsonl")
    rank_path = OUT / f"rank{suf}.jsonl"
    ranks = load(rank_path) if rank_path.exists() else []

    # ---- flatten prefix-level table ------------------------------------
    rows = []
    for rec in data:
        for p in rec["prefixes"]:
            rows.append({"pid": rec["id"], **{k: p[k] for k in
                        ("k", "v_gold", "v_sim", "v_verb")}, "hidden": p["hidden"]})
    if not rows:
        print("no data")
        return
    pids = sorted({r["pid"] for r in rows})
    v_gold = np.array([r["v_gold"] for r in rows])
    print(f"{len(data)} problems, {len(rows)} prefixes; "
          f"V* mean={v_gold.mean():.2f} std={v_gold.std():.2f}")

    # ---- probe: ridge on hidden states, problem-level split -------------
    rng = np.random.RandomState(0)
    test_pids = set(rng.choice(pids, size=max(1, len(pids) * 2 // 5), replace=False))
    H = np.array([r["hidden"] for r in rows])
    tr = np.array([r["pid"] not in test_pids for r in rows])
    te = ~tr
    probe = Ridge(alpha=10.0).fit(H[tr], v_gold[tr])
    v_probe = np.clip(probe.predict(H), 0, 1)
    for r, vp in zip(rows, v_probe):
        r["v_probe"] = float(vp)

    # ---- prefix-level metrics (probe scored on test split only) ---------
    print(f"\n== value prediction vs gold V* (K_gold rollouts) ==")
    print(f"{'method':<12}{'pearson':>9}{'spearman':>10}{'|err|<'+str(args.tol):>10}{'n':>6}")
    for name, key, mask in [("simulator", "v_sim", np.ones(len(rows), bool)),
                            ("verbalized", "v_verb", np.ones(len(rows), bool)),
                            ("probe(test)", "v_probe", te)]:
        v = np.array([r[key] for r in rows])[mask]
        g = v_gold[mask]
        if g.std() < 1e-9 or v.std() < 1e-9:
            pr = sp = float("nan")
        else:
            pr = stats.pearsonr(v, g)[0]
            sp = stats.spearmanr(v, g)[0]
        acc = float(np.mean(np.abs(v - g) < args.tol))
        print(f"{name:<12}{pr:>9.3f}{sp:>10.3f}{acc:>10.2f}{int(mask.sum()):>6}")

    # ---- within-problem step ranking from prefix values ------------------
    # (does the critic order this chain's own prefixes correctly?)
    print(f"\n== within-chain prefix ordering (Spearman per problem, mean) ==")
    for name, key in [("simulator", "v_sim"), ("verbalized", "v_verb"),
                      ("probe(test)", "v_probe")]:
        cors = []
        for pid in pids:
            if key == "v_probe" and pid not in test_pids:
                continue
            sub = [r for r in rows if r["pid"] == pid]
            g = np.array([r["v_gold"] for r in sub])
            v = np.array([r[key] for r in sub])
            if len(sub) >= 3 and g.std() > 1e-9 and v.std() > 1e-9:
                cors.append(stats.spearmanr(v, g)[0])
        m = np.mean(cors) if cors else float("nan")
        print(f"{name:<12} mean spearman = {m:.3f}   ({len(cors)} chains)")

    # ---- branch-point ranking (VinePPO's 'pick the better next step') ----
    if ranks:
        print(f"\n== branch ranking: pick better next-step (gold gap >= {args.gap}) ==")
        pairs = []
        for e in ranks:
            a, b = e["cands"]
            if abs(a["v_gold"] - b["v_gold"]) >= args.gap:
                hi, lo = (a, b) if a["v_gold"] > b["v_gold"] else (b, a)
                pairs.append((hi, lo))
        print(f"{len(pairs)} usable pairs (of {len(ranks)} branch points)")
        if pairs:
            Hh = np.array([p[0]["hidden"] for p in pairs])
            Hl = np.array([p[1]["hidden"] for p in pairs])
            ph, pl = probe.predict(Hh), probe.predict(Hl)
            for name, score in [
                ("simulator", np.mean([p[0]["v_sim"] > p[1]["v_sim"] for p in pairs])
                              + 0.5 * np.mean([p[0]["v_sim"] == p[1]["v_sim"] for p in pairs])),
                ("verbalized", np.mean([p[0]["v_verb"] > p[1]["v_verb"] for p in pairs])
                              + 0.5 * np.mean([p[0]["v_verb"] == p[1]["v_verb"] for p in pairs])),
                ("probe", float(np.mean(ph > pl)))]:
                print(f"{name:<12} ranking accuracy = {score:.2f}   (chance = 0.50)")
    else:
        print("\n(no ranking data)")


if __name__ == "__main__":
    main()
