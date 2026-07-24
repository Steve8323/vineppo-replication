#!/usr/bin/env python3
"""
Scaled local replication of VinePPO's value-prediction measurement
(Kazemnejad et al., arXiv 2410.01679, Figs. 5-6).

Question: can a ONE-PASS critic read V(prefix) = P(correct final answer | partial CoT)
off the text, or must you SIMULATE (MC rollouts) to get it?

Methods compared on identical GSM8K prefixes, all using the same model
(mlx-community/Qwen2.5-1.5B-Instruct-4bit, fully local on Apple M4):
  gold   V* : K_GOLD independent rollouts (ground truth)
  sim    V^ : K_SIM  disjoint rollouts     (the "simulator" critic)
  verb   V^ : one forward pass, model verbalizes P(success) as a number
  probe  V^ : ridge head on last hidden state of prefix (frozen features,
              trained on gold V* of train-split problems = MORE supervision
              per state than a real PPO value net ever gets)

Outputs results/data.jsonl (per-problem, resumable) + results/rank.jsonl.
Run analyze.py afterwards for the metrics table.
"""

import argparse, json, random, re, sys, time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
OUT = Path(__file__).parent / "results"
SYS = ("You are a careful math solver. Solve the problem step by step, "
       "with exactly one short step per line. Finish with a final line of the form: "
       "The answer is N")

# ---------------------------------------------------------------- utilities

def gold_answer(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "").replace("$", "")

def pred_answer(text: str):
    m = re.findall(r"answer is\s*\$?(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None

def is_correct(text: str, gold: str) -> bool:
    p = pred_answer(text)
    if p is None:
        return False
    try:
        return abs(float(p) - float(gold)) < 1e-4
    except ValueError:
        return False

def load_gsm8k(n: int, seed: int = 0):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = random.Random(seed).sample(range(len(ds)), n)
    return [{"id": int(i), "q": ds[int(i)]["question"], "gold": gold_answer(ds[int(i)]["answer"])}
            for i in idx]

# ---------------------------------------------------------------- model ops

class LM:
    def __init__(self, model_name=MODEL):
        print(f"loading {model_name} ...", flush=True)
        self.model, self.tok = load(model_name)
        self.n_tokens = 0
        self.t0 = time.time()

    def chat_prompt(self, question: str) -> str:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": question}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def gen(self, prompt: str, max_tokens=320, temp=0.8) -> str:
        out = generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens,
                       sampler=make_sampler(temp=temp, top_p=0.95), verbose=False)
        self.n_tokens += len(self.tok.encode(out))
        return out

    def rollout_value(self, question: str, prefix: str, k: int, gold: str,
                      max_tokens=320, temp=0.8):
        """simulator: continue the chain k times, count correct endings"""
        base = self.chat_prompt(question) + prefix
        wins = 0
        for _ in range(k):
            cont = self.gen(base, max_tokens=max_tokens, temp=temp)
            wins += is_correct(prefix + cont, gold)
        return wins / k

    def verbalized_value(self, question: str, prefix: str) -> float:
        """one-pass critic: P(Yes) from logits on 'will this end correct? Yes/No'"""
        msgs = [{"role": "system", "content":
                 "You judge partial math solutions. Answer with exactly one word: Yes or No."},
                {"role": "user", "content":
                 f"Problem:\n{question}\n\nPartial solution so far:\n{prefix}\n\n"
                 "If this solution is continued to the end, will the final answer be correct? "
                 "Answer Yes or No."}]
        p = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        toks = self.tok.encode(p)
        logits = self.model(mx.array([toks]))[0, -1, :]
        ids = []
        for w in ("Yes", "No"):
            enc = self.tok.encode(w)
            ids.append(enc[0] if len(enc) == 1 else self.tok.encode(w, add_special_tokens=False)[0])
        pair = mx.softmax(mx.array([logits[ids[0]], logits[ids[1]]]))
        return float(pair[0])

    def hidden_state(self, question: str, prefix: str):
        """last-token hidden state of the prefix (frozen features for the probe)"""
        toks = self.tok.encode(self.chat_prompt(question) + prefix)
        h = self.model.model(mx.array([toks]))
        return [float(x) for x in h[0, -1, :]]

    def speed(self):
        dt = time.time() - self.t0
        return self.n_tokens / dt if dt > 0 else 0.0

# ---------------------------------------------------------------- phases

def split_steps(sol: str):
    lines = [l.strip() for l in sol.split("\n") if l.strip()]
    return [l for l in lines if not l.lower().startswith("the answer is")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-screen", type=int, default=60)
    ap.add_argument("--n-keep", type=int, default=30)
    ap.add_argument("--k-screen", type=int, default=6)
    ap.add_argument("--k-gold", type=int, default=16)
    ap.add_argument("--k-sim", type=int, default=8)
    ap.add_argument("--k-rank", type=int, default=12)
    ap.add_argument("--max-prefixes", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    args = ap.parse_args()
    if args.smoke:
        args.n_screen, args.n_keep = 6, 2
        args.k_screen, args.k_gold, args.k_sim, args.k_rank = 3, 4, 2, 3
        args.max_prefixes = 2

    OUT.mkdir(parents=True, exist_ok=True)
    data_f = OUT / ("data_smoke.jsonl" if args.smoke else "data.jsonl")
    rank_f = OUT / ("rank_smoke.jsonl" if args.smoke else "rank.jsonl")
    done = set()
    if data_f.exists():
        done = {json.loads(l)["id"] for l in data_f.open() if l.strip()}
        print(f"resuming: {len(done)} problems already done")

    lm = LM()
    problems = load_gsm8k(args.n_screen)

    # -- phase 1: screen for informative problems (0 < p(success) < 1) -------
    kept = []
    for prob in problems:
        if len(kept) >= args.n_keep:
            break
        if prob["id"] in done:           # already fully processed earlier
            kept.append(prob)
            continue
        v = lm.rollout_value(prob["q"], "", args.k_screen, prob["gold"])
        if 0 < v < 1:
            prob["p_screen"] = v
            kept.append(prob)
        print(f"screen id={prob['id']} p={v:.2f} kept={len(kept)}  "
              f"[{lm.speed():.0f} tok/s]", flush=True)

    # -- phase 2+3: per problem: base chain -> prefixes -> gold/sim/verb/probe
    for pi, prob in enumerate(kept):
        if prob["id"] in done:
            continue
        q, gold = prob["q"], prob["gold"]
        base_sol = lm.gen(lm.chat_prompt(q), temp=0.8)
        steps = split_steps(base_sol)
        if len(steps) < 2:
            continue
        n_pref = min(args.max_prefixes, len(steps))
        rec = {"id": prob["id"], "q": q, "gold": gold, "base_sol": base_sol,
               "p_screen": prob.get("p_screen"), "prefixes": []}
        for k in range(1, n_pref + 1):
            prefix = "\n".join(steps[:k]) + "\n"
            v_gold = lm.rollout_value(q, prefix, args.k_gold, gold)
            v_sim = lm.rollout_value(q, prefix, args.k_sim, gold)
            v_verb = lm.verbalized_value(q, prefix)
            hid = lm.hidden_state(q, prefix)
            rec["prefixes"].append({"k": k, "prefix": prefix, "v_gold": v_gold,
                                    "v_sim": v_sim, "v_verb": v_verb, "hidden": hid})
            print(f"[{pi+1}/{len(kept)}] id={prob['id']} k={k} "
                  f"V*={v_gold:.2f} sim={v_sim:.2f} verb={v_verb:.2f}  "
                  f"[{lm.speed():.0f} tok/s]", flush=True)

        # -- phase 4: ranking test at the first-step branch point ------------
        prefix = steps[0] + "\n"
        cands = []
        for _ in range(2):
            c = lm.gen(lm.chat_prompt(q) + prefix, max_tokens=60, temp=1.0)
            first = c.split("\n")[0].strip()
            if first:
                cands.append(first)
        if len(cands) == 2 and cands[0] != cands[1]:
            entry = {"id": prob["id"], "prefix": prefix, "cands": []}
            for c in cands:
                p2 = prefix + c + "\n"
                entry["cands"].append({
                    "step": c,
                    "v_gold": lm.rollout_value(q, p2, args.k_rank, gold),
                    "v_sim": lm.rollout_value(q, p2, max(2, args.k_sim // 2), gold),
                    "v_verb": lm.verbalized_value(q, p2),
                    "hidden": lm.hidden_state(q, p2)})
            with rank_f.open("a") as f:
                f.write(json.dumps(entry) + "\n")

        with data_f.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"DONE. {lm.n_tokens} generated tokens, avg {lm.speed():.0f} tok/s")
    print(f"data: {data_f}\nrank: {rank_f}\nnow run: python3 analyze.py"
          + (" --smoke" if args.smoke else ""))

if __name__ == "__main__":
    main()
