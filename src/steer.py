"""Steering: push the fine-tuned model along its own "no tool fits" direction while it answers, and re-score BFCL.

python src/steer.py --adapter adapters/Qwen3-1.7B_n8000_nocall0.1_r16_s0 --alphas 0,1,2,4
python src/steer.py --adapter adapters/Qwen3-1.7B_n8000_nocall0.1_r16_s0 --alphas 0,2 --per-cat 5 --n-irr 10   # smoke test
python src/steer.py --adapter adapters/Qwen3-1.7B_n8000_nocall0.1_r16_s0 --first --alphas 0,2 --heldout   # fair test
Direction from probe.py (results/acts/<tag>/directions.npz): "calls" = no-fit minus fit among prompts where the model
is about to call (the knowledge direction, default), "all" = over all xLAM pairs (mostly the call/refuse decision).
Strength alpha is in units of that difference of means: alpha 1 moves a state by the full gap between the two groups.
It is added to every position at the output of the chosen layer, or with --first only at the last prompt token. Prompts: all 240 irrelevance + --per-cat random
prompts from each other category (seed 0), the same for every alpha. Writes results/steer_<tag>_<dir>_L<layer>.jsonl
(resumable) and prints accuracy per alpha. --heldout instead uses prompts never used to pick a strength: the rest of
the four call categories (~800) plus 300 BFCL live_irrelevance prompts (real user requests, up to 1,024 tokens).
"""
import argparse
import json
import pathlib
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import CATEGORIES, correct, load, parse_calls, schema, wilson

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dir", default="calls", choices=["calls", "all"])
    ap.add_argument("--layer", type=int, help="default: the layer probe.py picked")
    ap.add_argument("--first", action="store_true", help="push only at the last prompt token, where the call/refuse choice is made")
    ap.add_argument("--alphas", default="0,1,2,4")
    ap.add_argument("--per-cat", type=int, default=50)
    ap.add_argument("--n-irr", type=int, help="limit irrelevance prompts (smoke test)")
    ap.add_argument("--heldout", action="store_true", help="score on prompts not used to pick the strength")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    tag = pathlib.Path(args.adapter).name
    D = np.load(ROOT / f"results/acts/{tag}/directions.npz")
    layer = args.layer or int(D["layer"])
    vec = torch.tensor(D[args.dir][layer], dtype=torch.bfloat16, device="cuda")
    out = ROOT / "results" / f"steer_{tag}_{args.dir}_L{layer}{'_first' if args.first else ''}{'_heldout' if args.heldout else ''}{'-smoke' if args.n_irr else ''}.jsonl"
    done = {(r["alpha"], r["id"]): r for r in map(json.loads, open(out, encoding="utf-8"))} if out.exists() else {}

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    chat = lambda r: tok.apply_chat_template(r["question"][0], tools=[{"type": "function", "function": schema(fn)} for fn in r["function"]],
                                             add_generation_prompt=True, tokenize=False, enable_thinking=False)
    rng = random.Random(0)
    picked = {c: rng.sample(load(c), args.per_cat) for c in CATEGORIES[:-1]}  # the same draws every run
    if args.heldout:
        prompts = []
        for c in CATEGORIES[:-1]:
            seen = {r["id"] for r, _ in picked[c]}
            prompts += [(c, r, t) for r, t in load(c) if r["id"] not in seen]
        # ponytail: drops live prompts over 1,024 tokens (a few reach 6,000 and won't fit a batch in 6 GB), so it leans short
        live = [(r, t) for r, t in load("live_irrelevance") if len(tok(chat(r))["input_ids"]) <= 1024]
        prompts += [("live_irrelevance", r, t) for r, t in random.Random(1).sample(live, 300)]
    else:
        prompts = [("irrelevance", r, t) for r, t in load("irrelevance")[: args.n_irr]]
        prompts += [(c, r, t) for c in CATEGORIES[:-1] for r, t in picked[c]]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    alpha = [0.0]

    def hook(_, __, output):  # hidden_states[layer] is the output of decoder layer layer-1
        h = output[0] if isinstance(output, tuple) else output
        if args.first:  # only the prompt pass (left padding, so -1 is every prompt's last token); generated tokens untouched
            if h.shape[1] == 1:
                return output
            h = h.clone()
            h[:, -1] += alpha[0] * vec
        else:
            h = h + alpha[0] * vec
        return (h,) + output[1:] if isinstance(output, tuple) else h
    model.get_base_model().model.layers[layer - 1].register_forward_hook(hook)
    print(f"{tag}: direction '{args.dir}' at layer {layer}{' (last prompt token only)' if args.first else ''}, |d| = {vec.float().norm():.1f}, {len(prompts)} prompts")

    with open(out, "a", encoding="utf-8") as f:
        for a in map(float, args.alphas.split(",")):
            alpha[0] = a
            todo = [p for p in prompts if (a, p[1]["id"]) not in done]
            for i in range(0, len(todo), args.batch):
                chunk = todo[i : i + args.batch]
                text = [chat(r) for _, r, _ in chunk]
                ids = tok(text, return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    gen = model.generate(**ids, max_new_tokens=384, do_sample=False)  # same settings as evaluate.py
                for (c, r, t), g in zip(chunk, gen):
                    o = tok.decode(g[ids["input_ids"].shape[1] :], skip_special_tokens=True)
                    row = {"alpha": a, "id": r["id"], "category": c, "output": o, "correct": correct(parse_calls(o), t)}
                    done[(a, r["id"])] = row
                    f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"alpha {a}: {min(i + args.batch, len(todo))}/{len(todo)}", flush=True)

    print(f"\n{out.name}")
    for a in sorted({k[0] for k in done}):
        line = []
        for c in CATEGORIES + ["live_irrelevance"]:
            ks = [r["correct"] for (x, _), r in done.items() if x == a and r["category"] == c]
            if not ks:
                continue
            lo, hi = wilson(sum(ks), len(ks))
            line.append(f"{c} {sum(ks)}/{len(ks)} ({lo:.0%}-{hi:.0%})")
        print(f"  alpha {a:4.1f}: " + " | ".join(line))


if __name__ == "__main__":
    main()
