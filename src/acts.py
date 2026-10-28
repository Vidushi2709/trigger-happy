"""Hidden states at the last prompt token, every layer, for the probe experiment (method from week 06's acts.py).
Prompts: matched xLAM pairs (same request, same tool count; the called tool is either listed or removed)
plus BFCL irrelevance (no tool fits) and BFCL multiple (one of several tools fits).

python src/acts.py                                                         # untouched Qwen3-1.7B
python src/acts.py --adapter adapters/Qwen3-1.7B_n8000_nocall0.1_r16_s0
python src/acts.py --model Qwen/Qwen3-0.6B --n-pairs 10 --bfcl-limit 5 --device cpu   # smoke test
python src/acts.py --append simple                                         # add BFCL simple to an existing run
Writes results/acts/<tag>/acts.npy [prompts, layers+1, hidden] float16 and meta.json (one entry per prompt, same order).
"""
import argparse
import json
import pathlib
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import MAX_TOKENS, bfcl_questions, norm, tool
from evaluate import load, schema

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEED = 0


def chat(tok, user, tools):
    return tok.apply_chat_template(user, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def pairs(tok, n):
    rows = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    used = {e["id"] for f in ["data/train_calls.jsonl", "data/heldout.jsonl"] for e in map(json.loads, open(ROOT / f, encoding="utf-8"))}
    bfcl = [set(q.split()) for q in bfcl_questions()]
    order = list(range(len(rows)))
    rng = random.Random(SEED)
    rng.shuffle(order)
    out = []
    for i in order:
        if len(out) == 2 * n:
            break
        r = rows[i]
        tools, called = json.loads(r["tools"]), {c["name"] for c in json.loads(r["answers"])}
        other = [t for t in tools if t["name"] not in called]
        # duplicate tool names (161 single-call rows) would make nofit shorter than fit
        if f"calls_{i}" in used or len(called) != 1 or not other or len({t["name"] for t in tools}) != len(tools):
            continue
        w = set(norm(r["query"]).split())
        if any(len(w & b) / len(w | b) >= 0.8 for b in bfcl if b):  # same leak rule as data.py
            continue
        drop = rng.choice(other)
        fit, nofit = [t for t in tools if t is not drop], other
        assert len(fit) == len(nofit) and called <= {t["name"] for t in fit} and not called & {t["name"] for t in nofit}
        user = [{"role": "user", "content": r["query"]}]
        p = {s: chat(tok, user, [tool(t) for t in ts]) for s, ts in [("fit", fit), ("nofit", nofit)]}
        if max(len(tok(x)["input_ids"]) for x in p.values()) > MAX_TOKENS:
            continue
        out += [{"set": s, "id": f"calls_{i}", "pair": i, "prompt": p[s]} for s in ("fit", "nofit")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter")
    ap.add_argument("--n-pairs", type=int, default=1000)
    ap.add_argument("--bfcl-limit", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--append", help="BFCL category to add to an existing run, e.g. simple")
    args = ap.parse_args()

    name = pathlib.Path(args.adapter).name if args.adapter else args.model.split("/")[-1]
    tag = name + (f"-n{args.n_pairs}" if args.n_pairs != 1000 or args.bfcl_limit else "")
    out = ROOT / "results/acts" / tag
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.append:
        meta = json.load(open(out / "meta.json"))
        assert f"bfcl_{args.append}" not in {e["set"] for e in meta["prompts"]}, f"{tag} already has {args.append}"
    prompts = [] if args.append else pairs(tok, args.n_pairs)
    for c in [args.append] if args.append else ["irrelevance", "multiple"]:
        for r, _ in load(c)[: args.bfcl_limit]:  # prompt built exactly as in evaluate.py
            p = chat(tok, r["question"][0], [{"type": "function", "function": schema(fn)} for fn in r["function"]])
            prompts.append({"set": f"bfcl_{c}", "id": r["id"], "pair": -1, "prompt": p})

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    acts = []
    for k, e in enumerate(prompts):
        ids = tok(e["prompt"], return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        acts.append(torch.stack([h[0, -1] for h in hs]).float().cpu().half().numpy())
        e["ntok"] = ids.shape[1]
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(prompts)}", flush=True)

    acts = np.stack(acts)
    if args.append:
        acts, prompts = np.concatenate([np.load(out / "acts.npy"), acts]), meta["prompts"] + prompts
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "acts.npy", acts)
    json.dump({"tag": tag, "bfcl_results": f"results/{name}_bfcl.jsonl", "prompts": prompts}, open(out / "meta.json", "w"))
    print(f"{tag}: {acts.shape}, {sum(e['set'] == 'fit' for e in prompts)} pairs")


if __name__ == "__main__":
    main()
