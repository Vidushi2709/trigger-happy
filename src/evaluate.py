"""Score a model on BFCL v3 single-turn categories.

python src/evaluate.py --model Qwen/Qwen3-1.7B [--four-bit] [--limit 20]
Writes results/<tag>_bfcl.jsonl (one line per prompt, resumable) and prints accuracy per category.
"""
import argparse
import itertools
import json
import math
import pathlib
import re
import string

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
CATEGORIES = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]
ROOT = pathlib.Path(__file__).resolve().parent.parent
TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def load(category):
    get = lambda f: [json.loads(l) for l in open(hf_hub_download(REPO, f, repo_type="dataset", local_dir=ROOT / "data/bfcl"), encoding="utf-8")]
    rows = get(f"BFCL_v3_{category}.json")
    answers = {} if category.endswith("irrelevance") else {a["id"]: a["ground_truth"] for a in get(f"possible_answer/BFCL_v3_{category}.json")}
    return [(r, answers.get(r["id"], [])) for r in rows]


def schema(x):
    """BFCL's Python-style types -> JSON schema types, as the training data will use."""
    if isinstance(x, dict):
        return {k: TYPES.get(v, v) if k == "type" and isinstance(v, str) else schema(v) for k, v in x.items()}
    return [schema(v) for v in x] if isinstance(x, list) else x


def parse_calls(text):
    """[(name, args)] from <tool_call> blocks; None if any block is not valid JSON."""
    calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        try:
            c = json.loads(block)
            calls.append((c["name"], c.get("arguments") or {}))
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    return calls


def norm(v):
    # BFCL's rule: strings compare case-, space- and punctuation-insensitively; ints count as floats
    if isinstance(v, str):
        return v.lower().translate(str.maketrans("", "", string.punctuation + " "))
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    return v


def call_ok(call, truth):
    (name, args), ((tname, params),) = call, truth.items()
    if name != tname and name.replace("_", ".") != tname.replace("_", "."):
        return False
    if set(args) - set(params):
        return False  # an argument the function doesn't take
    for p, accepted in params.items():
        if p not in args:
            if "" not in accepted:
                return False  # required argument missing
        elif norm(args[p]) not in [norm(a) for a in accepted if a != ""]:
            return False
    return True


def correct(calls, truth):
    if calls is None:
        return False
    if not truth:  # irrelevance: right answer is no call
        return calls == []
    # ponytail: tries every pairing of calls to answers; fine for BFCL's <= 4 calls per prompt
    return len(calls) == len(truth) and any(all(call_ok(c, t) for c, t in zip(p, truth)) for p in itertools.permutations(calls))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--adapter", help="LoRA adapter folder from train.py, applied on top of --model")
    ap.add_argument("--limit", type=int, help="first N prompts per category (smoke test)")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    tag = (pathlib.Path(args.adapter).name if args.adapter else args.model.split("/")[-1]) + ("-4bit" if args.four_bit else "") + (f"-limit{args.limit}" if args.limit else "")
    out = ROOT / "results" / f"{tag}_bfcl.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"]: json.loads(l) for l in open(out, encoding="utf-8")} if out.exists() else {}

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) if args.four_bit else None
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", quantization_config=quant)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)

    todo = [(c, r, t) for c in CATEGORIES for r, t in load(c)[: args.limit] if r["id"] not in done]
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            prompts = [tok.apply_chat_template(r["question"][0], tools=[{"type": "function", "function": schema(fn)} for fn in r["function"]],
                                               add_generation_prompt=True, tokenize=False, enable_thinking=False) for _, r, _ in chunk]
            ids = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=384, do_sample=False)
            for (c, r, t), g in zip(chunk, gen):
                text = tok.decode(g[ids["input_ids"].shape[1] :], skip_special_tokens=True)
                row = {"id": r["id"], "category": c, "output": text, "correct": correct(parse_calls(text), t)}
                done[r["id"]] = row
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{min(i + args.batch, len(todo))}/{len(todo)}", flush=True)
    print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    print(f"\n{tag}")
    rows = list(done.values())
    for c in CATEGORIES + ["all"]:
        ks = [r["correct"] for r in rows if c in ("all", r["category"])]
        lo, hi = wilson(sum(ks), len(ks))
        print(f"  {c:18s} {sum(ks):4d}/{len(ks):<4d} {sum(ks) / max(len(ks), 1):6.1%}  (95% {lo:.1%}-{hi:.1%})")


if __name__ == "__main__":
    main()
