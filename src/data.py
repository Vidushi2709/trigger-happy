"""Build training and held-out files from xLAM 60k + Hammer's xLAM-irrelevance.

python src/data.py
Writes data/train_calls.jsonl, data/train_nocall.jsonl, data/heldout.jsonl.
Each line: {"id", "prompt", "completion", "n_tokens"}; prompt is exactly what evaluate.py feeds the model.
train.py mixes the two train files at the chosen no-call share.
"""
import json
import pathlib
import random
import re
import string

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
N_CALLS, N_NOCALL, HELD_CALLS, HELD_NOCALL = 8000, 2700, 500, 100
MAX_TOKENS = 1024
REFUSAL = "None of the available tools can do that, so I can't make a call for this request."
SEED = 0


def ptype(t):
    """xLAM types ("str, optional", "List[int]", "Dict", ...) -> JSON schema types."""
    base = t.split(",")[0].strip().lower()
    if base.startswith(("list", "tuple", "set")):
        return "array"
    return {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "dict": "object"}.get(base, "string")


def tool(t):
    props = {k: {"type": ptype(p["type"]), "description": p["description"]} | ({"default": p["default"]} if "default" in p else {})
             for k, p in t["parameters"].items()}
    required = [k for k, p in t["parameters"].items() if "optional" not in p["type"] and "default" not in p]
    return {"type": "function", "function": {"name": t["name"], "description": t["description"],
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


def norm(q):
    return " ".join(q.lower().translate(str.maketrans("", "", string.punctuation)).split())


def bfcl_questions():
    qs = []
    for c in ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]:
        for line in open(ROOT / f"data/bfcl/BFCL_v3_{c}.json", encoding="utf-8"):
            qs.append(norm(json.loads(line)["question"][0][-1]["content"]))
    return qs


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")  # 0.6B / 1.7B / 4B share this tokenizer and template
    bfcl = bfcl_questions()
    bfcl_exact, bfcl_words = set(bfcl), [set(q.split()) for q in bfcl]

    def leaks(q):
        # ponytail: exact match + word-overlap (Jaccard >= 0.8) against all 1,240 BFCL questions; misses paraphrases
        n, w = norm(q), set(norm(q).split())
        return n in bfcl_exact or any(len(w & b) / len(w | b) >= 0.8 for b in bfcl_words if b)

    def example(i, row):
        tools = [tool(t) for t in json.loads(row["tools"])]
        calls = json.loads(row["answers"])
        user = [{"role": "user", "content": row["query"]}]
        reply = ({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": c} for c in calls]}
                 if calls else {"role": "assistant", "content": REFUSAL})
        prompt = tok.apply_chat_template(user, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        full = tok.apply_chat_template(user + [reply], tools=tools, tokenize=False, enable_thinking=False)
        assert full.startswith(prompt)
        completion = full[len(prompt):]
        return {"id": i, "prompt": prompt, "completion": completion, "n_tokens": len(tok(full)["input_ids"])}

    stats = {}
    for name, repo, n_train, n_held in [("calls", "Salesforce/xlam-function-calling-60k", N_CALLS, HELD_CALLS),
                                         ("nocall", "MadeAgents/xlam-irrelevance-7.5k", N_NOCALL, HELD_NOCALL)]:
        rows = list(load_dataset(repo, split="train"))
        order = list(range(len(rows)))
        random.Random(SEED).shuffle(order)
        kept, dropped = [], {"bfcl_overlap": 0, "too_long": 0}
        for i in order:
            if len(kept) == n_train + n_held:
                break
            if rows[i]["tools"] == "[]":  # ~18% of the no-call set lists no tools; BFCL never does, so they teach nothing
                dropped["no_tools"] = dropped.get("no_tools", 0) + 1
                continue
            if leaks(rows[i]["query"]):
                dropped["bfcl_overlap"] += 1
                continue
            ex = example(f"{name}_{i}", rows[i])
            if ex["n_tokens"] > MAX_TOKENS:
                dropped["too_long"] += 1
                continue
            kept.append(ex)
        held, train = kept[:n_held], kept[n_held:]
        with open(ROOT / f"data/train_{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e) + "\n" for e in train)
        with open(ROOT / "data/heldout.jsonl", "a" if name == "nocall" else "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e) + "\n" for e in held)
        lens = sorted(e["n_tokens"] for e in kept)
        stats[name] = {"train": len(train), "heldout": len(held), "dropped": dropped, "scanned": sum(dropped.values()) + len(kept),
                       "tokens_median": lens[len(lens) // 2], "tokens_p95": lens[int(len(lens) * 0.95)], "tokens_max": lens[-1]}
    json.dump(stats | {"seed": SEED, "max_tokens": MAX_TOKENS, "refusal": REFUSAL}, open(ROOT / "data/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
