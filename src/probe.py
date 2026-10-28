"""Does the fine-tuned model still know when no tool fits? Difference-of-means probes per layer (method from week 06).

python src/probe.py Qwen3-1.7B Qwen3-1.7B_n8000_nocall0.1_r16_s0     # untouched tag, fine-tuned tag (results/acts/<tag>/)
Splits xLAM pairs 70/30 by pair. At the last prompt token the model is about to choose between a tool call and a
refusal, so a fit-vs-no-fit probe there can just read that choice. So each model's choice is read off its final
hidden state, and the main test holds it fixed: among xLAM prompts where the model is about to call a tool, can a
probe still tell "no tool fits" from "a tool fits"? If yes, the model knows and calls anyway; if no, it has lost it.
Control: 200 random directions per layer (a real probe has to beat their 95th percentile).
Also: fit-vs-no-fit over all pairs (fresh, frozen untouched direction on the fine-tuned model, cosine).
BFCL irrelevance vs multiple is not compared: they differ in format (1 tool vs ~3), which layer 1 already separates.
If the acts include BFCL simple (acts.py --append simple), the xLAM direction is also applied to BFCL: irrelevance
prompts the model calls on vs simple prompts it calls on (both list 1 tool), and the subset the fine-tune broke.
Writes results/probe_<fine-tuned tag>.json (figures: plots.py).
"""
import json
import pathlib
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from scipy.stats import rankdata

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE, TUNED = sys.argv[1:3]
rng = np.random.default_rng(0)
B = 1000  # bootstrap resamples
R = 200  # random directions for the control
TOOL_CALL = 151657  # "<tool_call>"


def load(tag):
    meta = json.load(open(ROOT / f"results/acts/{tag}/meta.json"))
    return np.load(ROOT / f"results/acts/{tag}/acts.npy"), meta


def auc(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(pos, neg):
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)] if min(len(pos), len(neg)) >= 2 else [np.nan]
    return [float(auc(pos, neg)), *np.percentile(bs, [2.5, 97.5]).tolist()]


def calls(X, W):
    """Is the model about to start a tool call? Top next token from the final (normed) hidden state."""
    with torch.no_grad():
        return np.concatenate([(torch.from_numpy(X[i:i + 256, -1].astype(np.float32)) @ W.T).argmax(1).numpy() == TOOL_CALL
                               for i in range(0, len(X), 256)])


Xb, mb = load(BASE)
Xt, mt = load(TUNED)
P = mb["prompts"]
assert [(e["set"], e["id"]) for e in P] == [(e["set"], e["id"]) for e in mt["prompts"]], "acts built from different prompts"
sets, pair, ntok = (np.array([e[k] for e in P]) for k in ("set", "pair", "ntok"))
y = sets == "nofit"
u = np.unique(pair[pair >= 0])
test_pairs = rng.permutation(u)[: int(round(0.3 * len(u)))]
tr = np.isin(sets, ["fit", "nofit"]) & ~np.isin(pair, test_pairs)
te = np.isin(pair, test_pairs)

# tied embeddings, and LoRA leaves them alone, so the untouched output layer reads every model's choice
repo = "Qwen/" + BASE
shard = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]["model.embed_tokens.weight"]
with safe_open(hf_hub_download(repo, shard), "pt") as f:
    W = f.get_tensor("model.embed_tokens.weight").float()
cb, ct = calls(Xb, W), calls(Xt, W)
del W
readout = {}
for name, c, m in [("base", cb, mb), ("tuned", ct, mt)]:
    gen = {r["id"]: "<tool_call>" in r["output"] for r in map(json.loads, open(ROOT / m["bfcl_results"], encoding="utf-8"))}
    b = np.isin(sets, ["bfcl_irrelevance", "bfcl_multiple"])
    readout[name] = {"agrees_with_generated_bfcl": float(np.mean([c[i] == gen[P[i]["id"]] for i in np.where(b)[0]])),
                     **{f"calls_{s}": int((c & (sets == s)).sum()) for s in ("fit", "nofit", "bfcl_irrelevance")}}


def direction(X, m, labels=y):
    return X[m & labels].mean(0) - X[m & ~labels].mean(0), (X[m & labels].mean(0) + X[m & ~labels].mean(0)) / 2


def score(X, d, mid, m):
    return (X[m] - mid) @ d


def random_ctrl(X, pos, neg):
    """95th percentile AUROC of random directions, pos vs neg prompts."""
    D = rng.standard_normal((X.shape[1], R)).astype(np.float32)
    m = pos | neg
    S = X[m] @ D  # one product: scoring the groups separately leaves rounding differences between tied prompts
    return float(np.percentile([auc(S[pos[m], j], S[neg[m], j]) for j in range(R)], 95))


# BFCL, same format both sides (1 tool listed) and the model calls on both: irrelevance (tool doesn't fit) vs simple (fits)
has_simple = "bfcl_simple" in sets
irr_b, irr_t = (sets == "bfcl_irrelevance") & cb, (sets == "bfcl_irrelevance") & ct
sim_b, sim_t = (sets == "bfcl_simple") & cb, (sets == "bfcl_simple") & ct
broke = irr_t & ~cb  # the untouched model was about to refuse, the fine-tuned one calls


L = Xb.shape[1]
keys = ["base_fresh", "tuned_fresh", "frozen", "cosine", "base_random", "tuned_random",
        "base_calls", "tuned_calls", "base_calls_random", "tuned_calls_random", "base_calls_inner", "tuned_calls_inner"]
keys += ["base_bfcl", "tuned_bfcl", "broke_bfcl", "base_bfcl_random", "tuned_bfcl_random", "broke_bfcl_random"] if has_simple else []
res = {k: [] for k in keys}
steer = {"calls": [], "all": []}  # the fine-tuned model's directions per layer, for steer.py
for l in range(L):
    Bl, Tl = Xb[:, l].astype(np.float32), Xt[:, l].astype(np.float32)
    (db, mid_b), (dt, mid_t) = direction(Bl, tr), direction(Tl, tr)
    res["base_fresh"].append(auc_ci(score(Bl, db, mid_b, te & y), score(Bl, db, mid_b, te & ~y)))
    res["tuned_fresh"].append(auc_ci(score(Tl, dt, mid_t, te & y), score(Tl, dt, mid_t, te & ~y)))
    res["frozen"].append(auc_ci(score(Tl, db, mid_b, te & y), score(Tl, db, mid_b, te & ~y)))
    steer["all"].append(dt)
    res["cosine"].append(float(db @ dt / (np.linalg.norm(db) * np.linalg.norm(dt) + 1e-12)))
    res["base_random"].append(random_ctrl(Bl, te & y, te & ~y))
    res["tuned_random"].append(random_ctrl(Tl, te & y, te & ~y))
    # main test: only prompts where this model is about to call a tool
    for name, X, c, irr, sim in [("base", Bl, cb, irr_b, sim_b), ("tuned", Tl, ct, irr_t, sim_t)]:
        d, mid = direction(X, tr & c)
        res[f"{name}_calls"].append(auc_ci(score(X, d, mid, te & c & y), score(X, d, mid, te & c & ~y)))
        res[f"{name}_calls_random"].append(random_ctrl(X, te & c & y, te & c & ~y))
        if name == "tuned":
            steer["calls"].append(d)
        if has_simple:  # the xLAM direction, unchanged, on BFCL
            res[f"{name}_bfcl"].append(auc_ci(score(X, d, mid, irr), score(X, d, mid, sim)))
            res[f"{name}_bfcl_random"].append(random_ctrl(X, irr, sim))
            if name == "tuned":
                res["broke_bfcl"].append(auc_ci(score(X, d, mid, broke), score(X, d, mid, sim)))
                res["broke_bfcl_random"].append(random_ctrl(X, broke, sim))
        # layer choice uses training pairs only: direction on even pair ids, AUROC on odd
        d, mid = direction(X, tr & c & (pair % 2 == 0))
        o = tr & c & (pair % 2 == 1)
        res[f"{name}_calls_inner"].append(auc(score(X, d, mid, o & y), score(X, d, mid, o & ~y)))

summary = {}
for name, c in [("base", cb), ("tuned", ct)]:
    best = int(np.nanargmax(res[f"{name}_calls_inner"][1:])) + 1
    summary[name] = {"layer": best, "auroc": res[f"{name}_calls"][best], "random_95th": res[f"{name}_calls_random"][best],
                     "n_test": {"nofit_calls": int((te & c & y).sum()), "fit_calls": int((te & c & ~y).sum())},
                     "length_only": auc_ci(ntok[te & c & y], ntok[te & c & ~y])}
    if has_simple:
        irr, sim = (irr_b, sim_b) if name == "base" else (irr_t, sim_t)
        summary[name]["bfcl"] = {"auroc": res[f"{name}_bfcl"][best], "random_95th": res[f"{name}_bfcl_random"][best],
                                 "n": {"irrelevance_calls": int(irr.sum()), "simple_calls": int(sim.sum())},
                                 "length_only": auc_ci(ntok[irr], ntok[sim])}
        if name == "tuned":
            summary[name]["bfcl"]["broke"] = {"auroc": res["broke_bfcl"][best], "random_95th": res["broke_bfcl_random"][best], "n": int(broke.sum()),
                                              "length_only": auc_ci(ntok[broke], ntok[sim])}
out = {"base": BASE, "tuned": TUNED, "n_train_pairs": int(tr.sum() // 2), "n_test_pairs": int(te.sum() // 2),
       "readout": readout, "knows_but_calls": summary, "length_only_all": auc_ci(ntok[te & y], ntok[te & ~y]), "layers": res}
json.dump(out, open(ROOT / f"results/probe_{TUNED}.json", "w"), indent=1)
np.savez(ROOT / f"results/acts/{TUNED}/directions.npz", **{k: np.stack(v) for k, v in steer.items()}, layer=summary["tuned"]["layer"])

print(json.dumps({k: out[k] for k in ("readout", "knows_but_calls")}, indent=1))
print("layer  base_fresh tuned_fresh frozen cosine random | base_calls (random) tuned_calls (random)")
for l in range(0, L, 2):
    print(f"{l:5d}  {res['base_fresh'][l][0]:.3f}      {res['tuned_fresh'][l][0]:.3f}       {res['frozen'][l][0]:.3f}  {res['cosine'][l]:.2f}   "
          f"{res['tuned_random'][l]:.3f} | {res['base_calls'][l][0]:.3f} ({res['base_calls_random'][l]:.3f})      "
          f"{res['tuned_calls'][l][0]:.3f} ({res['tuned_calls_random'][l]:.3f})")
