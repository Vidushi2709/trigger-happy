"""Figures for the write-up: baselines, training loss, fine-tune evaluation, probe AUROC by layer, steering.

python src/plots.py      # writes results/figures/*.png from files already in results/
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import wilson

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results/figures"
CATS = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]
FT = "Qwen3-1.7B_n8000_nocall0.1_r16_s{}"
FT25 = "Qwen3-1.7B_n8000_nocall0.25_r16_s{}"
# one colour per model everywhere (validated: CVD-safe as a set, aqua needs the legend, which every chart has)
COLOR = {"Qwen3-0.6B": "#008300", "Qwen3-1.7B": "#2a78d6", FT.format(0): "#eb6834", FT.format(1): "#1baf7a",
         FT25.format(0): "#eda100", FT25.format(1): "#6250d6"}
LABEL = {"Qwen3-0.6B": "Qwen3-0.6B, untouched", "Qwen3-1.7B": "Qwen3-1.7B, untouched",
         FT.format(0): "10% no-call, seed 0", FT.format(1): "10% no-call, seed 1",
         FT25.format(0): "25% no-call, seed 0", FT25.format(1): "25% no-call, seed 1"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def scores(name):
    rows = [json.loads(l) for l in open(ROOT / f"results/{name}_bfcl.jsonl", encoding="utf-8")]
    out = {}
    for c in CATS + ["all"]:
        r = [x for x in rows if c == "all" or x["category"] == c]
        k = sum(x["correct"] for x in r)
        out[c] = (k / len(r) * 100, *[v * 100 for v in wilson(k, len(r))])
    return out


def bars(models, title, path):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cats = CATS + ["all"]
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        s = scores(m)
        x = np.arange(len(cats)) + (j - (len(models) - 1) / 2) * w
        v = np.array([s[c] for c in cats])
        ax.bar(x, v[:, 0], w - 0.02, color=COLOR[m], label=LABEL[m])
        ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.set_xticks(range(len(cats)), [c.replace("_", "-") for c in cats[:-1]] + ["overall"])
    ax.set_ylim(50, 100); ax.set_ylabel("BFCL accuracy, % (95% CI)"); ax.set_title(title, loc="left", color=INK, pad=12 + 14 * -(-len(models) // 3))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / path, dpi=150); plt.close(fig)


def training():
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for m in [FT.format(0), FT.format(1), FT25.format(0), FT25.format(1)]:
        log = [x for x in json.load(open(ROOT / f"results/{m}_train.json"))["log"] if "loss" in x]
        ax.plot([x["step"] for x in log], [x["loss"] for x in log], lw=2, color=COLOR[m], label=LABEL[m])
    ax.set_yscale("log"); ax.set_xlabel("training step (16 examples each)"); ax.set_ylabel("training loss (log scale)")
    ax.set_title("Fine-tuning Qwen3-1.7B: loss per 10 steps", loc="left", color=INK); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "2_training_loss.png", dpi=150); plt.close(fig)


def probes():
    runs = {s: json.load(open(ROOT / f"results/probe_{FT.format(s)}.json")) for s in (0, 1)}
    base = runs[0]["layers"]  # the untouched model's numbers are the same in both files
    series = [("Qwen3-1.7B", base, "base")] + [(FT.format(s), runs[s]["layers"], "tuned") for s in (0, 1)]
    panels = [("fresh", "random", "A. All xLAM pairs: no tool fits vs a tool fits"),
              ("calls", "calls_random", "B. Only prompts where the model is about to call a tool"),
              ("bfcl", "bfcl_random", "C. BFCL, model calls on both: irrelevance vs simple (B's direction, unchanged)")]
    has_bfcl = "base_bfcl" in base
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (key, rkey, title) in zip(axes.flat, panels):
        if key == "bfcl" and not has_bfcl:
            ax.text(0.5, 0.5, "no BFCL simple activations yet", ha="center", color=MUTED, transform=ax.transAxes); continue
        for m, L, who in series:
            v = np.array(L[f"{who}_{key}"])
            ax.plot(v[:, 0], lw=2, color=COLOR[m], label=LABEL[m])
            ax.fill_between(range(len(v)), v[:, 1], v[:, 2], color=COLOR[m], alpha=0.15, lw=0)
            r = L[f"{who}_{rkey}"] if f"{who}_{rkey}" in L else L["tuned_random"] if rkey == "random" else None
            if r is not None:
                ax.plot(r, lw=1, ls="--", color=COLOR[m])
        ax.plot([], [], lw=1, ls="--", color=MUTED, label="random directions, 95th pct (same colour = same model)")
        ax.axhline(0.5, color=MUTED, lw=0.6); ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("layer"); ax.set_ylabel("AUROC (95% CI shaded)"); ax.set_title(title, loc="left", color=INK, fontsize=9)
    axes[0, 0].legend(loc="lower right", fontsize=7.5)
    ax = axes[1, 1]
    for s in (0, 1):
        # layer 0 is the token embedding: the last token is the same in both prompts, so there is no direction to compare
        ax.plot(range(1, len(runs[s]["layers"]["cosine"])), runs[s]["layers"]["cosine"][1:], lw=2, color=COLOR[FT.format(s)], label=LABEL[FT.format(s)])
    ax.axhline(0, color=MUTED, lw=0.6); ax.set_ylim(-0.1, 1.0); ax.set_xlabel("layer"); ax.set_ylabel("cosine similarity")
    ax.set_title("D. Probe direction before vs after fine-tuning (panel A directions)", loc="left", color=INK, fontsize=9); ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "4_probes_by_layer.png", dpi=150); plt.close(fig)


OUT.mkdir(parents=True, exist_ok=True)
bars(["Qwen3-0.6B", "Qwen3-1.7B"], "Baselines: untouched Qwen3 models on BFCL single-turn", "1_baselines.png")
training()
bars(["Qwen3-1.7B", FT.format(0), FT.format(1), FT25.format(0), FT25.format(1)],
     "Fine-tuning Qwen3-1.7B: better parallel calls, worse holding back, with 10% or 25% no-call data", "3_finetune_eval.png")
probes()


def steering():
    runs = {s: [json.loads(l) for l in open(next(ROOT.glob(f"results/steer_{FT.format(s)}_calls_L??.jsonl")), encoding="utf-8")] for s in (0, 1)}
    ids = {r["id"] for r in runs[0]}
    unt = [json.loads(l) for l in open(ROOT / "results/Qwen3-1.7B_bfcl.jsonl", encoding="utf-8")]

    def point(rs):
        irr = [r["correct"] for r in rs if r["category"] == "irrelevance"]
        oth = [r["correct"] for r in rs if r["category"] != "irrelevance"]
        return [(sum(x) / len(x) * 100, *[v * 100 for v in wilson(sum(x), len(x))]) for x in (irr, oth)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s, rows in runs.items():
        alphas = sorted({r["alpha"] for r in rows})
        pts = [point([r for r in rows if r["alpha"] == a]) for a in alphas]
        ax.plot([q[0][0] for q in pts], [q[1][0] for q in pts], lw=2, color=COLOR[FT.format(s)], marker="o", ms=6,
                label=f"fine-tuned seed {s}, steered (strength next to each point)")
        for a, q in zip(alphas, pts):
            ax.annotate(f"{a:g}", (q[0][0], q[1][0]), textcoords="offset points", xytext=(6, 4 if s == 0 else -11), color=INK, fontsize=8)
    # decision-moment steering (last prompt token only), seed 0; strengths above 3 are off the chart
    rows = [json.loads(l) for l in open(ROOT / f"results/steer_{FT.format(0)}_calls_L17_first.jsonl", encoding="utf-8")]
    alphas = [a for a in sorted({r["alpha"] for r in rows}) if a <= 3]
    pts = [point([r for r in rows if r["alpha"] == a]) for a in alphas]
    ax.plot([q[0][0] for q in pts], [q[1][0] for q in pts], lw=2, ls="--", color=COLOR[FT.format(0)], marker="D", ms=6,
            label="fine-tuned seed 0, steered at the decision moment only")
    for a, q in zip(alphas[1:], pts[1:]):  # strength 0 is the same point as the solid seed-0 line
        ax.annotate(f"{a:g}", (q[0][0], q[1][0]), textcoords="offset points", xytext=(-4, -14), color=INK, fontsize=8)
    (ix, ilo, ihi), (ox, olo, ohi) = point([r for r in unt if r["id"] in ids])
    ax.errorbar([ix], [ox], xerr=[[ix - ilo], [ihi - ix]], yerr=[[ox - olo], [ohi - ox]], fmt="s", ms=7, color=COLOR["Qwen3-1.7B"],
                ecolor=COLOR["Qwen3-1.7B"], elinewidth=0.8, capsize=2, label="Qwen3-1.7B, untouched (95% CI)")
    ax.set_xlabel("irrelevance: holds back correctly, % of 240"); ax.set_ylabel("other categories: correct, % of 200")
    ax.set_title("Steering along the 'no tool fits' direction (seed 0: layer 17, seed 1: layer 16)", fontsize=9, loc="left", color=INK); ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "5_steering.png", dpi=150); plt.close(fig)



def nocall():
    """How often the steered model answers without a tool call, push 0 vs 2, on three groups of held-out prompts."""
    rows = [json.loads(l) for f in ("first_heldout", "first_live")
            for l in open(ROOT / f"results/steer_{FT.format(0)}_calls_L17_{f}.jsonl", encoding="utf-8")]
    groups = [("Real users, no tool fits\n(live_irrelevance, 300)", {"live_irrelevance"}),
              ("Real users, a tool fits\n(live_simple + live_multiple, 558)", {"live_simple", "live_multiple"}),
              ("Synthetic, a tool fits\n(BFCL held-out, 800)", set(CATS[:-1]))]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, (a, color, label) in enumerate([(0.0, "#a8a7a2", "no push"), (2.0, COLOR[FT.format(0)], "push 2 at the decision moment")]):
        for g, (_, cats) in enumerate(groups):
            x = [r["category"] in cats and "<tool_call>" not in r["output"] for r in rows if r["alpha"] == a and r["category"] in cats]
            k, n = sum(x), len(x)
            lo, hi = wilson(k, n)
            ax.bar(g + (j - 0.5) * 0.36, k / n * 100, 0.34, color=color, label=label if g == 0 else None)
            ax.errorbar(g + (j - 0.5) * 0.36, k / n * 100, yerr=[[k / n * 100 - lo * 100], [hi * 100 - k / n * 100]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
            ax.annotate(f"{k / n:.0%}", (g + (j - 0.5) * 0.36, hi * 100), textcoords="offset points", xytext=(0, 3), ha="center", color=INK, fontsize=8)
    ax.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=8); ax.set_ylim(0, 100)
    ax.set_ylabel("answers with no tool call, % (95% CI)")
    ax.set_title("Steering adds about 19 points of 'no call' on real-user prompts, whether or not a tool fits", loc="left", color=INK, fontsize=9, pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "6_steering_nocall.png", dpi=150); plt.close(fig)


steering()
nocall()
print(sorted(p.name for p in OUT.glob("*.png")))
