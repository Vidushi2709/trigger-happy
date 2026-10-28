"""LoRA (or QLoRA) fine-tune on data/train_calls.jsonl + a share of data/train_nocall.jsonl.

python src/train.py --model Qwen/Qwen3-0.6B --n-calls 1000              # step 4 bug check
python src/train.py --model Qwen/Qwen3-1.7B --nocall-share 0.1 --seed 0  # normal model
python src/train.py --model Qwen/Qwen3-1.7B --nocall-share 0 --seed 0    # over-caller
Saves the adapter to adapters/<tag>/ and run stats to results/<tag>_train.json.
Checkpoints go to adapters/<tag>/checkpoint-N every 50 steps; if stopped, rerun the same command to resume.
"""
import argparse
import json
import pathlib
import random
import time

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = pathlib.Path(__file__).resolve().parent.parent


def rows(name):
    return [json.loads(l) for l in open(ROOT / f"data/train_{name}.jsonl", encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--nocall-share", type=float, default=0.1, help="share of the training mix that is 'call nothing'")
    ap.add_argument("--n-calls", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--four-bit", action="store_true", help="QLoRA: 4-bit frozen base")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke test)")
    args = ap.parse_args()

    calls, nocall = rows("calls")[: args.n_calls], rows("nocall")
    n_nocall = round(len(calls) * args.nocall_share / (1 - args.nocall_share))
    assert n_nocall <= len(nocall), f"need {n_nocall} no-call examples, have {len(nocall)}"
    mix = calls + nocall[:n_nocall]
    random.Random(args.seed).shuffle(mix)
    # data.py stores "...<|im_end|>\n"; TRL appends <|im_end|> unless the completion already ends with it
    data = Dataset.from_list([{"prompt": e["prompt"], "completion": e["completion"].rstrip("\n")} for e in mix])

    tag = f"{args.model.split('/')[-1]}{'-qlora' if args.four_bit else ''}_n{len(calls)}_nocall{args.nocall_share:g}_r{args.rank}_s{args.seed}"
    tok = AutoTokenizer.from_pretrained(args.model)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16) if args.four_bit else None
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", quantization_config=quant)
    if args.four_bit:
        model = prepare_model_for_kbit_training(model)

    config = SFTConfig(
        output_dir=str(ROOT / "adapters" / tag), max_length=1024, completion_only_loss=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=16, gradient_checkpointing=True,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, num_train_epochs=1, max_steps=args.max_steps,
        bf16=True, logging_steps=10, save_strategy="steps", save_steps=50, save_total_limit=2, report_to="none", seed=args.seed,
    )
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=config, train_dataset=data, processing_class=tok, peft_config=lora)

    # check what the loss sees on example 0: only the reply, ending in exactly one <|im_end|>
    ex = trainer.train_dataset[0]
    learned = [t for t in ex["labels"] if t != -100]
    print(f"{len(mix)} examples ({len(calls)} calls + {n_nocall} no-call) | example 0 learns: {tok.decode(learned)!r}")
    assert "<|im_start|>" not in tok.decode(learned) and tok.decode(learned).count("<|im_end|>") == 1, "loss covers the wrong text"
    trainer.model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    # a checkpoint every 50 steps; rerunning the same command resumes from the latest one
    resume = any((ROOT / "adapters" / tag).glob("checkpoint-*"))
    print(f"resuming from checkpoint: {resume}")
    result = trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model()
    stats = {"tag": tag, **vars(args), "examples": len(mix), "n_nocall": n_nocall, "steps": result.global_step,
             "train_loss": result.training_loss, "minutes": (time.time() - t0) / 60,
             "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "log": trainer.state.log_history}
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(stats, open(ROOT / f"results/{tag}_train.json", "w"), indent=1)
    print(f"done: {result.global_step} steps, {stats['minutes']:.1f} min, peak GPU {stats['peak_gpu_gb']:.2f} GB, loss {result.training_loss:.4f}")


if __name__ == "__main__":
    main()
