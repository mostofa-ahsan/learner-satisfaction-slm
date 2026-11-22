"""LoRA SFT of Llama-3.1-8B-Instruct with Unsloth (chapter replication).

- Llama-3.1 chat template, loss on assistant tokens only (unsloth train_on_responses_only).
- 2 % of train PDFs held out as validation; eval loss every eval_steps; checkpoints every save_steps.
- Resumes from the last checkpoint in checkpoint_dir automatically.
- --benchmark N: run N steps only and write the s/step estimate to outputs/benchmark.json.
- --subset N: train on a random subset of N training examples (seed 42) when the full epoch does not fit.

Usage: python src/train_lora.py [--config configs/train.yaml] [--benchmark 20] [--subset N]
       [--grad-ckpt] [--max-seq-len 1024]
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

import unsloth  # noqa: F401  (must be imported before transformers/trl)
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import torch
import yaml
from datasets import Dataset
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

INSTR = "<|start_header_id|>user<|end_header_id|>\n\n"
RESP = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class TimingCallback(TrainerCallback):
    def __init__(self, path):
        self.path, self.t0, self.s0 = path, None, None

    def on_step_end(self, args, state, control, **kw):
        if self.t0 is None:  # skip the first (warm-up / compile) step
            self.t0, self.s0 = time.time(), state.global_step
            return
        if state.global_step % 10 == 0:
            sps = (time.time() - self.t0) / max(1, state.global_step - self.s0)
            info = dict(step=state.global_step, max_steps=state.max_steps, sec_per_step=sps,
                        eta_min=(state.max_steps - state.global_step) * sps / 60,
                        peak_vram_gb=torch.cuda.max_memory_reserved() / 2**30)
            Path(self.path).write_text(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--benchmark", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tc, lc = cfg["train"], cfg["lora"]
    seed = tc["seed"]
    max_len = args.max_seq_len or tc["max_seq_len"]
    gc = "unsloth" if (args.grad_ckpt or tc["gradient_checkpointing"]) else False

    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    sixteen = vram >= cfg["vram_16bit_min_gb"]
    model_name = cfg["load_model_16bit"] if sixteen else cfg["load_model"]
    bf16 = torch.cuda.is_bf16_supported()

    # ---- data: by-PDF validation hold-out
    chat = read_jsonl(cfg["train_file"])
    meta = read_jsonl(cfg["train_meta"])
    assert len(chat) == len(meta) and all(c["messages"][1]["content"] == m["question"] for c, m in zip(chat, meta))
    pdfs = sorted({m["file_id"] for m in meta})
    rng = random.Random(seed)
    shuffled = pdfs[:]
    rng.shuffle(shuffled)
    val_pdfs = set(shuffled[:max(1, math.ceil(len(pdfs) * cfg["val_frac_pdfs"]))])
    train_rows = [c for c, m in zip(chat, meta) if m["file_id"] not in val_pdfs]
    val_rows = [c for c, m in zip(chat, meta) if m["file_id"] in val_pdfs]
    if args.subset and args.subset < len(train_rows):
        train_rows = random.Random(seed).sample(train_rows, args.subset)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_len, dtype=torch.bfloat16 if bf16 else torch.float16,
        load_in_4bit=not sixteen, load_in_16bit=sixteen, use_gradient_checkpointing=gc, random_state=seed)
    model = FastLanguageModel.get_peft_model(
        model, r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["target_modules"],
        bias="none", use_gradient_checkpointing=gc, random_state=seed)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    # 4-bit params are packed 2 per element; count the base as the unquantized 8.03B for the % (as the chapter)
    total_logical = sum(p.numel() * (2 if getattr(p, "quant_state", None) is not None or p.dtype == torch.uint8 else 1)
                        for p in model.parameters())

    def fmt(rows):
        return Dataset.from_list([{"text": tok.apply_chat_template(r["messages"], tokenize=False)} for r in rows])

    train_ds, val_ds = fmt(train_rows), fmt(val_rows)
    out_dir, ck_dir = Path(cfg["output_dir"]), Path(cfg["checkpoint_dir"])
    if args.benchmark:
        ck_dir = ck_dir.parent / "bench_checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)

    sft = SFTConfig(
        output_dir=str(ck_dir), dataset_text_field="text", max_length=max_len, packing=False,
        num_train_epochs=tc["epochs"], max_steps=args.benchmark or -1,
        per_device_train_batch_size=tc["per_device_batch_size"], gradient_accumulation_steps=tc["grad_accum"],
        per_device_eval_batch_size=8, learning_rate=tc["learning_rate"], lr_scheduler_type=tc["lr_scheduler"],
        warmup_ratio=tc["warmup_ratio"], weight_decay=tc["weight_decay"], optim=tc["optim"],
        logging_steps=tc["logging_steps"], eval_strategy="no" if args.benchmark else "steps",
        eval_steps=tc["eval_steps"], save_strategy="no" if args.benchmark else "steps",
        save_steps=tc["save_steps"], save_total_limit=tc["save_total_limit"], seed=seed, data_seed=seed,
        bf16=bf16, fp16=not bf16, report_to="none", dataset_num_proc=4)
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=train_ds,
                         eval_dataset=None if args.benchmark else val_ds, args=sft)
    trainer = train_on_responses_only(trainer, instruction_part=INSTR, response_part=RESP)
    # sanity: some label tokens unmasked, prompt masked
    lab = trainer.train_dataset[0]["labels"]
    n_lab = sum(l != -100 for l in lab)
    assert 0 < n_lab < len(lab), "response-only masking failed"

    timing_path = Path("outputs") / ("benchmark.json" if args.benchmark else "train_timing.json")
    trainer.add_callback(TimingCallback(timing_path))
    info = dict(model_loaded=model_name, mode="16-bit LoRA" if sixteen else "4-bit QLoRA", bf16=bf16,
                vram_gb=round(vram, 1), max_seq_len=max_len, gradient_checkpointing=gc,
                trainable_params=trainable, total_params_reported=total, total_params_logical=total_logical,
                trainable_pct=100 * trainable / total_logical, train_examples=len(train_rows),
                val_examples=len(val_rows), val_pdfs=sorted(val_pdfs), subset=args.subset or None,
                first_example_label_tokens=n_lab, first_example_tokens=len(lab))
    print("TRAIN INFO", json.dumps(info))

    last = None if args.benchmark else get_last_checkpoint(str(ck_dir))
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=last)
    info.update(train_seconds=time.time() - t0, resumed_from=last, global_steps=trainer.state.global_step,
                peak_vram_gb=torch.cuda.max_memory_reserved() / 2**30,
                peak_vram_allocated_gb=torch.cuda.max_memory_allocated() / 2**30)
    if args.benchmark:
        Path("outputs/benchmark.json").write_text(json.dumps(
            {**json.loads(timing_path.read_text()), **info, "steps_per_epoch_full":
             math.ceil(len(train_rows) / (tc["per_device_batch_size"] * tc["grad_accum"]))}, indent=1, default=str))
        print("BENCHMARK", timing_path.read_text())
        return

    final_eval = trainer.evaluate()
    info["final_eval_loss"] = final_eval.get("eval_loss")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    hist = trainer.state.log_history
    Path("outputs/train_log_history.json").write_text(json.dumps(hist, indent=1))
    Path("outputs/train_info.json").write_text(json.dumps(info, indent=1, default=str))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(*zip(*tr), label="train loss", lw=1.2)
    if ev:
        plt.plot(*zip(*ev), "o-", label="validation loss", lw=1.2)
    plt.xlabel("optimizer step")
    plt.ylabel("loss (assistant tokens)")
    plt.title("Llama-3.1-8B-Instruct LoRA SFT")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/plots/loss.png", dpi=150)
    print("DONE", json.dumps(info, default=str))


if __name__ == "__main__":
    main()
