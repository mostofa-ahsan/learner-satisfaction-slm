"""Greedy inference on the test set for the base and fine-tuned models (resumable).

Both models use the same 4-bit base weights that were trained on and the same system prompt (from
test_chat.jsonl). Items are generated one at a time so the per-item latency is the wall time for a
single request (prefill + decode).

Usage: python src/infer.py --which ft|base [--max-new-tokens 256]
Output: outputs/preds_{ft,base}.jsonl  {idx, qa_id, file_id, question, reference, prediction,
         latency_s, new_tokens, prompt_tokens}
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

import unsloth  # noqa: F401
from unsloth import FastLanguageModel
import torch
import yaml


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["ft", "base"], required=True)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--test", default="data/qa_clean/test_chat.jsonl")
    ap.add_argument("--test-meta", default="data/qa_clean/test.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    sixteen = vram >= cfg["vram_16bit_min_gb"]
    name = cfg["output_dir"] if args.which == "ft" else (cfg["load_model_16bit"] if sixteen else cfg["load_model"])
    model, tok = FastLanguageModel.from_pretrained(model_name=name, max_seq_length=2048,
                                                   load_in_4bit=not sixteen, load_in_16bit=sixteen)
    FastLanguageModel.for_inference(model)
    model.generation_config.temperature = None  # greedy: silence sampling-param warnings
    model.generation_config.top_p = None

    chat, meta = read_jsonl(args.test), read_jsonl(args.test_meta)
    assert len(chat) == len(meta)
    out = Path(f"outputs/preds_{args.which}.jsonl")
    done = {r["idx"] for r in read_jsonl(out)} if out.exists() else set()
    print(f"{args.which}: model {name}; {len(done)}/{len(chat)} already done", flush=True)
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    with open(out, "a", encoding="utf-8") as f:
        for i, (c, m) in enumerate(zip(chat, meta)):
            if i in done:
                continue
            msgs = c["messages"][:2]  # system + user
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True).to("cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     eos_token_id=eos, pad_token_id=tok.eos_token_id, use_cache=True)
            torch.cuda.synchronize()
            lat = time.perf_counter() - t0
            new = gen[0, ids["input_ids"].shape[1]:]
            pred = tok.decode(new, skip_special_tokens=True).strip()
            f.write(json.dumps(dict(idx=i, qa_id=m["qa_id"], file_id=m["file_id"], question=c["messages"][1]["content"],
                                    reference=c["messages"][2]["content"], prediction=pred, latency_s=round(lat, 4),
                                    new_tokens=int(new.shape[0]), prompt_tokens=int(ids["input_ids"].shape[1])),
                               ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"{args.which}: {i + 1}/{len(chat)} last latency {lat:.2f}s", flush=True)
    print(f"{args.which}: DONE", flush=True)


if __name__ == "__main__":
    main()
