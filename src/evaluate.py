"""Quantitative + LLM-judge evaluation of base vs fine-tuned predictions -> outputs/eval_results.xlsx + plots.

Metrics (both models): BLEU-1 / BLEU-2 n-gram precision (corpus, sacrebleu), sentence BLEU (0-100, mean),
ROUGE-L F1 (rouge-score, stemmed), BERTScore F1 (roberta-large; mean/SD/min/max), mean predicted and
reference length (words), length ratio (pred/ref words per item), Detoxify mean toxicity, mean latency,
and judge means (Claude Sonnet 5.5, Table 6.1 rubric) from outputs/judge/results.jsonl.

Usage: python src/evaluate.py [--preds-dir outputs] [--out outputs/eval_results.xlsx] [--device cuda]
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

import numpy as np
import pandas as pd
import sacrebleu
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from rouge_score import rouge_scorer

DIMS = ["relevance", "correctness", "completeness", "conciseness", "overall"]
MODELS = {"base": "Base (Llama-3.1-8B-Instruct)", "ft": "Fine-tuned (LoRA)"}


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def xl(df):
    return df.map(lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default="outputs")
    ap.add_argument("--judge", default="outputs/judge/results.jsonl")
    ap.add_argument("--out", default="outputs/eval_results.xlsx")
    ap.add_argument("--plots", default="outputs/plots")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    plots = Path(args.plots)
    plots.mkdir(parents=True, exist_ok=True)

    preds = {w: {r["idx"]: r for r in read_jsonl(Path(args.preds_dir) / f"preds_{w}.jsonl")} for w in MODELS}
    common = sorted(set(preds["base"]) & set(preds["ft"]))
    assert common, "no predictions"
    print(f"items with both predictions: {len(common)}")

    # judge results: last successful record per custom_id
    judge = {}
    for r in read_jsonl(args.judge):
        if r.get("scores"):
            judge[(r["model"], r["idx"])] = r
    judge_cost = sum(r["cost_usd"] for r in read_jsonl(args.judge))

    from bert_score import BERTScorer
    from detoxify import Detoxify
    scorer = BERTScorer(model_type="roberta-large", lang="en", device=args.device)  # layer 17, no baseline rescale
    tox = Detoxify("original", device=args.device)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    per_item = pd.DataFrame({"idx": common})
    first = preds["ft"]
    for k in ("qa_id", "file_id", "question", "reference"):
        per_item[k] = [first[i][k] for i in common]
    refs = per_item["reference"].tolist()
    summary, extra = {}, {}
    for w in MODELS:
        hyp = [preds[w][i]["prediction"] for i in common]
        per_item[f"pred_{w}"] = hyp
        corpus = sacrebleu.corpus_bleu(hyp, [refs])
        sbleu = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyp, refs)]
        rl = [rouge.score(r, h)["rougeL"].fmeasure for h, r in zip(hyp, refs)]
        _, _, f1 = scorer.score(hyp, refs, batch_size=32)
        f1 = f1.cpu().numpy()
        toxic = np.array(tox.predict(hyp)["toxicity"])
        plen = np.array([len(h.split()) for h in hyp])
        rlen = np.array([len(r.split()) for r in refs])
        ratio = plen / np.maximum(rlen, 1)
        lat = np.array([preds[w][i]["latency_s"] for i in common])
        ntok = np.array([preds[w][i]["new_tokens"] for i in common])
        per_item[f"sent_bleu_{w}"] = np.round(sbleu, 3)
        per_item[f"rougeL_{w}"] = np.round(rl, 4)
        per_item[f"bertscore_f1_{w}"] = np.round(f1, 4)
        per_item[f"toxicity_{w}"] = np.round(toxic, 5)
        per_item[f"pred_words_{w}"] = plen
        per_item[f"length_ratio_{w}"] = np.round(ratio, 3)
        per_item[f"latency_s_{w}"] = lat
        per_item[f"new_tokens_{w}"] = ntok
        jd = [judge.get((w, i)) for i in common]
        for d in DIMS:
            per_item[f"judge_{d}_{w}"] = [j["scores"][d] if j else None for j in jd]
        per_item[f"judge_rationale_{w}"] = [j["scores"]["rationale"] if j else None for j in jd]
        js = {d: per_item[f"judge_{d}_{w}"].dropna().astype(float) for d in DIMS}
        summary[w] = {
            "BLEU-1 precision": corpus.precisions[0], "BLEU-2 precision": corpus.precisions[1],
            "Corpus BLEU (0-100)": corpus.score, "Sentence BLEU (0-100, mean)": float(np.mean(sbleu)),
            "ROUGE-L F1": float(np.mean(rl)),
            "BERTScore F1 mean": float(f1.mean()), "BERTScore F1 SD": float(f1.std(ddof=1)),
            "BERTScore F1 min": float(f1.min()), "BERTScore F1 max": float(f1.max()),
            "Mean predicted length (words)": float(plen.mean()), "Mean reference length (words)": float(rlen.mean()),
            "Length ratio (pred/ref, mean)": float(ratio.mean()),
            "Detoxify toxicity (mean)": float(toxic.mean()),
            "Mean latency (s)": float(lat.mean()), "Mean generated tokens": float(ntok.mean()),
            "Hit max_new_tokens (256) share": float((ntok >= 256).mean()),
            **{f"Judge {d.capitalize()} (1-5)": float(js[d].mean()) if len(js[d]) else None for d in DIMS},
            **{f"Judge {d.capitalize()} SD": float(js[d].std(ddof=1)) if len(js[d]) > 1 else None for d in DIMS},
            "Judge items scored": int(len(js["overall"])),
        }
        extra[w] = dict(f1=f1, ratio=ratio)

    definitions = {
        "BLEU-1 precision": "Corpus unigram precision of predictions vs references (sacrebleu, %).",
        "BLEU-2 precision": "Corpus bigram precision (sacrebleu, %).",
        "Corpus BLEU (0-100)": "sacrebleu corpus BLEU-4 (13a tokenizer).",
        "Sentence BLEU (0-100, mean)": "Mean per-item sacrebleu sentence BLEU (exp smoothing).",
        "ROUGE-L F1": "Mean per-item longest-common-subsequence F1 (rouge-score, Porter stemming).",
        "BERTScore F1 mean": "Mean BERTScore F1, roberta-large layer 17, no baseline rescaling.",
        "BERTScore F1 SD": "Standard deviation of per-item BERTScore F1.",
        "BERTScore F1 min": "Minimum per-item BERTScore F1.",
        "BERTScore F1 max": "Maximum per-item BERTScore F1.",
        "Mean predicted length (words)": "Whitespace words in the model answer.",
        "Mean reference length (words)": "Whitespace words in the reference answer.",
        "Length ratio (pred/ref, mean)": "Mean of per-item predicted/reference word counts.",
        "Detoxify toxicity (mean)": "Mean Detoxify 'original' toxicity probability of model answers.",
        "Mean latency (s)": "Wall time per item, single request, greedy, max_new_tokens 256, RTX 3090, 4-bit.",
        "Mean generated tokens": "New tokens per answer.",
        "Hit max_new_tokens (256) share": "Share of answers truncated at 256 new tokens.",
        "Judge Relevance (1-5)": "LLM judge: 1 = off-topic ... 5 = perfectly relevant.",
        "Judge Correctness (1-5)": "LLM judge: 1 = wrong ... 5 = perfectly correct (vs reference).",
        "Judge Completeness (1-5)": "LLM judge: 1 = misses all key points ... 5 = fully complete.",
        "Judge Conciseness (1-5)": "LLM judge: 1 = very verbose/padded ... 5 = perfectly concise.",
        "Judge Overall (1-5)": "LLM judge: 1 = poor ... 5 = excellent.",
        "Judge items scored": "Items with a valid judge score (Claude Sonnet 5.5, batch API).",
    }
    rows = [k for k in summary["base"] if not k.endswith(" SD") or k.startswith("BERTScore")]
    table = pd.DataFrame({
        "metric": rows,
        "definition": [definitions.get(k, "") for k in rows],
        MODELS["base"]: [summary["base"][k] for k in rows],
        MODELS["ft"]: [summary["ft"][k] for k in rows],
    })
    for k in rows:
        if k.startswith("Judge") and k in summary["base"] and not k.endswith("scored"):
            d = k.split()[1].lower()
            sb, sf = summary["base"].get(f"Judge {d.capitalize()} SD"), summary["ft"].get(f"Judge {d.capitalize()} SD")
            table.loc[table.metric == k, "SD base"] = sb
            table.loc[table.metric == k, "SD fine-tuned"] = sf
    table["difference (ft - base)"] = [
        (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a is not None and b is not None else None
        for a, b in zip(table[MODELS["base"]], table[MODELS["ft"]])]
    num = table.select_dtypes("number").columns
    table[num] = table[num].astype(float).round(4)

    # judge rationale sample (30, stratified-ish: 15 per model)
    jr = []
    rng = np.random.default_rng(42)
    for w in MODELS:
        have = per_item[per_item[f"judge_overall_{w}"].notna()]
        for _, r in have.sample(min(15, len(have)), random_state=42).iterrows():
            jr.append(dict(model=MODELS[w], idx=r["idx"], qa_id=r["qa_id"], question=r["question"],
                           reference=r["reference"], prediction=r[f"pred_{w}"],
                           **{d: r[f"judge_{d}_{w}"] for d in DIMS}, rationale=r[f"judge_rationale_{w}"]))
    jr = pd.DataFrame(jr)

    info = json.loads(Path("outputs/train_info.json").read_text()) if Path("outputs/train_info.json").exists() else {}
    import yaml
    cfg = yaml.safe_load(open("configs/train.yaml"))
    judge_state = json.loads(Path("outputs/judge/state.json").read_text()) if Path("outputs/judge/state.json").exists() else {}
    import importlib.metadata as md
    import platform
    import torch
    vers = {p: md.version(p) for p in ["unsloth", "transformers", "trl", "peft", "bitsandbytes", "accelerate",
                                       "datasets", "sacrebleu", "rouge-score", "bert-score", "detoxify", "anthropic"]}
    config = [("Base model", cfg["base_model"]), ("Weights loaded", info.get("model_loaded")),
              ("Training mode", info.get("mode")),
              *[(f"LoRA {k}", str(v)) for k, v in cfg["lora"].items()],
              *[(f"train {k}", str(v)) for k, v in cfg["train"].items()],
              ("Effective max_seq_len", info.get("max_seq_len")), ("Gradient checkpointing", str(info.get("gradient_checkpointing"))),
              ("Train examples", info.get("train_examples")), ("Validation examples (2% PDFs)", info.get("val_examples")),
              ("Validation PDFs", ", ".join(info.get("val_pdfs", []))), ("Test examples", len(common)),
              ("Trainable params", info.get("trainable_params")),
              ("Trainable % (of 8.03B)", round(100 * info["trainable_params"] / 8.03e9, 3) if info else None),
              ("Optimizer steps", info.get("global_steps")), ("Train time (min)", round(info["train_seconds"] / 60, 1) if info else None),
              ("Peak VRAM reserved (GB)", round(info["peak_vram_gb"], 2) if info else None),
              ("Final validation loss", info.get("final_eval_loss")),
              ("Inference", "greedy, max_new_tokens 256, batch 1, same system prompt, 4-bit base for both models"),
              ("Hardware", f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB), WSL2"),
              ("Python", platform.python_version()), ("torch / CUDA", f"{torch.__version__} / {torch.version.cuda}"),
              *[(p, v) for p, v in vers.items()],
              ("Judge model", "claude-sonnet-5-5 (Message Batches API)"),
              ("Judge settings", "thinking adaptive, effort low, structured JSON output (json_schema), no temperature/top_p, "
                                 "rubric in cached system block, blind to model identity, base/ft items shuffled"),
              ("Judge batches", ", ".join(v["id"] for k, v in judge_state.items() if k.startswith("batch_"))),
              ("Judge cost USD (batch price $1/$5 per MTok)", round(judge_cost, 4)),
              ("Judge rubric", (ROOT / "src" / "judge.py").read_text().split('RUBRIC = """')[1].split('"""')[0])]
    config = pd.DataFrame(config, columns=["item", "value"])
    config["value"] = config["value"].astype(str)

    with pd.ExcelWriter(args.out, engine="openpyxl") as w:
        xl(table).to_excel(w, sheet_name="Table6_1", index=False)
        xl(per_item).to_excel(w, sheet_name="PerItem", index=False)
        xl(jr).to_excel(w, sheet_name="JudgeRationales", index=False)
        xl(config).to_excel(w, sheet_name="Config", index=False)
        for name, widths in {"Table6_1": [34, 70, 26, 22, 12, 14, 20], "Config": [40, 110],
                             "JudgeRationales": [26, 6, 18, 50, 60, 60, 9, 11, 12, 11, 8, 70]}.items():
            for col, wd in zip("ABCDEFGHIJKL", widths):
                w.sheets[name].column_dimensions[col].width = wd

    # ---- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"base": "#8c8c8c", "ft": "#2f6db5"}
    plt.figure(figsize=(7, 4))
    bins = np.linspace(min(extra["base"]["f1"].min(), extra["ft"]["f1"].min()), 1.0, 40)
    for w in MODELS:
        plt.hist(extra[w]["f1"], bins=bins, alpha=0.6, color=colors[w], label=MODELS[w])
    plt.xlabel("BERTScore F1 (roberta-large)")
    plt.ylabel("test items")
    plt.legend()
    plt.title("BERTScore F1 distribution")
    plt.tight_layout()
    plt.savefig(plots / "bertscore_hist.png", dpi=150)
    plt.close()

    if any(summary[w]["Judge items scored"] for w in MODELS):
        x = np.arange(len(DIMS))
        plt.figure(figsize=(7.5, 4))
        for k, w in enumerate(MODELS):
            means = [summary[w][f"Judge {d.capitalize()} (1-5)"] or 0 for d in DIMS]
            sds = [summary[w][f"Judge {d.capitalize()} SD"] or 0 for d in DIMS]
            plt.bar(x + (k - 0.5) * 0.38, means, 0.38, yerr=sds, capsize=3, color=colors[w], label=MODELS[w])
        plt.xticks(x, [d.capitalize() for d in DIMS])
        plt.ylim(0, 5.5)
        plt.ylabel("mean judge score (1-5, ±SD)")
        plt.title("LLM judge (Claude Sonnet 5.5), Table 6.1 rubric")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots / "judge_scores.png", dpi=150)
        plt.close()

    plt.figure(figsize=(5, 4))
    plt.boxplot([extra[w]["ratio"] for w in MODELS], tick_labels=[MODELS[w] for w in MODELS], showfliers=True)
    plt.axhline(1.0, ls="--", c="k", lw=0.8)
    plt.ylabel("predicted / reference length (words)")
    plt.title("Length ratio")
    plt.tight_layout()
    plt.savefig(plots / "length_ratio_box.png", dpi=150)
    plt.close()

    print(table[["metric", MODELS["base"], MODELS["ft"], "difference (ft - base)"]].to_string(index=False))
    print(f"\njudge cost ${judge_cost:.4f}; wrote {args.out} and plots in {plots}/")


if __name__ == "__main__":
    main()
