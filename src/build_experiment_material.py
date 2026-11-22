"""Assemble experiment_material/ for paper writing (data, results, figures, logs, prompts, README).

Usage: python src/build_experiment_material.py
Re-running rebuilds the folder from data/, outputs/, logs/ and configs/. Files > 50 MB are left out
(and listed in the README) so the folder can be committed.
"""
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_material"
MAX_MB = 50
SEED = 42

sys.path.insert(0, str(ROOT / "src"))


def copy(src, dst_dir, name=None):
    src = ROOT / src
    dst = OUT / dst_dir / (name or src.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def sign_tests():
    d = pd.read_excel(ROOT / "outputs/eval_results.xlsx", sheet_name="PerItem")
    rows = []
    for k in ["relevance", "correctness", "completeness", "conciseness", "overall"]:
        a, b = d[f"judge_{k}_ft"], d[f"judge_{k}_base"]
        w, l = int((a > b).sum()), int((a < b).sum())
        n, m = w + l, min(w, l)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
        rows.append(dict(criterion=k, n_items=len(d), mean_base=round(b.mean(), 4), mean_ft=round(a.mean(), 4),
                         diff_ft_minus_base=round(a.mean() - b.mean(), 4), ft_better=w, ft_worse=l,
                         ties=len(d) - n, sign_test_p_two_sided=f"{p:.3e}"))
    return pd.DataFrame(rows)


def pipeline_png(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    steps = [
        ("PDF corpus", "232 PDFs\n(157 used)"),
        ("Text extraction", "PyMuPDF,\npage markers"),
        ("Page filter", "7,784 pages\n(refs/TOC/short out)"),
        ("QA generation", "Claude Haiku 5.5\n4 pairs/page"),
        ("Cleaning", "dedup, source refs,\ncap 400/PDF → 15,265"),
        ("QLoRA SFT", "Llama-3.1-8B-Instruct\nr=64, 1 epoch"),
        ("Inference", "base vs fine-tuned\n640 test Qs, greedy"),
        ("Evaluation", "BLEU, ROUGE-L,\nBERTScore, Detoxify\n+ Sonnet 5.5 judge"),
    ]
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.set_xlim(0, len(steps) * 1.9)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    colors = ["#e8eef7"] * 3 + ["#fdf0e1"] * 2 + ["#e6f2ea"] * 2 + ["#f1e8f5"]
    w, h, y = 1.55, 1.35, 0.45
    for i, (title, sub) in enumerate(steps):
        x = i * 1.9 + 0.15
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=colors[i], ec="#44546a", lw=1.2))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=10.5, weight="bold", color="#1f2a3a")
        ax.text(x + w / 2, y + 0.5, sub, ha="center", va="center", fontsize=8.3, color="#33404f")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.9, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#44546a", lw=1.4))
    for label, x0, x1, c in [("Corpus preparation", 0, 3, "#5b7db1"), ("Synthetic QA data", 3, 5, "#c98a3a"),
                             ("Fine-tuning & inference", 5, 7, "#4f9466"), ("Evaluation", 7, 8, "#8a5aa3")]:
        ax.plot([x0 * 1.9 + 0.2, x1 * 1.9 - 0.2], [2.0, 2.0], color=c, lw=3)
        ax.text((x0 + x1) / 2 * 1.9, 2.1, label, ha="center", va="bottom", fontsize=9.5, color=c, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    # 01_data
    for f in ["data/inventory/pdf_inventory.xlsx", "data/inventory/pdf_inventory.csv", "data/inventory/selected.csv",
              "data/inventory/page_classification.csv", "data/qa/qa_summary.xlsx", "data/qa_clean/clean_summary.xlsx",
              "configs/seed_examples.json", "configs/sft_system_prompt.txt"]:
        copy(f, "01_data")
    copy("data/qa_clean/test.jsonl", "01_data", "test_clean.jsonl")
    train = [l for l in open(ROOT / "data/qa_clean/train.jsonl", encoding="utf-8") if l.strip()]
    sample = random.Random(SEED).sample(train, 200)
    (OUT / "01_data/train_sample_200.jsonl").write_text("".join(sample), encoding="utf-8")

    # 02_results
    copy("outputs/eval_results.xlsx", "02_results")
    for sheet in ["Table6_1", "PerItem", "JudgeRationales", "Config"]:
        pd.read_excel(ROOT / "outputs/eval_results.xlsx", sheet_name=sheet).to_csv(
            OUT / "02_results" / f"{sheet}.csv", index=False)
    for f in ["outputs/train_info.json", "outputs/train_log_history.json"]:
        copy(f, "02_results")
    sign_tests().to_csv(OUT / "02_results/judge_sign_tests.csv", index=False)

    # 03_figures
    for p in sorted((ROOT / "outputs/plots").glob("*.png")):
        copy(p.relative_to(ROOT), "03_figures")
    pipeline_png(OUT / "03_figures/pipeline.png")

    # 04_logs
    for f in ["logs/DECISIONS.md", "logs/DECISIONS_ft.md", "logs/preflight.md", "logs/pilot_sample.md"]:
        copy(f, "04_logs")

    # 05_prompts
    (OUT / "05_prompts").mkdir()
    seeds = json.loads((ROOT / "configs/seed_examples.json").read_text())
    seeds = seeds["examples"] if isinstance(seeds, dict) else seeds
    sys_prompt = (ROOT / "configs/system_prompt.txt").read_text(encoding="utf-8").strip()
    gen_src = (ROOT / "src/generate_qa.py").read_text(encoding="utf-8")
    user_tpl = re.search(r'def user_message[^\n]*\n(.*?)\n\n', gen_src, re.S)
    qa_md = ["# QA generation prompt (Claude Haiku 5.5)", "",
             "Model `claude-haiku-5-5`, thinking disabled, effort `low`, structured JSON output "
             "(`{skip, pairs:[{type, question, answer}]}`), 4 pairs per page (3 paragraph + 1 summary). The system "
             "block (prompt + seed examples) is cached. Source: `configs/system_prompt.txt`, `configs/seed_examples.json`, "
             "`src/generate_qa.py`.", "", "## System prompt", "", "```text", sys_prompt, "```", "",
             "## Seed examples appended to the system prompt", ""]
    for i, e in enumerate(seeds, 1):
        qa_md += [f"{i}. **[{e.get('category', '')} / {e.get('type', 'paragraph')}]** Q: {e['question']}", "",
                  f"   A: {e['answer']}", ""]
    if user_tpl:
        qa_md += ["## User message builder (from `src/generate_qa.py`)", "", "```python", user_tpl.group(0).strip(), "```", ""]
    qa_md += ["## SFT system prompt (used in training and inference)", "", "```text",
              (ROOT / "configs/sft_system_prompt.txt").read_text(encoding="utf-8").strip(), "```", ""]
    (OUT / "05_prompts/qa_generation_prompt.md").write_text("\n".join(qa_md), encoding="utf-8")

    import judge
    judge_md = ["# LLM judge prompt and rubric (Claude Sonnet 5.5)", "",
                f"Model `{judge.MODEL}` via the Message Batches API; `thinking: {{type: adaptive}}`, "
                "`output_config: {effort: low, format: json_schema}`, no temperature/top_p, "
                f"max_tokens {judge.MAX_TOKENS}. The rubric is a cached system block. The judge is blind to model "
                "identity (opaque custom_ids, base and fine-tuned items shuffled into one batch). Source: `src/judge.py`.", "",
                "Deviation from the chapter: GPT-4.1 judge → Claude Sonnet 5.5. The completeness and conciseness "
                "anchors were written in the style of the given relevance, correctness and overall anchors "
                "(chapter text not available); verify them against Table 6.1.", "",
                "## System block (rubric)", "", "```text", judge.RUBRIC, "```", "",
                "## User message template", "", "```text",
                "QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nCANDIDATE ANSWER:\n{candidate}", "```", "",
                "## Output schema", "", "```json", json.dumps(judge.SCHEMA, indent=2), "```", "",
                f"Pricing used: batch ${judge.PRICE['batch']['in']}/${judge.PRICE['batch']['out']} per MTok "
                f"(50 % of standard ${judge.PRICE['std']['in']}/${judge.PRICE['std']['out']}); cache write "
                f"{judge.CACHE_WRITE_X}×, cache read {judge.CACHE_READ_X}×.", ""]
    (OUT / "05_prompts/judge_prompt_rubric.md").write_text("\n".join(judge_md), encoding="utf-8")

    # size check
    skipped = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.stat().st_size > MAX_MB * 2**20:
            skipped.append((str(p.relative_to(OUT)), p.stat().st_size / 2**20))
            p.unlink()
    write_readme(skipped)
    print(f"built {OUT}; skipped (> {MAX_MB} MB): {skipped or 'none'}")


DESCR = {
    "01_data/pdf_inventory.xlsx": ("Inventory of all 232 source PDFs: pages, words, tokens, empty/scanned flags, extraction errors, sha256 duplicates; Summary sheet.", "Methods: corpus (Data collection); corpus descriptives table"),
    "01_data/pdf_inventory.csv": ("Same inventory as CSV (no Summary sheet).", "Methods: corpus"),
    "01_data/selected.csv": ("The 7,784 eligible pages (157 PDFs) sent to QA generation, with words/tokens per page.", "Methods: page selection"),
    "01_data/page_classification.csv": ("Per-page features and skip reason (references, short, TOC/index, front matter) for every page of the filtered PDFs.", "Methods: page filter; appendix"),
    "01_data/qa_summary.xlsx": ("Raw synthetic QA set: totals (27,576 pairs → 26,327 after dedup), per-PDF counts, split, cost, 50-pair sample.", "Methods: QA generation; dataset table"),
    "01_data/clean_summary.xlsx": ("Cleaning rules and counts (26,327 → 15,265), per-PDF before/after and cap, split, 50-pair sample.", "Methods: data cleaning; dataset table"),
    "01_data/test_clean.jsonl": ("Cleaned test set: 640 QA pairs from 8 held-out PDFs (document-level split).", "Methods: evaluation data"),
    "01_data/train_sample_200.jsonl": ("Random sample (seed 42) of 200 of the 14,625 cleaned training pairs.", "Methods: training data examples; appendix"),
    "01_data/seed_examples.json": ("The 6 seed QA pairs used as style examples in the generation prompt.", "Methods: QA generation; appendix"),
    "01_data/sft_system_prompt.txt": ("System prompt used for SFT training and for both models at inference.", "Methods: fine-tuning / inference"),
    "02_results/eval_results.xlsx": ("Full evaluation workbook: Table6_1, PerItem, JudgeRationales, Config.", "Results: Table 6.1"),
    "02_results/Table6_1.csv": ("Table 6.1: base vs fine-tuned on BLEU-1/2 precision, sentence BLEU, ROUGE-L, BERTScore (mean/SD/min/max), lengths, length ratio, toxicity, latency, judge means.", "Results: Table 6.1"),
    "02_results/PerItem.csv": ("Per-question predictions of both models with all metrics and judge scores/rationales (640 rows).", "Results: error analysis; supplementary"),
    "02_results/JudgeRationales.csv": ("30 judge rationales (15 per model) with question, reference and answer.", "Results/Discussion: qualitative examples"),
    "02_results/Config.csv": ("Hyperparameters, software versions, hardware, judge settings and rubric.", "Methods: experimental setup table"),
    "02_results/train_info.json": ("Trainable params (167.8M, 2.09 %), train/val sizes, train time, peak VRAM, final val loss.", "Methods: training; Results: training"),
    "02_results/train_log_history.json": ("Trainer log: train loss every 10 steps, validation loss every 100 steps.", "Results: loss curve (Figure)"),
    "02_results/judge_sign_tests.csv": ("Paired sign tests (ft vs base) per judge criterion: better/worse/ties and two-sided p.", "Results: Table 6.1 significance; Discussion"),
    "03_figures/pipeline.png": ("Pipeline diagram: PDFs → extraction → page filter → Haiku QA generation → cleaning → QLoRA SFT → inference → metrics + Sonnet judge.", "Methods: overview figure"),
    "03_figures/loss.png": ("Training and validation loss curve.", "Results: training figure"),
    "03_figures/bertscore_hist.png": ("BERTScore F1 distribution, base vs fine-tuned.", "Results: BERTScore figure"),
    "03_figures/judge_scores.png": ("Mean judge scores (±SD) per criterion, base vs fine-tuned.", "Results: judge figure"),
    "03_figures/length_ratio_box.png": ("Predicted/reference length ratio box plot, base vs fine-tuned.", "Results: length figure"),
    "04_logs/DECISIONS.md": ("Decision log for page selection, QA generation (pricing, validation, pilot, prompt fix), post-processing and cleanup.", "Methods: data pipeline details; Limitations"),
    "04_logs/DECISIONS_ft.md": ("Fine-tuning decision log with the final summary, deviations from the chapter, Table 6.1 numbers and cost.", "Methods: deviations; Results; Limitations"),
    "04_logs/preflight.md": ("Hardware, disk, access checks and exact software versions.", "Methods: computational environment"),
    "04_logs/pilot_sample.md": ("20 random QA pairs from the generation pilot with pilot metrics.", "Methods: QA quality check; appendix"),
    "05_prompts/qa_generation_prompt.md": ("QA generation system prompt, seed examples, user-message builder and SFT system prompt.", "Methods: QA generation; appendix prompts"),
    "05_prompts/judge_prompt_rubric.md": ("Judge rubric (Table 6.1 definitions), message template, JSON schema, settings and pricing.", "Methods: LLM-as-judge; appendix prompts"),
}


def write_readme(skipped):
    lines = ["# Experiment material: learner-satisfaction SLM (Llama-3.1-8B QLoRA)", "",
             "Material for writing the paper. It replicates the chapter design (Unsloth LoRA SFT of "
             "Llama-3.1-8B-Instruct, Table 6.1 evaluation) on a synthetic QA set built from 157 PDFs. The judge is "
             "Claude Sonnet 5.5 instead of GPT-4.1. Built by `src/build_experiment_material.py`.", "",
             "| File | Contents | Supports (paper section / table / figure) |", "|---|---|---|"]
    for p in sorted(OUT.rglob("*")):
        rel = str(p.relative_to(OUT))
        if p.is_file() and rel != "README.md":
            d, s = DESCR.get(rel, ("", ""))
            lines.append(f"| `{rel}` | {d} | {s} |")
    lines += ["", f"## Files left out (> {MAX_MB} MB)", ""]
    lines += [f"- `{r}` ({mb:.1f} MB)" for r, mb in skipped] if skipped else ["None. Every file is under 50 MB."]
    lines += ["", "## Not included", "",
              "- Raw PDFs, extracted text, full train/QA jsonl, model adapter and checkpoints (copyright and size). "
              "They are regenerated by the scripts in `src/`.",
              "- `.env` and any API keys.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
