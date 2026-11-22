# Experiment material: learner-satisfaction SLM (Llama-3.1-8B QLoRA)

Material for writing the paper. It replicates the chapter design (Unsloth LoRA SFT of Llama-3.1-8B-Instruct, Table 6.1 evaluation) on a synthetic QA set built from 157 PDFs. The judge is Claude Sonnet 5.5 instead of GPT-4.1. Built by `src/build_experiment_material.py`.

| File | Contents | Supports (paper section / table / figure) |
|---|---|---|
| `01_data/clean_summary.xlsx` | Cleaning rules and counts (26,327 → 15,265), per-PDF before/after and cap, split, 50-pair sample. | Methods: data cleaning; dataset table |
| `01_data/page_classification.csv` | Per-page features and skip reason (references, short, TOC/index, front matter) for every page of the filtered PDFs. | Methods: page filter; appendix |
| `01_data/pdf_inventory.csv` | Same inventory as CSV (no Summary sheet). | Methods: corpus |
| `01_data/pdf_inventory.xlsx` | Inventory of all 232 source PDFs: pages, words, tokens, empty/scanned flags, extraction errors, sha256 duplicates; Summary sheet. | Methods: corpus (Data collection); corpus descriptives table |
| `01_data/qa_summary.xlsx` | Raw synthetic QA set: totals (27,576 pairs → 26,327 after dedup), per-PDF counts, split, cost, 50-pair sample. | Methods: QA generation; dataset table |
| `01_data/seed_examples.json` | The 6 seed QA pairs used as style examples in the generation prompt. | Methods: QA generation; appendix |
| `01_data/selected.csv` | The 7,784 eligible pages (157 PDFs) sent to QA generation, with words/tokens per page. | Methods: page selection |
| `01_data/sft_system_prompt.txt` | System prompt used for SFT training and for both models at inference. | Methods: fine-tuning / inference |
| `01_data/test_clean.jsonl` | Cleaned test set: 640 QA pairs from 8 held-out PDFs (document-level split). | Methods: evaluation data |
| `01_data/train_sample_200.jsonl` | Random sample (seed 42) of 200 of the 14,625 cleaned training pairs. | Methods: training data examples; appendix |
| `02_results/Config.csv` | Hyperparameters, software versions, hardware, judge settings and rubric. | Methods: experimental setup table |
| `02_results/JudgeRationales.csv` | 30 judge rationales (15 per model) with question, reference and answer. | Results/Discussion: qualitative examples |
| `02_results/PerItem.csv` | Per-question predictions of both models with all metrics and judge scores/rationales (640 rows). | Results: error analysis; supplementary |
| `02_results/Table6_1.csv` | Table 6.1: base vs fine-tuned on BLEU-1/2 precision, sentence BLEU, ROUGE-L, BERTScore (mean/SD/min/max), lengths, length ratio, toxicity, latency, judge means. | Results: Table 6.1 |
| `02_results/eval_results.xlsx` | Full evaluation workbook: Table6_1, PerItem, JudgeRationales, Config. | Results: Table 6.1 |
| `02_results/judge_sign_tests.csv` | Paired sign tests (ft vs base) per judge criterion: better/worse/ties and two-sided p. | Results: Table 6.1 significance; Discussion |
| `02_results/train_info.json` | Trainable params (167.8M, 2.09 %), train/val sizes, train time, peak VRAM, final val loss. | Methods: training; Results: training |
| `02_results/train_log_history.json` | Trainer log: train loss every 10 steps, validation loss every 100 steps. | Results: loss curve (Figure) |
| `03_figures/bertscore_hist.png` | BERTScore F1 distribution, base vs fine-tuned. | Results: BERTScore figure |
| `03_figures/judge_scores.png` | Mean judge scores (±SD) per criterion, base vs fine-tuned. | Results: judge figure |
| `03_figures/length_ratio_box.png` | Predicted/reference length ratio box plot, base vs fine-tuned. | Results: length figure |
| `03_figures/loss.png` | Training and validation loss curve. | Results: training figure |
| `03_figures/pipeline.png` | Pipeline diagram: PDFs → extraction → page filter → Haiku QA generation → cleaning → QLoRA SFT → inference → metrics + Sonnet judge. | Methods: overview figure |
| `04_logs/DECISIONS.md` | Decision log for page selection, QA generation (pricing, validation, pilot, prompt fix), post-processing and cleanup. | Methods: data pipeline details; Limitations |
| `04_logs/DECISIONS_ft.md` | Fine-tuning decision log with the final summary, deviations from the chapter, Table 6.1 numbers and cost. | Methods: deviations; Results; Limitations |
| `04_logs/pilot_sample.md` | 20 random QA pairs from the generation pilot with pilot metrics. | Methods: QA quality check; appendix |
| `04_logs/preflight.md` | Hardware, disk, access checks and exact software versions. | Methods: computational environment |
| `05_prompts/judge_prompt_rubric.md` | Judge rubric (Table 6.1 definitions), message template, JSON schema, settings and pricing. | Methods: LLM-as-judge; appendix prompts |
| `05_prompts/qa_generation_prompt.md` | QA generation system prompt, seed examples, user-message builder and SFT system prompt. | Methods: QA generation; appendix prompts |

## Files left out (> 50 MB)

None. Every file is under 50 MB.

## Not included

- Raw PDFs, extracted text, full train/QA jsonl, model adapter and checkpoints (copyright and size). They are regenerated by the scripts in `src/`.
- `.env` and any API keys.
