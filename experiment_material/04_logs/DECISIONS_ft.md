# Fine-tuning replication: decision log (2026-11-22)

## FINAL SUMMARY (written 2026-11-22 13:25 ET)

**Status: complete.** Trained → inference on both models → judge → evaluation. Run start 10:44, train end 11:50, inference end 12:21, judge end 12:42, evaluation end 13:16 ET. Three power-loss shutdowns happened during step 2/3; details in the Timeline. No retraining.

**Hardware:** NVIDIA RTX 3090 24 GB (driver 610.88, CUDA 13), WSL2, 31 GB RAM. Python 3.12.3, torch 2.14.1+cu130, unsloth 2026.11.9, transformers 5.17.0, trl 1.15.0, peft 0.22.0, bitsandbytes 0.50.2. Full list in `logs/preflight.md` and `requirements-ft.txt`.

**Training:**
- 14,115 train pairs (2 % of PDFs = 510 pairs held out as validation), 1 epoch = 883 steps.
- **55.8 min, peak VRAM 9.1 GB.** Train loss 2.50 → 2.02. Validation loss 2.213 (step 100) → **2.105** (final).
- Trainable **167,772,160 params (2.09 % of 8.03B)**.

**Deviations from the chapter:**
1. **Judge = Claude Sonnet 5.5** (`claude-sonnet-5-5`, Message Batches API, adaptive thinking, effort low, structured JSON) instead of GPT-4.1.
2. **4-bit QLoRA** (NF4, `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`) instead of 16-bit LoRA, because the GPU has 24 GB of VRAM, below 40 GB.
3. Both models are evaluated on the same 4-bit base.
4. **Trainable-parameter mismatch:** r = 64 on 7 modules gives 167.8M (2.09 %). The chapter's 41.9M (0.52 %) is exactly r = 16. I kept r = 64 as specified; **check the chapter for a typo.**
5. The completeness and conciseness rubric anchors were written by me in the style of the given anchors, because the chapter text isn't available. **Check them against Table 6.1.**
6. Optimizer adamw_8bit. BERTScore and Detoxify ran on CPU after the power losses (results are identical to GPU; it only affects speed).

**Table 6.1, test set (640 Qs, 8 held-out PDFs): base vs fine-tuned**
| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| BLEU-1 precision | 24.82 | **39.30** | +14.48 |
| BLEU-2 precision | 5.41 | **10.20** | +4.79 |
| Sentence BLEU (0-100) | 3.47 | **5.57** | +2.11 |
| ROUGE-L F1 | 0.188 | **0.228** | +0.040 |
| BERTScore F1 mean (SD; min–max) | 0.867 (0.013; 0.821–0.917) | **0.878** (0.015; 0.769–0.935) | +0.011 |
| Mean predicted / reference length (words) | 137.8 / 79.3 | 72.4 / 79.3 | |
| Length ratio | 1.78 | **0.94** | |
| Detoxify toxicity | 0.0006 | 0.0006 | 0 |
| Mean latency (s, batch 1, 4-bit) | 1.73 | **1.14** | −0.59 |
| Judge relevance (1-5) | **3.66** | 3.57 | −0.09 |
| Judge correctness | **2.31** | 2.10 | −0.21 |
| Judge completeness | **1.97** | 1.74 | −0.23 |
| Judge conciseness | 3.29 | **3.72** | +0.42 |
| Judge overall | **2.31** | 2.07 | −0.24 |

Paired sign tests on the judge scores (ft better / worse / tie):
- overall: 86 / 216 / 338, p = 5e-14
- correctness: 107 / 221 / 312, p = 3e-10
- completeness: 53 / 190 / 397, p = 3e-19
- conciseness: 318 / 51 / 271, p = 3e-48
- relevance: 106 / 158 / 376, p = 2e-3

**Interpretation:** fine-tuning clearly improved reference overlap (BLEU, ROUGE-L, BERTScore), length calibration (ratio 1.78 → 0.94) and latency. But **the judge rates the fine-tuned answers slightly lower on correctness, completeness and overall**; it rates them higher only on conciseness.
- The rationales point to the cause. The test PDFs are held out, and their questions ask for document-specific findings that neither model knows. The fine-tuned model learned the confident, concise reference style and often supplies plausible but unsupported specifics (sometimes invented quotes). The base model's longer, generic answers cover more reference points and are penalized less.
- This is a substantive finding for the paper, not a pipeline error. Absolute judge scores are low for both models (overall about 2.1–2.3 of 5) because the reference answers are document-specific.

**API cost (this task):**
- Judge batch: **$1.6022** (1,280 items, batch price $1 / $5 per MTok, rubric cached).
- Smoke tests: $0.0057 (standard calls) + $0.0028 (2-item test batch) + a negligible preflight call.
- **Total ≈ $1.61 of the $15 cap.**

**Files:**
- Adapter: `outputs/llama31-8b-ls-lora/`. Checkpoints: `outputs/checkpoints/checkpoint-{800,883}`.
- Predictions: `outputs/preds_{ft,base}.jsonl`. Judge: `outputs/judge/results.jsonl` and `state.json` (batch `msgbatch_01GveYNR6MoW9X9yaC7v7Fcw`).
- `outputs/eval_results.xlsx` (Table6_1, PerItem, JudgeRationales, Config).
- Plots in `outputs/plots/`: loss.png, bertscore_hist.png, judge_scores.png, length_ratio_box.png.
- Training metadata: `outputs/train_info.json`, `outputs/train_log_history.json`.
- Logs: `logs/train.log`, `logs/infer_*.log`, `logs/judge.log`, `logs/evaluate.log`, `logs/gpu_temp.log`, `logs/preflight.md`.
- Code: `src/train_lora.py`, `src/infer.py`, `src/judge.py`, `src/evaluate.py`, `configs/train.yaml` (commits 02b8e26 + the judge fsync fix).


## Preflight (10:44 ET), details in logs/preflight.md
- RTX 3090 24 GB, bf16 ok, torch 2.14.1+cu130. 38 GB disk free, below the 60 GB target. **Not treated as a blocker:** the 4-bit plan needs about 10 GB; see preflight.md.
- `.env`: HF_TOKEN **no**, ANTHROPIC_API_KEY **yes** (Sonnet 5.5 test call ok). HF access comes from a cached login on the machine, and the gated meta-llama repo is readable. No fallback was needed for access; the weights come from the Unsloth NF4 repo for disk and speed reasons (below).
- Installs: `pip --isolated --index-url https://pypi.org/simple`, because pip's config points at the unreachable NGC index. The PyTorch CUDA index wasn't needed (the torch wheel from PyPI ships CUDA 13 libraries). Pinned in `requirements-ft.txt`.

## Deviations from the chapter
1. **Judge = Claude Sonnet 5.5** (`claude-sonnet-5-5`) instead of GPT-4.1. This is the planned deviation.
2. **4-bit QLoRA instead of 16-bit LoRA.** VRAM is 24 GB, below 40 GB. Weights: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, the Meta Llama-3.1-8B-Instruct weights pre-quantized to NF4, 5.7 GB. It's identical to quantizing `meta-llama/Llama-3.1-8B-Instruct` at load time, and it saves a 16 GB download on a nearly full disk. The LoRA adapters train in bf16.
3. **Base and fine-tuned models are both evaluated on the same 4-bit base**, so the comparison isolates the adapter. Absolute base scores may be slightly below a bf16 base.
4. **Trainable parameters: 167.8M (2.09 % of 8.03B), not the chapter's 41.9M (0.52 %).** The 41.9M figure is exactly r = 16 on the 7 target modules of Llama-3.1-8B. r = 64 on the same modules is 4× that: 167,772,160. I kept **r = 64 / alpha = 128 as specified**. Either the chapter's parameter count or its rank is a typo; please check which.
5. Optimizer `adamw_8bit` (Unsloth/QLoRA default). The chapter doesn't specify one, so this is not a deviation from a stated setting.
6. Validation: 2 % of train PDFs → 3 PDFs (P0024, P0107, P0211; 510 pairs) held out. **Training uses 14,115 pairs**, not 14,625. Eval loss every 100 steps.
7. Rubric: the chapter text isn't in the project. The relevance, correctness and overall anchors are verbatim from the brief. **I wrote the completeness ("1 = misses all key points … 5 = fully complete") and conciseness ("1 = very verbose or padded … 5 = perfectly concise") anchors in the same style; check them against Table 6.1.**

## Training (`src/train_lora.py`, `configs/train.yaml`)
- Llama-3.1 chat template (tokenizer default). Loss only on assistant tokens via Unsloth `train_on_responses_only`; asserted on the first example: 86 of 171 tokens are labels.
- No packing; max_seq_len 2048 (the longest example is about 260 tokens, so nothing is truncated). Gradient checkpointing off. The planned OOM fallback (Unsloth gradient checkpointing, then 1024 tokens) wasn't needed: peak VRAM was 8.6 GB.
- Benchmark (20 steps): **3.22 s/step → 883 steps ≈ 47 min**, far under 7 h, so it trains on **the full train set, no subset**.
- Checkpoints every 200 steps in `outputs/checkpoints/` (keeps the last 3). `logs/ft_pipeline.sh` resumes from the last checkpoint.

## Inference (`src/infer.py`)
- Greedy, max_new_tokens 256, batch size 1 (so latency is per request), the same system prompt as training (`configs/sft_system_prompt.txt`, taken from test_chat.jsonl). Stops at `<|eot_id|>`. Resumable by idx.

## Judge (`src/judge.py`)
- Message Batches API, `thinking: {type: adaptive}`, `output_config: {effort: low, format: json_schema}`, no temperature/top_p. Schema: 5 integer criteria (enum 1–5) plus a rationale.
- The rubric is a cached system block (946 tokens; caching confirmed in the smoke test).
- Blind: the judge sees only question / reference / candidate. custom_ids are opaque hashes, and base and ft items are shuffled into one batch.
- Errored, expired or unparseable items are retried once. If not done within 3 h, the batch is cancelled and the rest goes through regular calls (8 workers).
- Cost: batch price $1 / $5 per MTok (50 % of $2 / $10), cache write 1.25×, cache read 0.1×. **Hard cap $15**, enforced with a worst-case reservation (full max_tokens of output for every request) before each submit.
- Smoke tests before the run: 2 standard calls ($0.0057) plus a 2-item test batch. Both worked: the identical answer scored 5/5 everywhere, an off-topic answer scored 1. Estimated full cost: about $1.5 for 1,280 items.

## Timeline
- 10:52 ET: full training started (`logs/ft_pipeline.sh`: train → infer ft → infer base → judge → evaluate).
- [2026-11-22 12:42:52] Power loss ~12:23 during judge collection: state.json said batch_1 collected but results.jsonl was empty (unflushed). Reset collected=false (backup state.json.precrash.bak) so judge.py re-downloads msgbatch_01GveYNR6MoW9X9yaC7v7Fcw instead of submitting a duplicate batch; save_state now fsyncs results first. GPU power limit 280 W NOT applied (needs admin). Resumed via logs/ft_resume.sh.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    - 11:50 ET: training finished (883 steps, 55.8 min, final val loss 2.105).
- 12:02 / 12:21 ET: ft / base inference finished (640 + 640). Judge batch `msgbatch_01GveYNR6MoW9X9yaC7v7Fcw` was submitted at 12:21 (worst-case reservation $11.20 < $15).
- **Three PC shutdowns during this phase** (suspected RTX 3090 power spikes). After the restart at 12:42 the pipeline resumed: the judge re-collected the ended batch (1,280 scored, 0 failed, $1.60). `judge.py` now fsyncs results before marking a batch collected, because one crash had left collected=true next to an empty results file. The evaluation started at 12:53 died in the next power loss (empty evaluate.log).
- 13:09 ET resume:
  - **Power limit is 420 W (default 420, max 500), above the 280 W target. WARNING:** not changed, because `nvidia-smi -pl` needs admin rights, and on WSL it has to be set from Windows (e.g. elevated `nvidia-smi -pl 280` in Windows or MSI Afterburner).
  - Started `logs/gpu_monitor.sh` (power/temp/util/clock every 120 s, synced to disk) → `logs/gpu_temp.log`.
  - Removed checkpoint-600 (keeps 2). Free disk 20 GB, above the 15 GB stop threshold.
  - Verified the adapter: it loads as a PEFT model, and greedy output for test item 0 is identical to the stored prediction.
  - Both prediction files were checked: 640 valid lines each, no duplicates, no empty answers. So no inference was re-run.
  - The judge resume made no new calls.
  - Evaluation ran on **CPU** (BERTScore, Detoxify) to keep load off the GPU. It finished at 13:16.
