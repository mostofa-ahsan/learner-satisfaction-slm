# Preflight (2026-11-22 10:44–10:55 ET)

**Result: PASS, with one soft warning (disk). There is no blocker.**

| Check | Result |
|---|---|
| GPU (nvidia-smi in WSL) | NVIDIA GeForce RTX 3090, **24,576 MiB VRAM**, driver 610.88 (KMD) / 610.57.01, CUDA UMD 13.3 |
| torch.cuda.is_available() | True; compute capability 8.6 |
| bf16 supported | True |
| Disk free | **38 GB before installs, 29 GB after** (target ≥ 60 GB). **WARNING, soft.** See the disk plan below. |
| RAM / CPU | 31 GB / 24 threads |
| .env HF_TOKEN | **no** |
| huggingface_hub whoami | ok, using a cached HF login on the machine (not from .env) |
| meta-llama/Llama-3.1-8B-Instruct config (gated) | readable, so the license is accepted |
| .env ANTHROPIC_API_KEY | yes. claude-sonnet-5-5 test call (max_tokens 16) answered "OK" |

## Disk plan (why the 38 GB isn't a blocker)
The 60 GB target assumes downloading the full bf16 weights (16 GB) for 16-bit LoRA. With 24 GB of VRAM the run uses 4-bit QLoRA anyway (see the deviations in DECISIONS_ft.md). So I load the pre-quantized NF4 checkpoint `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`: the same Meta weights, quantized with bitsandbytes, 5.7 GB.
- Budget: about 5.7 GB model + about 1.5 GB checkpoints (save_total_limit 3) + 1.4 GB roberta-large + 0.5 GB Detoxify, roughly 10 GB of the 29 GB free. No cache cleanup or deletion was needed.

## Versions (for the paper)
- Python 3.12.3, torch 2.14.1+cu130 (CUDA runtime 13.0), triton 3.8.0
- unsloth 2026.11.9, unsloth_zoo 2026.11.9, transformers 5.17.0, trl 1.15.0, peft 0.22.0, bitsandbytes 0.50.2, accelerate 1.16.0, datasets 4.8.5
- evaluate 0.4.6, sacrebleu 2.6.0, rouge-score 0.1.2, bert-score 0.3.13, detoxify 0.5.4
- anthropic 2.12.0, pandas 3.1.1, openpyxl 3.1.5, matplotlib 3.11.2, huggingface_hub 1.33.0
- Attention: PyTorch SDPA. xformers 0.0.36 was pulled in but doesn't load with torch 2.14, so Unsloth falls back to SDPA automatically.
- Install note: pip's config adds the unreachable NGC index, so I installed with `pip --isolated --index-url https://pypi.org/simple`. torch came from PyPI with bundled CUDA 13 libraries, so the PyTorch CUDA index wasn't needed.
