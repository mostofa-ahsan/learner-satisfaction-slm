# Decisions log: QA generation run (2026-11-21)

## BLOCKER (10:25 ET): the API key in .env is rejected
- `.env` contains `ANTHROPIC_API_KEY`. It is well-formed (starts with `sk-ant-`, no quotes, whitespace, CR or BOM), and no shell variable overrides it.
- Every call returns **401 authentication_error "invalid x-api-key"**. That includes the free Models API (`models.retrieve`), so it is not a model, parameter or billing problem. The key is revoked, mistyped or from another org.
- **$0.00 spent.** No QA pairs were generated.
- What I did instead: finished every non-API step, tested the generator and post-processor offline with a mocked client, and started `src/run_pipeline.py` under nohup. It re-reads `.env` every 10 minutes. If a valid key appears before 20:30 ET, it runs pilot, full run and post-processing automatically (see "Unattended fallback" below).
- **To fix:** put a valid key in `.env`, then either wait for the next 10-minute check (if it's still running: `pgrep -f run_pipeline.py`) or run:
  `nohup .venv/bin/python src/run_pipeline.py > logs/pipeline_stdout.log 2>&1 &`
  After 20:30 ET use `--give-up 23:59 --stop-new <HH:MM>` to set new times. Every step resumes from where it stopped.

## Pricing used (claude-haiku-5-5)
Source: claude-api skill model table (cached 2026-11-03): "$0.10 / $0.50 per MTok when the prompt is 100K tokens or fewer". Caching multipliers from the skill's prompt-caching reference: cache read 0.1× input, 5-minute cache write 1.25× input.
| token type | USD / 1M |
|---|---|
| input (uncached) | 0.10 |
| cache write (5m) | 0.125 |
| cache read | 0.01 |
| output | 0.50 |
cost = (input·0.10 + cache_write·0.125 + cache_read·0.01 + output·0.50) / 1e6, summed over every attempt, retries included. All pages are far below 100K tokens, so the higher long-context rate never applies.

## Step 1: selection (`src/select_pages.py`)
- PDF filter: pages < 500, no extract_error, not likely_scanned, not is_duplicate. 232 → 159 PDFs (8 dropped at ≥500 pages, 2 scanned, 68 duplicate copies). 2 of the 159 have no eligible pages, so **157 PDFs** remain.
- Page filter results: **7,784 eligible pages (4.70M cl100k tokens)**. Skipped: 1,252 references, 530 short (< 80 words), 75 TOC/index, 62 front matter. Per-page features and reasons are in `data/inventory/page_classification.csv`.
- How references are detected: density of author initials ("Smith, J. A.", "Cho, C.H.;"), at ≥ 3 per 100 words. Measured: body pages have p95 = 0.4, reference pages have median 7.7. Backup rules use the share of lines with years, doi/vol(issue)/pp./"et al." and a "References" heading. Index/TOC detection runs first so index pages aren't labelled as references. A page with ≥ 120 words of prose before a mid-page "References" heading is kept. Some reference-heavy pages in the 2–3 density band still pass (about half that band is real prose); the generator prompt tells the model to return `skip` for citation-only pages.
- Front matter: 3+ signals (copyright, ISBN, "all rights reserved", ProQuest, approval page, Creative Commons…), or 2+ on a page under 250 words.

## Step 2: seeds (`configs/seed_examples.json`)
- 6 placeholder pairs covering learner, instructor, course design, platform, institutional/policy (emergency remote teaching vs planned online learning), and measurement (reliability/validity). Institutional and policy share one pair so there are 6 pairs, not 7.
- Seed answers avoid author/year citations, so the generator doesn't learn to cite.
- To replace them: swap the `examples` list for the 28 expert pairs (keys `question`, `answer`, optional `type` and `category`). A bare JSON list also works. No code change needed.

## Step 3: generator (`src/generate_qa.py`)
- Model `claude-haiku-5-5`, thinking **disabled** (allowed on Haiku 5.5 at effort ≤ high), effort `low`. Grounded extraction doesn't need reasoning, and thinking tokens are billed as output.
- Strict JSON via **structured outputs** (`output_config.format` json_schema: `{skip: bool, pairs: [{type: paragraph|summary, question, answer}]}`). The schema guarantees valid JSON; the counts and lengths are checked in code.
- Caching: one system block (prompt + seeds) with `cache_control: ephemeral`. It's about 1.6K tokens, above Haiku 5.5's 512-token minimum. The filename goes in the user message, marked "context only, never cite".
- Validation: exactly 4 pairs (3 paragraph + 1 summary), non-empty fields, answers 40–200 words, questions end with "?". Rejected: questions that refer to the source ("this study", "the authors", "the passage"…), refusal/hedging phrases, and prompt leakage. Retries (up to 2) send the validation error back as a follow-up turn; the cached prefix is unchanged. After that the page is marked `invalid` and the run moves on.
- Transient errors (408/409/429/5xx/529, connection, timeout): exponential backoff 2·2^n s, capped at 60 s, 50–100 % jitter, up to 8 tries. SDK auto-retry is off so the backoff is mine.
- Resume: `data/qa/pages_done.jsonl` has one row per page (ok/skip/invalid/error, tokens, cost). On restart, ok/skip/invalid pages are skipped and `error` pages are retried. Spend already in that file plus `--extra-spent` (the pilot) counts toward the cap.
- Cap: before each call, spent + in-flight reservations (3× the observed mean call cost) must fit under $90; otherwise the run stops cleanly. `--deadline` stops new pages after a time (ET).
- Output fields per pair: qa_id, file_id, filename, page, type, question, answer, model, input_tokens, cache_read_tokens, output_tokens, cache_write_tokens, cost_usd, created_at. **Token fields are per page call** (shared by the 4 pairs). `cost_usd` is the pair's quarter of the page cost, so summing cost over pairs is correct; summing tokens over pairs is not (use pages_done.jsonl for that).
- Tested offline with a mocked client: cap stop, resume with no duplicate pages, invalid → retry, 529 backoff, skip handling, and cost bookkeeping all behave as intended.

## Step 4/5: pilot & full run (`src/run_pipeline.py`)
- Pilot: 3 different kinds of document, P0073 (journal article, 12 eligible pages), P0005 (dissertation) and P0107 (practitioner book), 24 pages each, about 60 pages. Output goes to `data/qa_pilot/`, 20 random pairs to `logs/pilot_sample.md`. Pilot spend counts toward the cap.
- Projection before the run (from token counts): 7,784 pages × (~1.7K cached prefix at $0.01/M + ~0.8K page tokens at $0.10/M + ~1.0K output at $0.50/M) ≈ $0.0006/page ≈ **$5 total, far under $90**. So the expected order is `file`; `fewest_pages` only applies if the pilot projects over the cap. Throughput depends on rate limits: at 8 workers × ~10 s/call that's about 48 pages/min, ≈ 2.7 h. The orchestrator switches to 16 workers if 8 would miss the deadline.
- Unattended fallback: since no one can inspect the pilot by hand, the orchestrator checks pilot metrics automatically and writes `logs/pilot_sample.md`. Validation rules are enforced in code, so a bad prompt mostly shows up as `invalid` pages rather than bad data.
- Times: no new pages after **21:15 ET**, so generation is done before the 21:30 deadline; it stops waiting for a key at 20:30 ET.

## Step 6: post-processing (`src/postprocess_qa.py`)
- Dedup: NFKC + lowercase + punctuation removed, then exact matches on the normalized question are dropped. Then near-duplicates: rapidfuzz `ratio` > 90 against any earlier question, chunked `cdist`, first occurrence kept. About 5 s for 31K questions.
- Split by document: sorted file_ids shuffled with `random.Random(42)`, ceil(5 %) of PDFs to test, the rest to train. An assert checks that no PDF is in both.
- Chat format: `{"messages": [system, user=question, assistant=answer]}`, system prompt from `configs/sft_system_prompt.txt`.
- `qa_summary.xlsx`: Totals, PerFile, Split (with the list of test PDFs), Sample (50 random pairs, seed 42).

## Automatic entries
(The orchestrator adds a line below each time it makes a decision.)

- [2026-11-21 10:56 ET] New key in .env: it authenticates (no longer 401), but returns 400 "This API key is not scoped to a workspace, so this request must include the anthropic-workspace-id header". Added optional `ANTHROPIC_WORKSPACE_ID` in .env; generate_qa.py and run_pipeline.py send it as the `anthropic-workspace-id` header when set. Watcher restarted (PID 56498). Still $0.00 spent.

- [2026-11-21 11:06 ET, auto] Pilot: 60 pages, $0.0455, $0.00076/page, 49.3 pages/min at 8 workers, invalid 1, skip 6, mean answer 83 words, cache-read share 66%. Projected full cost $5.95 for 7784 pages (cap $90); ETA at 8 workers 158 min vs 609 min available. Chose workers=8, order=file.

## Pilot results and prompt fix (11:05–11:09 ET)
- 11:00 ET: you replaced the key with a Default Workspace key, and it works (test call answered "OK"). The watcher started pilot v1 at 11:05.
- **Pilot v1** (60 pages: P0073, P0005, P0107; $0.0455; $0.00076/page; 49 pages/min at 8 workers; 53 ok / 6 skip / 1 invalid). The orchestrator projected **$5.95** in total, so it chose order=file with 8 workers and started the full run.
- **My manual review of the v1 pairs:** grounding and length were good (mean 83 words). But **104 of 360 answers referred to the source** ("the study", "the page", "the authors"), and 6 questions did too ("this discussion", "this introduction", "the guide"). A fine-tuned model would pick up that habit. **I stopped the full run after 37 pages ($0.0251) and moved its output to `data/qa_discarded_v1/`.**
- **Fix:** the prompt now says to write answers as direct expert knowledge, with examples ("A survey of 1,414 military online learners found…" instead of "The study found…"). The validator now also checks answers for source references, and the question check covers discussion/introduction/part/guide/analysis/work. Pairs that fail get a retry with the error sent back.
- **Pilot v2** (same 60 pages, `data/qa_pilot_v2/`): $0.0409; 54 ok / 6 skip / 0 invalid; 15 of 60 pages needed one retry; **0 answers with source references**; mean answer 75 words. I reviewed 20 random pairs by hand: grounded, standalone, the right length, no prompt leakage. Leftover: a few answers name authors whose arguments the page discusses ("Carnevale, Smith, and Strohl predicted…"). That's grounded and acceptable, so I didn't add more retries for it.
- Cost per page (v2): $0.00068 → projected **≈ $5.4 for 7,784 pages**, far under the $90 cap, so no reduction in scope. Throughput is about 95 pages/min at 8 workers, ≈ 80 min, finishing around 12:30 ET, well before 21:15. **8 workers kept.**
- **Spend before the full run: $0.1115** (v1 $0.0455 + stopped run $0.0251 + v2 $0.0409). It's passed as `--extra-spent` so the $90 cap and the summary totals include it.
- **Full run started at 11:09 ET**: `logs/full_run.sh` (generate_qa, then postprocess_qa), detached with setsid/nohup, output in `logs/generate_stdout.log`. The original watcher (run_pipeline.py) is stopped; I ran the remaining steps by hand after the prompt fix.

## Final summary (12:26 ET)
- **Generation finished 12:25 ET** (started 11:09; ~101 pages/min at 8 workers; no crashes, 0 API errors). generate exit 0, postprocess exit 0.
- Pages: 7,784 processed → 6,894 ok, 744 skipped by the model (citation-only/fragment pages), 146 invalid after 3 attempts (mostly persistent "the study/the text" phrasing; dropped, not patched).
- Pairs: 27,576 raw → 302 exact + 947 near-duplicate (ratio > 0.9) questions dropped → **26,327 kept** (20,502 paragraph, 5,825 summary) from 156 PDFs.
- Split by PDF (seed 42, ceil 5 % of PDFs): **train 148 PDFs / 25,321 pairs; test 8 PDFs / 1,006 pairs** (3.8 % of pairs: the test PDFs happen to be shorter than average; accepted, since splitting by document matters more than the exact pair share).
- Quality check on the kept set: 0 answers and 0 questions match the source-reference regexes. 196 answers (0.7 %) still contain an "et al." or "Name (year)" citation; left in. Filter with a regex before fine-tuning if you want them out.
- Mean answer 79 words, mean question 18 words.
- **Cost: $5.5089 total** ($5.3974 full run + $0.1115 pilots/stopped run), 6 % of the $90 cap, $0.00069/page.
- Outputs in `data/qa/`: qa_pairs.jsonl (raw), pages_done.jsonl, qa_dedup.jsonl, train/test.jsonl, train_chat/test_chat.jsonl, qa_summary.xlsx.
- Seeds: replacing `configs/seed_examples.json` with the 28 expert pairs and re-running into a new `--out-dir` would cost about $5.5 again.

## Cleanup (2026-11-22): `src/clean_qa.py` → `data/qa_clean/`
No API calls. The input `data/qa/qa_dedup.jsonl` is unchanged. Seed 42. Rules run in order, and each counts on what the previous rules left.
- **Rule 1, source references: dropped 570.** The literal spec patterns were too broad. "in this" alone matched 1,097 pairs, mostly "in this way" or "in this context". "according to the" matched 398, mostly "according to the framework" or "the Community of Inquiry". "the present" hit "the present day". So I narrowed them:
  - "this/that + noun" is always dropped.
  - "the + noun" is dropped unless followed by of/on/in/about/that/which/among/involving/examining/into/with/for ("the study of online learning" and "the passage of legislation" stay).
  - "according to the", "in this" and "the present" are dropped only before a source noun (study, research, literature, discussion, analysis, sample, review, ...).
  - "the researchers" is dropped as is.
  - Built-in asserts: 6 legitimate sentences ("A study of 300 learners...", "In this context...", "According to the Community of Inquiry framework...") are kept, and 6 real references are caught.
  - Breakdown: the+noun 283, in this+noun 131, according to the+noun 95, this/that+noun 55, the present+noun 5, the researchers 1.
- **Rule 2, citations in answers: dropped 191.** "Name (YYYY)" 118, "et al." 71, "(Name, YYYY)" 2.
- **Rule 3, boilerplate questions** ("What is/are the main/central/overall/key… argument/idea/point/theme…"): 5,636 pairs, 22.0% of all pairs and **98.1% of summary pairs**. That is above the 15% limit, so I kept a random 50% per PDF and **dropped 2,826**. Nearly every summary question is boilerplate because the generator prompt prescribed "What is the main argument about <topic>?".
- **Rule 4, cap of 400 pairs per PDF: 19 PDFs capped (mostly books), dropped 7,475.** Within each PDF, the paragraph:summary ratio is kept by largest-remainder quotas, and picks are spread round-robin across shuffled pages. Page coverage is unchanged except P0014 (376 → 350) and P0031/P0015 (−1 each). Per-PDF before/after is in `clean_summary.xlsx` (PerFile sheet).
- **Split:** reuses the 8 test file_ids from `data/qa/test.jsonl`. An assert checks there is no overlap.
- **Result: 26,327 → 15,265 pairs** (58% kept; 13,324 paragraph, 1,941 summary). Train: 148 PDFs, 14,625 pairs. Test: 8 PDFs, 640 pairs (4.2%). Mean answer 78 words.
- Full console output, with dropped and kept examples: `logs/clean_stdout.log`.
