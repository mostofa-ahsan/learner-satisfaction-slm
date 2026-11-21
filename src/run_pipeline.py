"""Unattended orchestrator: wait for a working API key -> pilot -> full run -> post-process.

Every step is idempotent/resumable, so this can be re-launched at any time:
  nohup python src/run_pipeline.py > logs/pipeline_stdout.log 2>&1 &

1. Auth check with the free Models API (models.retrieve); re-checks every 10 min until a valid
   key appears in .env (re-read on every check) or the latest useful start time passes.
2. Pilot: 3 diverse PDFs, ~60 pages -> data/qa_pilot/. Writes logs/pilot_sample.md (20 random pairs)
   and pilot metrics. The pilot outputs are NOT reused; pilot spend counts toward the cap.
3. Projection: cost/page and pages/min from the pilot. If projected cost > cap, process PDFs in
   fewest-pages-first order so the cap only truncates the tail. If 8 workers would miss the deadline,
   use 16.
4. Full run -> data/qa/ (resumable; stops at the cap or at --stop-new HH:MM ET).
5. Post-process -> train/test/chat files + qa_summary.xlsx.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ET = ZoneInfo("America/New_York")
LOG = ROOT / "logs" / "pipeline.log"
DECISIONS = ROOT / "logs" / "DECISIONS.md"
PILOT_DIR = ROOT / "data" / "qa_pilot"
QA_DIR = ROOT / "data" / "qa"
PILOT_FILES = "P0073,P0005,P0107"   # journal article / dissertation / practitioner book
PILOT_PAGES_PER_FILE = 24

log = logging.getLogger("pipeline")


def now_et() -> datetime:
    return datetime.now(ET)


def at_et(hhmm: str) -> datetime:
    hh, mm = map(int, hhmm.split(":"))
    return now_et().replace(hour=hh, minute=mm, second=0, microsecond=0)


def decide(text: str) -> None:
    with open(DECISIONS, "a", encoding="utf-8") as f:
        f.write(f"\n- [{now_et():%Y-%m-%d %H:%M} ET, auto] {text}\n")
    log.info("DECISION: %s", text)


def key_works() -> tuple[bool, str]:
    """Free auth check. Re-reads .env each time so a fixed key is picked up."""
    from dotenv import dotenv_values
    import anthropic
    env = dotenv_values(ROOT / ".env")
    key = (env.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        return False, "ANTHROPIC_API_KEY missing from .env"
    try:
        ws = (env.get("ANTHROPIC_WORKSPACE_ID") or "").strip()
        anthropic.Anthropic(api_key=key, max_retries=2, timeout=30,
                            default_headers={"anthropic-workspace-id": ws} if ws else None).models.retrieve("claude-haiku-5-5")
        os.environ["ANTHROPIC_API_KEY"] = key  # children inherit; generate_qa's load_dotenv won't override
        if ws:
            os.environ["ANTHROPIC_WORKSPACE_ID"] = ws
        return True, "ok"
    except anthropic.AuthenticationError as e:
        return False, f"401 authentication_error ({getattr(e, 'request_id', '')})"
    except anthropic.APIStatusError as e:
        msg = e.body.get("error", {}).get("message", "") if isinstance(e.body, dict) else ""
        return False, f"HTTP {e.status_code}: {msg[:160]}"
    except anthropic.APIConnectionError as e:
        return False, f"connection error: {e}"


def run(cmd: list[str]) -> int:
    log.info("run: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def pilot_report() -> dict:
    import random
    import re
    pages = read_jsonl(PILOT_DIR / "pages_done.jsonl")
    qa = read_jsonl(PILOT_DIR / "qa_pairs.jsonl")
    if not pages:
        return {}
    first = datetime.fromisoformat(pages[0]["created_at"])
    last = datetime.fromisoformat(pages[-1]["created_at"])
    minutes = max((last - first).total_seconds() / 60, 0.5)
    st = {s: sum(p["status"] == s for p in pages) for s in ("ok", "skip", "invalid", "error")}
    aw = [len(q["answer"].split()) for q in qa]
    srcref = re.compile(r"\b(this|the) (study|paper|article|chapter|page|text|passage|author)s?\b", re.I)
    m = dict(pages=len(pages), **st, cost=sum(p["cost_usd"] for p in pages),
             cost_per_page=sum(p["cost_usd"] for p in pages) / len(pages),
             pages_per_min_8w=len(pages) / minutes,
             cache_hit_share=sum(p["cache_read_tokens"] for p in pages) /
                             max(1, sum(p["cache_read_tokens"] + p["cache_write_tokens"] + p["input_tokens"] for p in pages)),
             mean_answer_words=sum(aw) / max(1, len(aw)), pairs=len(qa),
             answers_with_source_refs=sum(bool(srcref.search(q["answer"])) for q in qa))
    random.seed(42)
    with open(ROOT / "logs" / "pilot_sample.md", "w", encoding="utf-8") as f:
        f.write(f"# Pilot sample (20 random pairs)\n\nmetrics: {json.dumps(m, indent=1)}\n\n")
        for q in random.sample(qa, min(20, len(qa))):
            f.write(f"## {q['qa_id']} [{q['type']}] {q['filename']} p{q['page']}\n**Q:** {q['question']}\n\n"
                    f"**A:** {q['answer']}\n\n")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=90.0)
    ap.add_argument("--stop-new", default="21:15", help="HH:MM ET: no new pages started after this")
    ap.add_argument("--give-up", default="20:30", help="HH:MM ET: stop waiting for a valid key after this")
    ap.add_argument("--poll-min", type=float, default=10)
    ap.add_argument("--skip-pilot", action="store_true")
    args = ap.parse_args()

    LOG.parent.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler(sys.stdout)])

    # 1. wait for a working key
    while True:
        ok, why = key_works()
        if ok:
            log.info("API key works")
            break
        log.warning("API key not usable: %s", why)
        if now_et() >= at_et(args.give_up):
            decide(f"Gave up waiting for a valid ANTHROPIC_API_KEY at {args.give_up} ET (last check: {why}). "
                   "No generation ran. Fix .env and re-run `nohup python src/run_pipeline.py ...` - everything resumes.")
            return
        time.sleep(args.poll_min * 60)

    # 2. pilot
    pilot_spent = sum(p["cost_usd"] for p in read_jsonl(PILOT_DIR / "pages_done.jsonl"))
    if not args.skip_pilot:
        run([PY, "src/generate_qa.py", "--files", PILOT_FILES, "--max-pages-per-file", str(PILOT_PAGES_PER_FILE),
             "--out-dir", str(PILOT_DIR), "--workers", "8", "--cap", "3"])
    m = pilot_report()
    pilot_spent = m.get("cost", pilot_spent)
    log.info("pilot metrics: %s", m)

    # 3. projection
    import pandas as pd
    sel = pd.read_csv(ROOT / "data" / "inventory" / "selected.csv")
    done = {(p["file_id"], p["page"]) for p in read_jsonl(QA_DIR / "pages_done.jsonl") if p["status"] != "error"}
    left = len(sel) - len(done & set(zip(sel.file_id, sel.page)))
    cpp = m.get("cost_per_page") or 0.002
    rate8 = m.get("pages_per_min_8w") or 40
    projected = cpp * len(sel) + pilot_spent
    minutes_avail = (at_et(args.stop_new) - now_et()).total_seconds() / 60
    workers = 8 if left / rate8 <= minutes_avail * 0.85 else 16
    order = "fewest_pages" if projected > args.cap else "file"
    decide(f"Pilot: {m.get('pages')} pages, ${m.get('cost', 0):.4f}, ${cpp:.5f}/page, {rate8:.1f} pages/min at 8 workers, "
           f"invalid {m.get('invalid')}, skip {m.get('skip')}, mean answer {m.get('mean_answer_words', 0):.0f} words, "
           f"cache-read share {m.get('cache_hit_share', 0):.0%}. Projected full cost ${projected:.2f} for {len(sel)} pages "
           f"(cap ${args.cap:.0f}); ETA at 8 workers {left / rate8:.0f} min vs {minutes_avail:.0f} min available. "
           f"Chose workers={workers}, order={order}.")

    # 4. full run (pilot spend counted toward the cap)
    rc = run([PY, "src/generate_qa.py", "--out-dir", str(QA_DIR), "--workers", str(workers), "--order", order,
              "--cap", str(args.cap), "--extra-spent", f"{pilot_spent:.6f}", "--deadline", args.stop_new])
    # one automatic resume for crashes (not for clean cap/deadline stops)
    if rc != 0 and now_et() < at_et(args.stop_new):
        decide(f"generate_qa.py exited with code {rc}; resuming once automatically.")
        run([PY, "src/generate_qa.py", "--out-dir", str(QA_DIR), "--workers", str(workers), "--order", order,
             "--cap", str(args.cap), "--extra-spent", f"{pilot_spent:.6f}", "--deadline", args.stop_new])

    # 5. post-process
    rc = run([PY, "src/postprocess_qa.py", "--qa-dir", str(QA_DIR), "--extra-spent", f"{pilot_spent:.6f}"])
    decide(f"Full run + post-processing finished (postprocess exit {rc}). See data/qa/qa_summary.xlsx and logs/generate.log.")


if __name__ == "__main__":
    main()
