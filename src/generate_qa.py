"""Generate grounded QA pairs from eligible pages with Claude Haiku 5.5.

One API call per page -> exactly 4 pairs (3 "paragraph" + 1 "summary"), strict JSON via
structured outputs. System prompt + seed examples sit in a cached prefix.

Inputs:  data/inventory/selected.csv, data/text/**.txt, configs/system_prompt.txt,
         configs/seed_examples.json, .env (ANTHROPIC_API_KEY, loaded with python-dotenv)
Outputs: data/qa/qa_pairs.jsonl     one line per QA pair
         data/qa/pages_done.jsonl   one line per processed page (status, tokens, cost) - used to resume
         logs/generate.log

Usage:
  python src/generate_qa.py                                   # full run
  python src/generate_qa.py --files P0012,P0044 --out-dir data/qa_pilot   # pilot
  python src/generate_qa.py --order fewest_pages --workers 16 --cap 90
"""
import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "data" / "text"
SELECTED = ROOT / "data" / "inventory" / "selected.csv"
CONFIGS = ROOT / "configs"
LOG_FILE = ROOT / "logs" / "generate.log"

MODEL = "claude-haiku-5-5"
# USD per 1M tokens, Claude Haiku 5.5, prompts <= 100K tokens (claude-api skill model table, cached
# 2026-11-03). Cache read = 0.1x input, 5-minute cache write = 1.25x input.
PRICE = {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125}
MAX_TOKENS = 3000
N_PAIRS, N_PARAGRAPH = 4, 3
ANSWER_WORDS = (40, 200)
MAX_ATTEMPTS = 3          # first try + 2 retries on invalid output
MAX_API_TRIES = 8         # transient API errors (429/5xx/529/connection)
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
ET = ZoneInfo("America/New_York")

SOURCE_REF = re.compile(
    r"\b(this|the|that|these|above)\s+(study|studies|paper|article|chapter|page|text|passage|document|"
    r"report|thesis|dissertation|book|section|excerpt|source|author|authors|researchers?|survey|review)\b"
    r"(?!\s+(of|on|in|about|by|that|which|with|from|among|conducted|involving|examining)\b)",
    re.I,
)
REFUSAL = re.compile(
    r"i (don't|do not|cannot|can't) (have|provide|find)|not (provided|mentioned|stated|specified|discussed) "
    r"(on|in) the (page|text|passage)|the (page|text|passage|excerpt) (does not|doesn't)|as an ai|"
    r"no information (is|was) (given|provided)|based on the (provided )?(page|text|passage)",
    re.I,
)
LEAK = re.compile(r"\bstand alone\b|exactly 4|question-answer pairs|\"paragraph\"|\"summary\"|schema|"
                  r"these instructions|the user message", re.I)

SCHEMA = {
    "type": "object",
    "properties": {
        "skip": {"type": "boolean"},
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["paragraph", "summary"]},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["type", "question", "answer"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["skip", "pairs"],
    "additionalProperties": False,
}

log = logging.getLogger("generate")


# ---------------------------------------------------------------- setup
def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.setLevel(logging.INFO)


def load_seeds(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["examples"] if isinstance(data, dict) else data
    out = []
    for e in items:
        if not e.get("question") or not e.get("answer"):
            raise ValueError(f"seed example missing question/answer: {e}")
        out.append({"type": e.get("type", "paragraph"), "question": e["question"].strip(),
                    "answer": e["answer"].strip()})
    return out


def build_system(seeds: list[dict]) -> list[dict]:
    prompt = (CONFIGS / "system_prompt.txt").read_text(encoding="utf-8").rstrip()
    ex = "\n\n".join(f"Example {i} (type: {s['type']})\nQ: {s['question']}\nA: {s['answer']}"
                     for i, s in enumerate(seeds, 1))
    # Single static block, cached. Nothing per-request goes here.
    return [{"type": "text", "text": f"{prompt}\n\n{ex}", "cache_control": {"type": "ephemeral"}}]


def page_text(rel_path: str, page: int, cache: dict) -> str:
    if rel_path not in cache:
        raw = (TEXT_DIR / Path(rel_path).with_suffix(".txt")).read_text(encoding="utf-8")
        parts = re.split(r"^=== Page (\d+) ===\n", raw, flags=re.M)
        cache.clear()  # keep memory small; pages of one file are processed close together
        cache[rel_path] = {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    return cache[rel_path][page].strip()


def user_message(filename: str, page: int, text: str) -> str:
    return (f"Source file name (context only - never cite or mention it): {filename}\n"
            f"Page number: {page}\n\n<page>\n{text}\n</page>\n\n"
            "Write the 4 question-answer pairs for this page as JSON, or return skip if it has no substantive content.")


# ---------------------------------------------------------------- validation
def validate(obj: dict) -> tuple[str, list[dict]]:
    """Return (error, pairs). error == '' means valid; 'SKIP' means model chose skip."""
    if obj.get("skip") is True:
        return "SKIP", []
    pairs = obj.get("pairs") or []
    if len(pairs) != N_PAIRS:
        return f"expected {N_PAIRS} pairs, got {len(pairs)}", []
    types = [p.get("type") for p in pairs]
    if types.count("paragraph") != N_PARAGRAPH or types.count("summary") != 1:
        return f"need {N_PARAGRAPH} paragraph + 1 summary, got {types}", []
    errs, seen = [], set()
    for i, p in enumerate(pairs, 1):
        q, a = (p.get("question") or "").strip(), (p.get("answer") or "").strip()
        if not q or not a:
            errs.append(f"pair {i}: empty question or answer")
            continue
        if not q.endswith("?"):
            errs.append(f"pair {i}: question must end with '?'")
        nw = len(a.split())
        if not ANSWER_WORDS[0] <= nw <= ANSWER_WORDS[1]:
            errs.append(f"pair {i}: answer has {nw} words (allowed 40-200, target 60-150)")
        if SOURCE_REF.search(q):
            errs.append(f"pair {i}: question refers to the source ('{SOURCE_REF.search(q).group(0)}'); name the concept instead")
        if REFUSAL.search(a) or REFUSAL.search(q):
            errs.append(f"pair {i}: refusal/hedging about missing information")
        if LEAK.search(a) or LEAK.search(q):
            errs.append(f"pair {i}: text repeats the instructions")
        key = q.lower()
        if key in seen:
            errs.append(f"pair {i}: duplicate question")
        seen.add(key)
        p["question"], p["answer"] = q, a
    pairs = sorted(pairs, key=lambda p: p["type"] != "paragraph")  # paragraphs first
    return ("; ".join(errs), pairs) if errs else ("", pairs)


def cost_of(u) -> float:
    return (u.input_tokens * PRICE["input"] + (u.cache_creation_input_tokens or 0) * PRICE["cache_write"]
            + (u.cache_read_input_tokens or 0) * PRICE["cache_read"] + u.output_tokens * PRICE["output"]) / 1e6


# ---------------------------------------------------------------- runner
class Runner:
    def __init__(self, args, todo: pd.DataFrame, spent0: float, out_dir: Path):
        self.args, self.todo, self.out_dir = args, todo, out_dir
        self.spent = spent0
        self.reserved = 0.0
        self.stop = asyncio.Event()
        self.stop_reason = ""
        self.done = self.ok = self.skipped = self.invalid = self.errors = 0
        self.t0 = time.monotonic()
        self.est_call_cost = 0.0015  # updated from observed mean
        # Keys not scoped to a workspace need the workspace header; set ANTHROPIC_WORKSPACE_ID in .env.
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        self.client = anthropic.AsyncAnthropic(max_retries=0, timeout=120.0,
                                               default_headers={"anthropic-workspace-id": ws} if ws else None)
        self.system = build_system(load_seeds(Path(args.seeds)))
        self.qa_f = open(out_dir / "qa_pairs.jsonl", "a", encoding="utf-8")
        self.pages_f = open(out_dir / "pages_done.jsonl", "a", encoding="utf-8")
        self.deadline = None
        if args.deadline:
            hh, mm = map(int, args.deadline.split(":"))
            self.deadline = datetime.now(ET).replace(hour=hh, minute=mm, second=0, microsecond=0)

    async def call(self, messages):
        for attempt in range(MAX_API_TRIES):
            try:
                return await self.client.messages.create(
                    model=MODEL, max_tokens=MAX_TOKENS, system=self.system, messages=messages,
                    thinking={"type": "disabled"},
                    output_config={"effort": self.args.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
                )
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                err = e
            except anthropic.APIStatusError as e:
                if e.status_code not in RETRY_STATUS:
                    raise
                err = e
            delay = min(60.0, 2.0 * 2 ** attempt) * random.uniform(0.5, 1.0)
            log.warning("transient API error (%s); retry %d in %.1fs", getattr(err, "status_code", type(err).__name__),
                        attempt + 1, delay)
            await asyncio.sleep(delay)
        raise err

    async def process(self, row) -> None:
        text = page_text(row.rel_path, int(row.page), self.cache)
        messages = [{"role": "user", "content": user_message(row.filename, int(row.page), text)}]
        tok = dict(input_tokens=0, cache_write_tokens=0, cache_read_tokens=0, output_tokens=0)
        cost, status, err, pairs, attempts = 0.0, "invalid", "", [], 0
        try:
            for attempts in range(1, MAX_ATTEMPTS + 1):
                resp = await self.call(messages)
                u = resp.usage
                tok["input_tokens"] += u.input_tokens
                tok["cache_write_tokens"] += u.cache_creation_input_tokens or 0
                tok["cache_read_tokens"] += u.cache_read_input_tokens or 0
                tok["output_tokens"] += u.output_tokens
                cost += cost_of(u)
                txt = next((b.text for b in resp.content if b.type == "text"), "")
                if resp.stop_reason == "refusal":
                    err = "model refusal"
                    break
                try:
                    obj = json.loads(txt)
                    err, pairs = validate(obj)
                except json.JSONDecodeError as e:
                    err = f"invalid JSON ({resp.stop_reason}): {e}"
                if err == "SKIP":
                    status, err = "skip", ""
                    break
                if not err:
                    status = "ok"
                    break
                # Retry with targeted feedback (system prefix stays cached).
                messages = messages[:1] + [
                    {"role": "assistant", "content": txt or "{}"},
                    {"role": "user", "content": f"That output failed validation: {err}. Return the corrected JSON "
                                                f"with exactly 3 paragraph pairs and 1 summary pair."},
                ]
        except Exception as e:  # non-retryable API error or exhausted transient retries
            status, err = "error", f"{type(e).__name__}: {e}"[:500]
        self.record(row, status, err, pairs, tok, cost, attempts)

    def record(self, row, status, err, pairs, tok, cost, attempts) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.spent += cost
        if status == "ok":
            for i, p in enumerate(pairs, 1):
                rec = dict(qa_id=f"{row.file_id}_p{int(row.page):04d}_{i}", file_id=row.file_id,
                           filename=row.filename, page=int(row.page), type=p["type"], question=p["question"],
                           answer=p["answer"], model=MODEL, **{k: tok[k] for k in ("input_tokens", "cache_read_tokens", "output_tokens")},
                           cache_write_tokens=tok["cache_write_tokens"], cost_usd=round(cost / len(pairs), 8),
                           created_at=now)
                self.qa_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.qa_f.flush()
            self.ok += 1
        elif status == "skip":
            self.skipped += 1
        elif status == "invalid":
            self.invalid += 1
            log.warning("invalid after %d attempts: %s p%s: %s", attempts, row.file_id, row.page, err)
        else:
            self.errors += 1
            log.error("API error: %s p%s: %s", row.file_id, row.page, err)
        self.pages_f.write(json.dumps(dict(file_id=row.file_id, page=int(row.page), status=status, error=err,
                                           attempts=attempts, cost_usd=round(cost, 8), **tok, created_at=now)) + "\n")
        self.pages_f.flush()
        self.done += 1
        if self.done % 100 == 0 or self.done == len(self.todo):
            self.progress()

    def progress(self) -> None:
        mins = (time.monotonic() - self.t0) / 60
        rate = self.done / mins if mins else 0
        left = len(self.todo) - self.done
        eta = datetime.now(ET) + pd.Timedelta(minutes=left / rate) if rate else None
        log.info("progress %d/%d pages (ok %d, skip %d, invalid %d, error %d) | %.1f pages/min | spent $%.4f of $%.2f | ETA %s",
                 self.done, len(self.todo), self.ok, self.skipped, self.invalid, self.errors, rate,
                 self.spent, self.args.cap, eta.strftime("%H:%M ET") if eta else "?")

    async def worker(self, queue: asyncio.Queue) -> None:
        while not self.stop.is_set():
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if self.deadline and datetime.now(ET) >= self.deadline:
                self.halt(f"deadline {self.args.deadline} ET reached")
                return
            # Budget guard: count in-flight calls at a pessimistic per-call estimate.
            reserve = max(self.est_call_cost * 3, 0.005)
            if self.spent + self.reserved + reserve > self.args.cap:
                self.halt(f"spend cap ${self.args.cap:.2f} reached (spent ${self.spent:.4f})")
                return
            self.reserved += reserve
            try:
                await self.process(row)
            finally:
                self.reserved -= reserve
            if self.done:
                self.est_call_cost = max(self.spent_this_run() / self.done, 0.0005)

    def spent_this_run(self) -> float:
        return self.spent - self.spent0

    def halt(self, reason: str) -> None:
        if not self.stop.is_set():
            self.stop_reason = reason
            log.warning("stopping: %s", reason)
            self.stop.set()

    async def run(self) -> None:
        self.spent0 = self.spent
        self.cache: dict = {}
        queue: asyncio.Queue = asyncio.Queue()
        for row in self.todo.itertuples():
            queue.put_nowait(row)
        log.info("start: %d pages to do, %d workers, model %s, effort %s, cap $%.2f (already spent $%.4f), prices/MTok %s",
                 len(self.todo), self.args.workers, MODEL, self.args.effort, self.args.cap, self.spent, PRICE)
        await asyncio.gather(*(self.worker(queue) for _ in range(self.args.workers)))
        self.progress()
        log.info("finished%s: spent this run $%.4f, total $%.4f", f" ({self.stop_reason})" if self.stop_reason else "",
                 self.spent_this_run(), self.spent)
        self.qa_f.close()
        self.pages_f.close()
        await self.client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "qa"))
    ap.add_argument("--files", help="comma-separated file_ids to process (pilot)")
    ap.add_argument("--max-pages-per-file", type=int, default=0, help="limit pages per file (pilot); 0 = all")
    ap.add_argument("--order", choices=["file", "fewest_pages"], default="file",
                    help="file = file_id/page order; fewest_pages = shortest PDFs first (so a cap cuts only the tail)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=float, default=90.0, help="hard USD cap, including spend already in pages_done.jsonl")
    ap.add_argument("--extra-spent", type=float, default=0.0, help="spend outside out-dir (e.g. pilots) to count toward cap")
    ap.add_argument("--deadline", default="", help="HH:MM ET; stop starting new pages after this time")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--seeds", default=str(CONFIGS / "seed_examples.json"))
    ap.add_argument("--retry-errors", action="store_true", default=True,
                    help="re-run pages whose last status was 'error' (API failure); invalid/skip are final")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")  # key is read by the SDK from the environment; never printed
    setup_logging()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(SELECTED)
    if args.files:
        sel = sel[sel["file_id"].isin(args.files.split(","))]
    if args.max_pages_per_file:
        sel = sel.groupby("file_id", group_keys=False).head(args.max_pages_per_file)
    sel = (sel.sort_values(["pdf_pages", "file_id", "page"]) if args.order == "fewest_pages"
           else sel.sort_values(["file_id", "page"]))

    # Resume: skip pages already final; count their spend toward the cap.
    done, spent = set(), args.extra_spent
    pages_path = out_dir / "pages_done.jsonl"
    if pages_path.exists():
        last = {}
        for line in pages_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                spent += r.get("cost_usd", 0.0)
                last[(r["file_id"], r["page"])] = r["status"]
        done = {k for k, s in last.items() if s != "error" or not args.retry_errors}
    todo = sel[[(f, int(p)) not in done for f, p in zip(sel["file_id"], sel["page"])]]
    log.info("selected %d pages; %d already done; %d to do", len(sel), len(sel) - len(todo), len(todo))
    if todo.empty:
        log.info("nothing to do")
        return
    asyncio.run(Runner(args, todo, spent, out_dir).run())


if __name__ == "__main__":
    main()
