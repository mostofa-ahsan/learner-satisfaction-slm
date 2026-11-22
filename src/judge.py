"""LLM-as-judge (Table 6.1 rubric) with Claude Sonnet 5.5 via the Message Batches API.

- One request per (model, item); the judge sees only question, reference answer and candidate answer
  (blind to which model produced it; custom_ids are opaque and items of both models are shuffled together).
- Rubric is a fixed, cached system block. thinking adaptive, effort low, structured JSON output.
  No temperature/top_p.
- State in outputs/judge/state.json (batch ids), results in outputs/judge/results.jsonl keyed by custom_id:
  re-running resumes (polls an open batch, never re-submits finished items).
- Errored/expired/unparseable items are retried once (second batch).
- Cost from usage at batch prices (50 % of $2 / $10 per MTok); hard cap enforced with a worst-case
  reservation before every submit. If batches are not done within --batch-hours, remaining items go
  through regular calls (8 workers, full price).

Usage: python src/judge.py [--cap 15] [--batch-hours 3] [--smoke 2]
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
load_dotenv(ROOT / ".env")
ET = ZoneInfo("America/New_York")

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 1536
# USD per 1M tokens. Standard Sonnet 5.5 = $2 in / $10 out; batch = 50 %.
PRICE = {"batch": {"in": 1.0, "out": 5.0}, "std": {"in": 2.0, "out": 10.0}}
CACHE_WRITE_X, CACHE_READ_X = 1.25, 0.10
DIMS = ["relevance", "correctness", "completeness", "conciseness", "overall"]

RUBRIC = """You are an impartial expert evaluator of answers about online learner satisfaction in higher education.
You will receive a question, a reference answer written from the source material, and a candidate answer.
Rate the candidate answer on five criteria, each an integer from 1 to 5, using this rubric (Table 6.1):

- Relevance: how well the candidate answer addresses the question asked.
  1 = off-topic; 5 = perfectly relevant.
- Correctness: factual accuracy of the candidate answer judged against the reference answer.
  1 = wrong; 5 = perfectly correct.
- Completeness: how fully the candidate answer covers the key points of the reference answer.
  1 = misses all key points; 5 = fully complete.
- Conciseness: whether the candidate answer is focused and free of unnecessary or repetitive content.
  1 = very verbose or padded; 5 = perfectly concise.
- Overall: overall quality of the candidate answer as a response to the question.
  1 = poor; 5 = excellent.

Use the full scale; intermediate scores (2-4) mean the answer lies between the anchors.
Judge only the candidate answer. The reference answer is the standard for correctness and completeness;
information in the candidate that is plausible but absent from the reference should not raise correctness.
Do not reward length for its own sake. Give a short rationale (one or two sentences).
Return only the JSON object required by the output schema."""

SCHEMA = {
    "type": "object",
    "properties": {**{d: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for d in DIMS},
                   "rationale": {"type": "string"}},
    "required": DIMS + ["rationale"],
    "additionalProperties": False,
}

OUT = Path("outputs/judge")
STATE = OUT / "state.json"
RESULTS = OUT / "results.jsonl"
LOG = Path("logs/judge.log")


def log(msg):
    line = f"{datetime.now(ET):%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def client():
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    return anthropic.Anthropic(max_retries=4, timeout=120,
                               default_headers={"anthropic-workspace-id": ws} if ws else None)


def params(item):
    user = (f"QUESTION:\n{item['question']}\n\nREFERENCE ANSWER:\n{item['reference']}\n\n"
            f"CANDIDATE ANSWER:\n{item['prediction'] or '(empty answer)'}")
    return {
        "model": MODEL, "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
    }


def cost(usage, tier):
    p = PRICE[tier]
    u = usage if isinstance(usage, dict) else usage.model_dump()
    return ((u.get("input_tokens") or 0) * p["in"] + (u.get("cache_creation_input_tokens") or 0) * p["in"] * CACHE_WRITE_X
            + (u.get("cache_read_input_tokens") or 0) * p["in"] * CACHE_READ_X
            + (u.get("output_tokens") or 0) * p["out"]) / 1e6


def worst_case(items, tier):
    # ~4 chars/token for the input, plus full max_tokens of output
    p = PRICE[tier]
    tin = sum((len(RUBRIC) + len(i["question"]) + len(i["reference"]) + len(i["prediction"] or "")) / 3.5 + 50
              for i in items)
    return (tin * p["in"] * CACHE_WRITE_X + len(items) * MAX_TOKENS * p["out"]) / 1e6


def parse_message(msg):
    if msg.stop_reason != "end_turn":
        return None, f"stop_reason={msg.stop_reason}"
    txt = "".join(b.text for b in msg.content if b.type == "text")
    try:
        d = json.loads(txt)
        assert all(d.get(k) in (1, 2, 3, 4, 5) for k in DIMS)
        return d, None
    except Exception as e:  # noqa: BLE001
        return None, f"unparseable: {type(e).__name__}"


def load_items():
    items = {}
    for which in ("ft", "base"):
        for r in read_jsonl(f"outputs/preds_{which}.jsonl"):
            cid = "j" + hashlib.sha1(f"{which}:{r['idx']}".encode()).hexdigest()[:16]  # opaque id
            items[cid] = dict(custom_id=cid, model=which, idx=r["idx"], qa_id=r["qa_id"], question=r["question"],
                              reference=r["reference"], prediction=r["prediction"])
    return items


def save_state(st):
    # Results must hit the disk before the state can say "collected"; a power cut once
    # left collected=true next to an empty results.jsonl.
    if RESULTS.exists():
        with open(RESULTS, "rb") as f:
            os.fsync(f.fileno())
    tmp = STATE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(st, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE)


def record(cid, item, scores, err, usage, tier, attempt):
    c = cost(usage, tier) if usage is not None else 0.0
    u = (usage if isinstance(usage, dict) else usage.model_dump()) if usage is not None else {}
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(custom_id=cid, model=item["model"], idx=item["idx"], qa_id=item["qa_id"],
                                scores=scores, error=err, tier=tier, attempt=attempt, cost_usd=c,
                                usage={k: u.get(k) for k in ("input_tokens", "output_tokens",
                                                             "cache_creation_input_tokens", "cache_read_input_tokens")}),
                           ensure_ascii=False) + "\n")
    return c


def status():
    res = read_jsonl(RESULTS)
    spent = sum(r["cost_usd"] for r in res)
    ok = {r["custom_id"] for r in res if r["scores"]}
    attempts = {}
    for r in res:
        attempts[r["custom_id"]] = max(attempts.get(r["custom_id"], 0), r["attempt"])
    return spent, ok, attempts


def run_batch(cl, items, todo, attempt, st, args):
    spent, _, _ = status()
    wc = worst_case([items[c] for c in todo], "batch")
    if spent + wc > args.cap:
        log(f"batch attempt {attempt}: worst case ${wc:.2f} + spent ${spent:.2f} > cap ${args.cap}; not submitting")
        return False
    key = f"batch_{attempt}"
    if key not in st:
        reqs = [{"custom_id": c, "params": params(items[c])} for c in todo]
        b = cl.messages.batches.create(requests=reqs)
        st[key] = {"id": b.id, "submitted": time.time(), "n": len(reqs)}
        save_state(st)
        log(f"submitted {key} {b.id} with {len(reqs)} requests (worst case ${wc:.2f})")
    bid, t_sub = st[key]["id"], st[key]["submitted"]
    while True:
        b = cl.messages.batches.retrieve(bid)
        rc = b.request_counts
        if b.processing_status == "ended":
            break
        if time.time() - st["start"] > args.batch_hours * 3600:
            log(f"{key} {bid} not ended after {args.batch_hours} h ({rc}); canceling and falling back")
            cl.messages.batches.cancel(bid)
            while cl.messages.batches.retrieve(bid).processing_status != "ended":
                time.sleep(20)
            break
        log(f"{key} {bid}: {b.processing_status} {rc.succeeded} ok / {rc.errored} err / {rc.processing} processing")
        time.sleep(60)
    n_ok = n_bad = 0
    _, ok, _ = status()
    for r in cl.messages.batches.results(bid):
        cid = r.custom_id
        if cid in ok or cid not in items:
            continue
        if r.result.type == "succeeded":
            scores, err = parse_message(r.result.message)
            record(cid, items[cid], scores, err, r.result.message.usage, "batch", attempt)
            n_ok += scores is not None
            n_bad += scores is None
        else:
            err = r.result.type
            if r.result.type == "errored":
                err += f": {getattr(r.result.error, 'error', r.result.error)}"[:200]
            record(cid, items[cid], None, err, None, "batch", attempt)
            n_bad += 1
    st[key]["collected"] = True
    save_state(st)
    log(f"{key} collected: {n_ok} scored, {n_bad} failed")
    return True


def run_direct(cl, items, todo, attempt, args):
    spent, _, _ = status()
    wc = worst_case([items[c] for c in todo], "std")
    if spent + wc > args.cap:
        # trim to what fits under the cap
        per = wc / max(1, len(todo))
        n = max(0, int((args.cap - spent) / per))
        log(f"direct fallback: worst case ${wc:.2f} would exceed cap; limiting to {n} of {len(todo)} items")
        todo = todo[:n]
    log(f"direct calls (attempt {attempt}) for {len(todo)} items, 8 workers")

    def one(cid):
        try:
            m = cl.messages.create(**params(items[cid]))
            s, e = parse_message(m)
            return cid, s, e, m.usage
        except anthropic.APIError as e:
            return cid, None, f"{type(e).__name__}: {str(e)[:150]}", None

    with cf.ThreadPoolExecutor(8) as ex:
        for cid, s, e, u in ex.map(one, todo):
            record(cid, items[cid], s, e, u, "std", attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=15.0)
    ap.add_argument("--batch-hours", type=float, default=3.0)
    ap.add_argument("--smoke", type=int, default=0, help="N direct calls to test the request shape, then exit")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cl = client()
    items = load_items()

    if args.smoke:
        for cid in list(items)[:args.smoke]:
            m = cl.messages.create(**params(items[cid]))
            s, e = parse_message(m)
            log(f"smoke {cid}: scores={s} err={e} usage={m.usage.model_dump()} cost_std=${cost(m.usage, 'std'):.4f}")
        return

    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    st.setdefault("start", time.time())
    save_state(st)
    n_models = {w: sum(i["model"] == w for i in items.values()) for w in ("ft", "base")}
    log(f"judge {MODEL}: {len(items)} items {n_models}, cap ${args.cap}")

    for attempt in (1, 2):  # attempt 2 = one retry of failed items
        spent, ok, attempts = status()
        todo = [c for c in items if c not in ok and attempts.get(c, 0) < attempt]
        if not todo:
            continue
        random.Random(42 + attempt).shuffle(todo)  # interleave base / ft items
        key = f"batch_{attempt}"
        fallback = time.time() - st["start"] > args.batch_hours * 3600
        if st.get(key, {}).get("collected"):
            continue
        if not fallback:
            run_batch(cl, items, todo, attempt, st, args)
            st = json.loads(STATE.read_text())
        # whatever is still missing after a batch timeout goes direct
        spent, ok, attempts = status()
        todo = [c for c in items if c not in ok and attempts.get(c, 0) < attempt]
        if todo and time.time() - st["start"] > args.batch_hours * 3600:
            run_direct(cl, items, todo, attempt, args)

    spent, ok, _ = status()
    log(f"judge finished: {len(ok)}/{len(items)} scored, spent ${spent:.4f} (cap ${args.cap})")


if __name__ == "__main__":
    main()
