"""Clean the deduplicated QA set before fine-tuning (no API calls).

Rules, applied in order (each counted on what is left after the previous one):
  1. source references in question or answer ("this study", "the authors", "according to the discussion", ...)
  2. citations in the answer ("et al.", "Name (YYYY)", "(Name, YYYY)")
  3. boilerplate questions ("What is the main argument about ..."): reported; if they exceed 15 % of
     summary pairs, a random 50 % per PDF is kept
  4. per-PDF cap (default 400): paragraph:summary ratio kept, sampled round-robin across pages
Then re-split by document with the SAME test file_ids as data/qa/test.jsonl.

Input : data/qa/qa_dedup.jsonl (never modified)
Output: data/qa_clean/{qa_clean,train,test,train_chat,test_chat}.jsonl, clean_summary.xlsx
Usage : python src/clean_qa.py [--in data/qa/qa_dedup.jsonl] [--out-dir data/qa_clean] [--cap 400]
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# ---- rule 1: source references -------------------------------------------------------------
# Literal patterns from the spec are applied with narrowing so ordinary prose survives:
#   "the <noun>" is not a reference when followed by of/on/in/about/... ("the study of online learning",
#   "the passage of legislation"); "this|that <noun>" always is. "a study of ..." never matches.
#   "according to the", "in this", "the present" only count when followed by a source noun
#   ("according to the framework", "in this context", "the present day" stay).
NOUNS = (r"(study|studies|text|page|paper|author|authors|chapter|article|passage|book|report|dissertation|"
         r"thesis|section|document|survey)")
SRC_NOUNS = (r"(study|studies|text|page|paper|authors?|chapter|article|passage|book|report|dissertation|thesis|"
             r"section|document|survey|discussion|research|literature|review|analysis|argument|evidence|"
             r"investigation|work|sample|researchers?|findings|excerpt|source|reviewed|framing|case)")
SOURCE_RULES = {
    "this/that + source noun": re.compile(rf"\b(this|that)\s+{NOUNS}\b", re.I),
    "the + source noun": re.compile(
        rf"\bthe\s+{NOUNS}\b(?!\s+(of|on|in|about|that|which|among|involving|examining|into|with|for)\b)", re.I),
    "according to the + source noun": re.compile(rf"\baccording\s+to\s+the\s+{SRC_NOUNS}\b", re.I),
    "the researchers": re.compile(r"\bthe\s+researchers\b", re.I),
    "in this + source noun": re.compile(rf"\bin\s+this\s+{SRC_NOUNS}\b", re.I),
    "the present + source noun": re.compile(rf"\bthe\s+present\s+{SRC_NOUNS}\b", re.I),
}
LITERAL_RULES = {  # the spec's unrestricted patterns, only for reporting how much the narrowing saves
    "the|this|that + noun (literal)": re.compile(rf"\b(the|this|that)\s+{NOUNS}\b", re.I),
    "according to the (literal)": re.compile(r"\baccording\s+to\s+the\b", re.I),
    "in this (literal)": re.compile(r"\bin\s+this\b", re.I),
    "the present (literal)": re.compile(r"\bthe\s+present\b", re.I),
}

# ---- rule 2: citations ------------------------------------------------------------------------
CITE_RULES = {
    "et al.": re.compile(r"\bet\s+al\b\.?", re.I),
    "Name (YYYY)": re.compile(r"\b[A-Z][A-Za-z'\-]+\s+\((19|20)\d{2}[a-z]?[,;)]"),
    "(Name, YYYY)": re.compile(r"\([A-Z][^()]{0,80}?,\s*(19|20)\d{2}[a-z]?[;)]"),
}

# ---- rule 3: boilerplate questions ------------------------------------------------------------
BOILER = re.compile(
    r"^\s*what\s+(is|are|was)\s+the\s+(main|central|overall|key|primary|core|principal)\s+"
    r"(argument|idea|point|claim|thesis|message|theme|takeaway|conclusion|finding)s?\b", re.I)
BOILER_SHARE_LIMIT = 0.15
BOILER_KEEP_FRAC = 0.5


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows) -> None:
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def first_hit(rules: dict, *texts: str):
    for name, rx in rules.items():
        for t in texts:
            m = rx.search(t)
            if m:
                return name, m
    return None, None


def snippet(text: str, m, width=60) -> str:
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return ("..." if a else "") + text[a:m.start()] + "[[" + m.group(0) + "]]" + text[m.end():b] + ("..." if b < len(text) else "")


def cap_file(rows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    """Sample `cap` pairs, keeping the type ratio and spreading picks round-robin across pages."""
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    n = len(rows)
    quotas = {t: int(cap * len(v) / n) for t, v in by_type.items()}
    # largest-remainder rounding so quotas sum to cap
    rem = sorted(by_type, key=lambda t: cap * len(by_type[t]) / n - quotas[t], reverse=True)
    for t in rem[: cap - sum(quotas.values())]:
        quotas[t] += 1
    keep = []
    for t, items in sorted(by_type.items()):
        pages = defaultdict(list)
        for r in items:
            pages[r["page"]].append(r)
        order = sorted(pages)
        rng.shuffle(order)
        for p in order:
            rng.shuffle(pages[p])
        picked, depth = [], 0
        while len(picked) < quotas[t]:
            for p in order:
                if depth < len(pages[p]):
                    picked.append(pages[p][depth])
                    if len(picked) == quotas[t]:
                        break
            depth += 1
        keep.extend(picked)
    return keep


def xl(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "data" / "qa" / "qa_dedup.jsonl"))
    ap.add_argument("--test-ref", default=str(ROOT / "data" / "qa" / "test.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "qa_clean"))
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows = read_jsonl(Path(args.inp))
    for r in rows:
        r.pop("split", None)
    n0 = len(rows)
    before_file = pd.Series([r["file_id"] for r in rows]).value_counts()
    totals = [("Input pairs (qa_dedup.jsonl)", n0)]

    # 1. source references
    for name, rx in LITERAL_RULES.items():
        totals.append((f"  info: would match {name}",
                       sum(bool(rx.search(r["question"]) or rx.search(r["answer"])) for r in rows)))
    kept, dropped, per_rule = [], [], defaultdict(int)
    for r in rows:
        name, m = first_hit(SOURCE_RULES, r["question"], r["answer"])
        if name:
            per_rule[name] += 1
            dropped.append((name, r, m))
        else:
            kept.append(r)
    print(f"\n[1] source references: dropped {len(dropped)} of {len(rows)}")
    for name in SOURCE_RULES:
        print(f"    {name}: {per_rule[name]}")
        totals.append((f"Rule 1 dropped: {name}", per_rule[name]))
    totals.append(("Rule 1 dropped: source references (total)", len(dropped)))
    print("    10 dropped examples:")
    for name, r, m in rng.sample(dropped, min(10, len(dropped))):
        text = r["question"] if m.string == r["question"] else r["answer"]
        print(f"     - {r['qa_id']} ({name}): {snippet(text, m)}")
    # spot-check: legitimate phrasing must survive
    legit = ["A study of 300 online learners found that satisfaction rose.",
             "Research on the study of online learning has grown.",
             "In this context, learners value prompt feedback.",
             "According to the Community of Inquiry framework, presence has three parts.",
             "Satisfaction matters in the present day.",
             "The passage of the Higher Education Act expanded access."]
    bad = ["This study found that learners were satisfied.", "The authors argue that design matters.",
           "According to the discussion, feedback matters.", "In this chapter, presence is defined.",
           "The researchers surveyed 200 students.", "The present study examined satisfaction."]
    assert not any(first_hit(SOURCE_RULES, s)[0] for s in legit), [s for s in legit if first_hit(SOURCE_RULES, s)[0]]
    assert all(first_hit(SOURCE_RULES, s)[0] for s in bad), [s for s in bad if not first_hit(SOURCE_RULES, s)[0]]
    survivors_with_literal = [r for r in kept if any(rx.search(r["question"]) or rx.search(r["answer"])
                                                     for rx in LITERAL_RULES.values())]
    print(f"    spot-check: {len(legit)} legitimate sentences kept, {len(bad)} source references caught (asserted)")
    print(f"    {len(survivors_with_literal)} kept pairs match only the literal (unrestricted) patterns, e.g.:")
    for r in rng.sample(survivors_with_literal, min(5, len(survivors_with_literal))):
        _, m = first_hit(LITERAL_RULES, r["question"], r["answer"])
        print(f"     + {r['qa_id']}: {snippet(m.string, m, 45)}")
    rows = kept

    # 2. citations (answers)
    kept, per_cite = [], defaultdict(int)
    cite_examples = []
    for r in rows:
        name, m = first_hit(CITE_RULES, r["answer"])
        if name:
            per_cite[name] += 1
            cite_examples.append((r, m))
        else:
            kept.append(r)
    n_cite = len(rows) - len(kept)
    print(f"\n[2] citations: dropped {n_cite} ({dict(per_cite)})")
    for r, m in rng.sample(cite_examples, min(5, len(cite_examples))):
        print(f"     - {r['qa_id']}: {snippet(r['answer'], m, 45)}")
    for name in CITE_RULES:
        totals.append((f"Rule 2 dropped: {name}", per_cite[name]))
    totals.append(("Rule 2 dropped: citations (total)", n_cite))
    rows = kept

    # 3. boilerplate questions
    boiler = [r for r in rows if BOILER.search(r["question"])]
    n_summary = sum(r["type"] == "summary" for r in rows)
    share_summary = len(boiler) / max(1, n_summary)
    share_all = len(boiler) / max(1, len(rows))
    print(f"\n[3] boilerplate questions: {len(boiler)} = {share_all:.1%} of all pairs, "
          f"{share_summary:.1%} of summary pairs ({n_summary})")
    totals += [("Rule 3 boilerplate questions found", len(boiler)),
               ("Rule 3 boilerplate share of all pairs", round(share_all, 3)),
               ("Rule 3 boilerplate share of summary pairs", round(share_summary, 3))]
    n_boiler_drop = 0
    if share_summary > BOILER_SHARE_LIMIT:
        by_file = defaultdict(list)
        for r in boiler:
            by_file[r["file_id"]].append(r)
        drop_ids = set()
        for fid in sorted(by_file):
            items = sorted(by_file[fid], key=lambda r: r["qa_id"])
            rng.shuffle(items)
            n_keep = round(len(items) * BOILER_KEEP_FRAC)
            drop_ids.update(r["qa_id"] for r in items[n_keep:])
        rows = [r for r in rows if r["qa_id"] not in drop_ids]
        n_boiler_drop = len(drop_ids)
        decision = (f"share {share_summary:.1%} > {BOILER_SHARE_LIMIT:.0%} of summary pairs -> kept a random "
                    f"{BOILER_KEEP_FRAC:.0%} per PDF, dropped {n_boiler_drop}")
    else:
        decision = f"share {share_summary:.1%} <= {BOILER_SHARE_LIMIT:.0%} of summary pairs -> all kept"
    print(f"    decision: {decision}")
    totals += [("Rule 3 decision", decision), ("Rule 3 dropped: boilerplate", n_boiler_drop)]

    # 4. per-PDF cap
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["file_id"]].append(r)
    capped, kept = [], []
    for fid in sorted(by_file):
        items = sorted(by_file[fid], key=lambda r: r["qa_id"])
        if len(items) > args.cap:
            sub = cap_file(items, args.cap, rng)
            def ratio(x): return sum(r["type"] == "summary" for r in x) / len(x)
            capped.append((fid, items[0]["filename"], len(items), len(sub), ratio(items), ratio(sub),
                           len({r["page"] for r in items}), len({r["page"] for r in sub})))
            kept.extend(sub)
        else:
            kept.extend(items)
    n_cap = len(rows) - len(kept)
    print(f"\n[4] per-PDF cap {args.cap}: {len(capped)} PDFs capped, dropped {n_cap}")
    for fid, fn, b, a, rb, ra, pb, pa in capped:
        print(f"    {fid} {fn[:60]}: {b} -> {a} pairs, summary share {rb:.2f} -> {ra:.2f}, pages {pb} -> {pa}")
    totals += [("Rule 4 PDFs capped", len(capped)), (f"Rule 4 dropped: cap {args.cap}/PDF", n_cap)]
    rows = sorted(kept, key=lambda r: r["qa_id"])

    # 5. split with the existing test file_ids
    test_ids = {r["file_id"] for r in read_jsonl(Path(args.test_ref))}
    for r in rows:
        r["split"] = "test" if r["file_id"] in test_ids else "train"
    tr = {r["file_id"] for r in rows if r["split"] == "train"}
    te = {r["file_id"] for r in rows if r["split"] == "test"}
    assert not tr & te, "file_id in both train and test"
    missing = test_ids - te
    print(f"\n[5] split: reused {len(test_ids)} test file_ids from {args.test_ref}"
          + (f"; {len(missing)} have no pairs left: {sorted(missing)}" if missing else ""))

    # 6. outputs
    sft_system = (ROOT / "configs" / "sft_system_prompt.txt").read_text(encoding="utf-8").strip()
    write_jsonl(out / "qa_clean.jsonl", rows)
    for split in ("train", "test"):
        part = [r for r in rows if r["split"] == split]
        write_jsonl(out / f"{split}.jsonl", ({k: v for k, v in r.items() if k != "split"} for r in part))
        write_jsonl(out / f"{split}_chat.jsonl", (
            {"messages": [{"role": "system", "content": sft_system},
                          {"role": "user", "content": r["question"]},
                          {"role": "assistant", "content": r["answer"]}]} for r in part))

    df = pd.DataFrame(rows)
    totals += [("Output pairs (qa_clean.jsonl)", len(df)),
               ("  paragraph", int((df["type"] == "paragraph").sum())),
               ("  summary", int((df["type"] == "summary").sum())),
               ("Dropped in total", n0 - len(df)),
               ("Share kept", round(len(df) / n0, 3)),
               ("Train pairs", int((df["split"] == "train").sum())),
               ("Test pairs", int((df["split"] == "test").sum())),
               ("Mean answer length (words)", round(float(df["answer"].str.split().str.len().mean()), 1))]
    totals_df = pd.DataFrame(totals, columns=["metric", "value"])

    after_file = df["file_id"].value_counts()
    cap_set = {c[0] for c in capped}
    names = {r["file_id"]: r["filename"] for r in read_jsonl(Path(args.inp))}
    per_file = pd.DataFrame({"file_id": sorted(before_file.index)})
    per_file["filename"] = per_file["file_id"].map(names)
    per_file["before"] = per_file["file_id"].map(before_file).fillna(0).astype(int)
    per_file["after"] = per_file["file_id"].map(after_file).fillna(0).astype(int)
    per_file["dropped"] = per_file["before"] - per_file["after"]
    per_file["capped"] = per_file["file_id"].map(lambda f: "yes" if f in cap_set else "no")
    per_file["split"] = per_file["file_id"].map(lambda f: "test" if f in test_ids else "train")

    split_tbl = df.groupby("split").agg(pdfs=("file_id", "nunique"), pairs=("qa_id", "size"),
                                        summary=("type", lambda s: int((s == "summary").sum())))
    split_tbl["share_of_pairs"] = (split_tbl["pairs"] / split_tbl["pairs"].sum()).round(3)
    split_tbl = split_tbl.reset_index()
    test_files = per_file[per_file["split"] == "test"][["file_id", "filename", "before", "after"]]
    sample = df.sample(min(50, len(df)), random_state=args.seed)[
        ["qa_id", "split", "type", "filename", "page", "question", "answer"]]

    xlsx = out / "clean_summary.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        xl(totals_df).to_excel(w, sheet_name="Totals", index=False)
        xl(per_file).to_excel(w, sheet_name="PerFile", index=False)
        xl(split_tbl).to_excel(w, sheet_name="Split", index=False)
        xl(test_files).to_excel(w, sheet_name="Split", index=False, startrow=len(split_tbl) + 3)
        xl(sample).to_excel(w, sheet_name="Sample", index=False)
        for name, widths in {"Totals": [55, 70], "PerFile": [9, 70], "Sample": [18, 7, 10, 40, 6, 60, 100]}.items():
            for col, wd in zip("ABCDEFG", widths):
                w.sheets[name].column_dimensions[col].width = wd

    print("\nTotals")
    print(totals_df.to_string(index=False))
    print()
    print(split_tbl.to_string(index=False))
    print(f"\nWrote {out}/ (qa_clean, train, test, train_chat, test_chat .jsonl) and {xlsx}")


if __name__ == "__main__":
    main()
