"""Select the PDFs and pages that go to QA generation.

PDF filter: pages < 500, no extract_error, not likely_scanned, not is_duplicate.
Page filter: drop pages with < MIN_WORDS words, and pages that look like
references/bibliography, table of contents/index, or front matter.

Output: data/inventory/selected.csv  (one row per eligible page)
        data/inventory/page_classification.csv  (every page of kept PDFs, with skip reason)
Usage:  python src/select_pages.py
"""
import re
from pathlib import Path

import pandas as pd
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / "data" / "inventory"
TEXT_DIR = ROOT / "data" / "text"

MAX_PAGES = 500
MIN_WORDS = 80

PAGE_SPLIT = re.compile(r"^=== Page (\d+) ===\n", re.M)

# --- reference-list signals (per line) ---
YEAR_PAREN = re.compile(r"\(\s*(19|20)\d{2}[a-z]?\s*[,)]")          # (2019) (2019a) (2019,
YEAR_BARE = re.compile(r"(?<!\d)(19|20)\d{2}[a-z]?[.;,)]")          # 2019. 2019;
STRONG_REF = re.compile(
    r"\bdoi\b|doi\.org|https?://|\bvol\.|\bpp\.|\d+\s*\(\d+\)\s*[,:]\s*\d+|"
    r"retrieved from|\bet al\.|journal of|\bproceedings\b|\bin press\b|isbn",
    re.I,
)
AUTHOR_START = re.compile(r"^[A-Z][A-Za-z'’\-]+,\s+(?:[A-Z]\.\s*){1,3}")  # "Smith, J. A."
REF_HEADING = re.compile(r"^\s*(references|bibliography|works cited|literature cited|reference list)\s*$", re.I)
# Reference-list author initials: "Smith, J. A.," / "Cho, C.H.;" / "J. Smith,". Near zero in body prose.
AUTHOR_INITIALS = re.compile(r"[A-Z][A-Za-z'’\-]+,\s*(?:[A-Z]\.\s*-?\s*){1,3}[,;&(]?|(?:[A-Z]\.\s*){1,2}[A-Z][a-z]+[,:]")
INITIALS_PER_100W = 3.0     # body pages: p95 = 0.4; reference pages: median = 7.7
MIXED_BODY_WORDS = 120      # page with this much prose before a mid-page "References" heading is kept

# --- TOC / index signals ---
TOC_HEADING = re.compile(r"^\s*(table of contents|contents|list of (tables|figures)|index)\s*$", re.I)
DOT_LEADER = re.compile(r"(\.\s*){4,}\s*[ivxlc\d]+\s*$", re.I)
ENDS_WITH_PAGENUM = re.compile(r"\s(\d{1,4}|[ivxlc]{1,6})\s*$", re.I)
INDEX_LINE = re.compile(r",\s*\d{1,4}(\s*[-–]\s*\d{1,4})?(\s*,\s*\d{1,4}(\s*[-–]\s*\d{1,4})?)*\s*$")

# --- front matter signals (per page) ---
FRONT_PATTERNS = [
    r"all rights reserved", r"\bisbn\b", r"library of congress", r"copyright\s*(©|\(c\)|\d{4})",
    r"©", r"printed in", r"published by", r"cataloging[- ]in[- ]publication", r"\bproquest\b",
    r"permission of the publisher", r"in partial fulfillment", r"approval page", r"dissertation committee",
    r"acknowledg(e)?ments?", r"\bdedicat(ed|ion)\b", r"british library", r"e-?isbn", r"\bdoi:\s*10\.\d+/[^\s]+\s*$",
    r"this work is licensed under", r"creative commons", r"\bprinted on acid-free\b",
]
FRONT_RE = [re.compile(p, re.I | re.M) for p in FRONT_PATTERNS]


def split_pages(text: str) -> list[tuple[int, str]]:
    parts = PAGE_SPLIT.split(text)
    # parts = ["", "1", body1, "2", body2, ...]
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def classify(body: str) -> tuple[str, dict]:
    """Return (reason, features). reason == "" means keep."""
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    words = len(body.split())
    n = max(len(lines), 1)

    year = sum(1 for l in lines if YEAR_PAREN.search(l) or YEAR_BARE.search(l))
    strong = sum(1 for l in lines if STRONG_REF.search(l))
    author = sum(1 for l in lines if AUTHOR_START.match(l))
    toc = sum(1 for l in lines if DOT_LEADER.search(l))
    endnum = sum(1 for l in lines if ENDS_WITH_PAGENUM.search(l) and len(l.split()) <= 12)
    idx = sum(1 for l in lines if INDEX_LINE.search(l))
    front = sum(1 for r in FRONT_RE if r.search(body))
    head = lines[:6]
    ref_head = any(REF_HEADING.match(l) for l in head)
    toc_head = any(TOC_HEADING.match(l) for l in head)

    f = dict(words=words, lines=len(lines), year_share=round(year / n, 3),
             strong_share=round(strong / n, 3), author_share=round(author / n, 3),
             toc_share=round(toc / n, 3), endnum_share=round(endnum / n, 3),
             index_share=round(idx / n, 3), front_hits=front)

    if words < MIN_WORDS:
        return "short", f
    # Index / TOC first (index pages are full of names and would otherwise look like references).
    if f["toc_share"] >= 0.25 or (toc_head and (f["endnum_share"] >= 0.3 or f["toc_share"] >= 0.1)):
        return "toc_index", f
    if f["endnum_share"] >= 0.55 or f["index_share"] >= 0.4:
        return "toc_index", f
    # Mixed page: real prose followed by the start of the reference list -> keep (the model skips citations).
    for i, l in enumerate(lines):
        if REF_HEADING.match(l):
            if len(" ".join(lines[:i]).split()) >= MIXED_BODY_WORDS:
                return "", f
            break
    init_d = len(AUTHOR_INITIALS.findall(body)) / words * 100
    f["initials_per_100w"] = round(init_d, 2)
    ref_score = f["year_share"] + f["strong_share"] + f["author_share"]
    if init_d >= INITIALS_PER_100W:
        return "references", f
    if ref_head and ref_score >= 0.3:
        return "references", f
    if ref_score >= 0.75 or (f["year_share"] >= 0.35 and f["strong_share"] >= 0.2) or f["author_share"] >= 0.25:
        return "references", f
    # Front matter: copyright/ISBN/approval pages. Require 2+ signals, or 1 on a light page.
    if front >= 3 or (front >= 2 and words < 250):
        return "front_matter", f
    return "", f


def main() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    inv = pd.read_csv(INV / "pdf_inventory.csv", keep_default_na=False)
    inv["likely_scanned"] = inv["likely_scanned"].astype(str).str.lower().eq("true")
    inv["is_duplicate"] = inv["is_duplicate"].astype(str).str.lower().eq("true")
    inv["pages"] = pd.to_numeric(inv["pages"], errors="coerce")

    keep = inv[(inv["pages"] < MAX_PAGES) & (inv["extract_error"] == "")
               & ~inv["likely_scanned"] & ~inv["is_duplicate"]]
    print(f"PDFs in inventory: {len(inv)}; kept after PDF filter: {len(keep)}")
    print(f"  dropped: >=500 pages {int((inv['pages'] >= MAX_PAGES).sum())}, "
          f"errors {int((inv['extract_error'] != '').sum())}, scanned {int(inv['likely_scanned'].sum())}, "
          f"duplicates {int(inv['is_duplicate'].sum())} (overlapping)")

    rows = []
    for r in keep.itertuples():
        txt = TEXT_DIR / Path(r.rel_path).with_suffix(".txt")
        for page, body in split_pages(txt.read_text(encoding="utf-8")):
            reason, f = classify(body)
            rows.append(dict(file_id=r.file_id, rel_path=r.rel_path, filename=r.filename,
                             pdf_pages=int(r.pages), page=page, skip_reason=reason,
                             tokens=len(enc.encode(body, disallowed_special=())) if not reason else None, **f))
    pages = pd.DataFrame(rows)
    pages.to_csv(INV / "page_classification.csv", index=False)

    counts = pages["skip_reason"].replace("", "eligible").value_counts()
    print("\nPage classification:")
    print(counts.to_string())

    sel = pages[pages["skip_reason"] == ""][
        ["file_id", "rel_path", "filename", "pdf_pages", "page", "words", "tokens"]]
    sel.to_csv(INV / "selected.csv", index=False)
    print(f"\nSelected: {sel['file_id'].nunique()} PDFs, {len(sel)} eligible pages, "
          f"{int(sel['tokens'].sum()):,} tokens (cl100k_base)")


if __name__ == "__main__":
    main()
