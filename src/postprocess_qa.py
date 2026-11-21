"""Deduplicate generated QA pairs, split by document, and write fine-tuning files + a summary workbook.

Inputs:  <qa-dir>/qa_pairs.jsonl, <qa-dir>/pages_done.jsonl, data/inventory/selected.csv
Outputs: <qa-dir>/qa_dedup.jsonl              all kept pairs (with split column)
         <qa-dir>/train.jsonl, test.jsonl      raw records
         <qa-dir>/train_chat.jsonl, test_chat.jsonl   {"messages": [system, user, assistant]}
         <qa-dir>/qa_summary.xlsx             Totals, PerFile, Split, Sample
Usage:   python src/postprocess_qa.py [--qa-dir data/qa] [--extra-spent 0.12]
"""
import argparse
import json
import math
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TEST_FRAC = 0.05
FUZZ_CUTOFF = 90  # rapidfuzz ratio is 0-100; > 90 == ratio > 0.9
CHUNK = 2000


def read_jsonl(path: Path) -> pd.DataFrame:
    # Plain json parsing: keeps created_at as a string and ints as ints.
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])


def norm(q: str) -> str:
    q = unicodedata.normalize("NFKC", q).lower()
    q = re.sub(r"[^\w\s]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop exact (normalized) duplicates, then near duplicates (ratio > 0.9) vs any earlier question."""
    df = df.copy()
    df["q_norm"] = df["question"].map(norm)
    exact = df.duplicated("q_norm", keep="first")
    df = df[~exact].reset_index(drop=True)
    qs = df["q_norm"].tolist()
    near = np.zeros(len(qs), dtype=bool)
    for start in range(0, len(qs), CHUNK):
        block = qs[start:start + CHUNK]
        # compare the block with everything before its end; keep the first occurrence of each cluster
        m = process.cdist(block, qs[:start + len(block)], scorer=fuzz.ratio, score_cutoff=FUZZ_CUTOFF + 1e-9,
                          dtype=np.uint8, workers=-1)
        for i in range(len(block)):
            gi = start + i
            if m[i, :gi].any():
                near[gi] = True
    out = df[~near].drop(columns="q_norm").reset_index(drop=True)
    return out, int(exact.sum()), int(near.sum())


def split_files(file_ids: list[str]) -> set[str]:
    ids = sorted(file_ids)
    random.Random(SEED).shuffle(ids)
    n_test = max(1, math.ceil(len(ids) * TEST_FRAC))
    return set(ids[:n_test])


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def xl(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-dir", default=str(ROOT / "data" / "qa"))
    ap.add_argument("--extra-spent", type=float, default=0.0, help="pilot/other spend to add to the cost total")
    args = ap.parse_args()
    qa_dir = Path(args.qa_dir)

    qa = read_jsonl(qa_dir / "qa_pairs.jsonl")
    pages = read_jsonl(qa_dir / "pages_done.jsonl")
    pages = pages.drop_duplicates(["file_id", "page"], keep="last")  # an 'error' row may be superseded on resume
    sel = pd.read_csv(ROOT / "data" / "inventory" / "selected.csv")
    all_pages = read_jsonl(qa_dir / "pages_done.jsonl")  # every attempt row, for cost

    n_raw = len(qa)
    kept, n_exact, n_near = dedup(qa)
    test_ids = split_files(kept["file_id"].unique().tolist())
    kept["split"] = np.where(kept["file_id"].isin(test_ids), "test", "train")
    assert not (set(kept.loc[kept.split == "train", "file_id"]) & set(kept.loc[kept.split == "test", "file_id"]))

    sft_system = (ROOT / "configs" / "sft_system_prompt.txt").read_text(encoding="utf-8").strip()
    write_jsonl(qa_dir / "qa_dedup.jsonl", kept.to_dict("records"))
    for split in ("train", "test"):
        part = kept[kept["split"] == split]
        write_jsonl(qa_dir / f"{split}.jsonl", part.drop(columns="split").to_dict("records"))
        write_jsonl(qa_dir / f"{split}_chat.jsonl", (
            {"messages": [{"role": "system", "content": sft_system},
                          {"role": "user", "content": r.question},
                          {"role": "assistant", "content": r.answer}]} for r in part.itertuples()))

    qw = kept["question"].str.split().str.len()
    aw = kept["answer"].str.split().str.len()
    run_cost = float(all_pages["cost_usd"].sum())
    status = pages["status"].value_counts()
    totals = pd.DataFrame([
        ("QA pairs generated (raw)", n_raw),
        ("Exact duplicate questions dropped", n_exact),
        ("Near-duplicate questions dropped (ratio > 0.9)", n_near),
        ("QA pairs kept", len(kept)),
        ("  paragraph", int((kept["type"] == "paragraph").sum())),
        ("  summary", int((kept["type"] == "summary").sum())),
        ("PDFs with pairs", kept["file_id"].nunique()),
        ("PDFs selected", sel["file_id"].nunique()),
        ("Eligible pages selected", len(sel)),
        ("Pages processed", len(pages)),
        ("Pages ok", int(status.get("ok", 0))),
        ("Pages skipped by model", int(status.get("skip", 0))),
        ("Pages invalid after retries", int(status.get("invalid", 0))),
        ("Pages API error", int(status.get("error", 0))),
        ("Pages with kept pairs", kept[["file_id", "page"]].drop_duplicates().shape[0]),
        ("Cost USD (full run)", round(run_cost, 4)),
        ("Cost USD (pilots/other)", round(args.extra_spent, 4)),
        ("Cost USD (total)", round(run_cost + args.extra_spent, 4)),
        ("Cost USD per processed page", round(run_cost / max(len(pages), 1), 6)),
        ("Input tokens (uncached)", int(all_pages["input_tokens"].sum())),
        ("Cache write tokens", int(all_pages["cache_write_tokens"].sum())),
        ("Cache read tokens", int(all_pages["cache_read_tokens"].sum())),
        ("Output tokens", int(all_pages["output_tokens"].sum())),
        ("Mean question length (words)", round(float(qw.mean()), 1)),
        ("Mean answer length (words)", round(float(aw.mean()), 1)),
        ("Median answer length (words)", float(aw.median())),
        ("Model", ", ".join(sorted(kept["model"].unique()))),
    ], columns=["metric", "value"])

    pf_pages = pages.pivot_table(index="file_id", columns="status", values="page", aggfunc="count", fill_value=0)
    pf_cost = all_pages.groupby("file_id")["cost_usd"].sum().rename("cost_usd")
    pf_sel = sel.groupby("file_id").agg(filename=("filename", "first"), pdf_pages=("pdf_pages", "first"),
                                        eligible_pages=("page", "size"))
    pf_pairs = kept.groupby("file_id").agg(pairs_kept=("qa_id", "size"))
    per_file = (pf_sel.join(pf_pages, how="left").join(pf_pairs, how="left").join(pf_cost, how="left")
                .fillna(0).reset_index())
    per_file["split"] = np.where(per_file["file_id"].isin(test_ids), "test",
                                 np.where(per_file["pairs_kept"] > 0, "train", "-"))
    per_file["cost_usd"] = per_file["cost_usd"].round(5)

    split_tbl = kept.groupby("split").agg(pdfs=("file_id", "nunique"), pairs=("qa_id", "size"))
    split_tbl["pages"] = kept.drop_duplicates(["file_id", "page"]).groupby("split").size()
    split_tbl["share_of_pairs"] = (split_tbl["pairs"] / split_tbl["pairs"].sum()).round(3)
    split_tbl = split_tbl.reset_index()
    test_files = per_file[per_file["split"] == "test"][["file_id", "filename", "pairs_kept"]]

    sample = kept.sample(min(50, len(kept)), random_state=SEED)[
        ["qa_id", "split", "type", "filename", "page", "question", "answer"]]

    out = qa_dir / "qa_summary.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        xl(totals).to_excel(w, sheet_name="Totals", index=False)
        xl(per_file).to_excel(w, sheet_name="PerFile", index=False)
        xl(split_tbl).to_excel(w, sheet_name="Split", index=False)
        xl(test_files).to_excel(w, sheet_name="Split", index=False, startrow=len(split_tbl) + 3)
        xl(sample).to_excel(w, sheet_name="Sample", index=False)
        for name, widths in {"Totals": [48, 20], "PerFile": [9, 70], "Sample": [18, 7, 10, 40, 6, 60, 100]}.items():
            for col, wd in zip("ABCDEFG", widths):
                w.sheets[name].column_dimensions[col].width = wd
    print(totals.to_string(index=False))
    print()
    print(split_tbl.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
