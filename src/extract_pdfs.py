"""Step 1: extract text from every PDF in data/raw and build an inventory.

Outputs
  data/text/<same subfolders>/<name>.txt   page-by-page text with page markers
  data/inventory/pdf_inventory.csv         one row per PDF
  data/inventory/pdf_inventory.xlsx        sheets: inventory, summary
  logs/extract_pdfs_<timestamp>.log        errors and warnings

Usage: python src/extract_pdfs.py [--raw data/raw] [--force]
"""
import argparse
import hashlib
import logging
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import tiktoken
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "data" / "text"
INV_DIR = ROOT / "data" / "inventory"
LOG_DIR = ROOT / "logs"

PAGE_MARKER = "\n\n===== PAGE {n} / {total} =====\n"
LONG_PDF_PAGES = 500
# A page with fewer than this many non-space chars counts as empty.
EMPTY_PAGE_CHARS = 20
# A PDF is "likely scanned" when its average text per page is below this.
SCANNED_CHARS_PER_PAGE = 100

ENC = tiktoken.get_encoding("cl100k_base")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_tokens(text: str) -> int:
    # disallowed_special=() so literal "<|endoftext|>" in a PDF doesn't raise
    return len(ENC.encode(text, disallowed_special=()))


def extract_one(pdf: Path, raw_dir: Path, force: bool, log: logging.Logger) -> dict:
    rel = pdf.relative_to(raw_dir)
    row = {
        "relative_path": rel.as_posix(),
        "subfolder": rel.parent.as_posix() if rel.parent != Path(".") else "",
        "filename": pdf.name,
        "size_MB": round(pdf.stat().st_size / 1e6, 3),
        "pages": None, "chars": None, "words": None, "tokens": None,
        "avg_tokens_per_page": None, "empty_pages": None,
        "likely_scanned": None, "extract_error": "",
        "sha256": sha256_of(pdf),
    }
    out_path = TEXT_DIR / rel.with_suffix(".txt")
    try:
        with fitz.open(pdf) as doc:
            if doc.needs_pass:
                raise RuntimeError("encrypted (password required)")
            total = doc.page_count
            parts, page_errors, empty = [], 0, 0
            for i, page in enumerate(doc, start=1):
                try:
                    txt = page.get_text("text")
                except Exception as e:  # one bad page shouldn't lose the file
                    page_errors += 1
                    log.warning("%s page %d: %s", rel, i, e)
                    txt = ""
                if len("".join(txt.split())) < EMPTY_PAGE_CHARS:
                    empty += 1
                parts.append(PAGE_MARKER.format(n=i, total=total) + txt)
        text = "".join(parts).lstrip("\n")
        body = "".join(p.split("=====\n", 1)[1] for p in parts)  # text without markers
        if force or not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
        row.update(
            pages=total,
            chars=len(body),
            words=len(body.split()),
            tokens=count_tokens(body),
            empty_pages=empty,
        )
        row["avg_tokens_per_page"] = round(row["tokens"] / total, 1) if total else 0
        row["likely_scanned"] = total == 0 or (
            len("".join(body.split())) / total < SCANNED_CHARS_PER_PAGE
        )
        if page_errors:
            row["extract_error"] = f"{page_errors} page(s) failed"
        if total == 0:
            row["extract_error"] = "zero pages"
    except Exception as e:
        row["extract_error"] = f"{type(e).__name__}: {e}"
        log.error("%s: %s", rel, row["extract_error"])
    return row


def build_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ok = df[df["pages"].notna()]
    uniq = ok[ok["duplicate_of"] == ""]
    totals = pd.DataFrame(
        [
            ("PDFs found", len(df)),
            ("PDFs extracted", len(ok)),
            ("PDFs with errors", int((df["extract_error"] != "").sum())),
            ("Likely scanned", int(df["likely_scanned"].eq(True).sum())),
            ("Duplicate files (extra copies)", int(df["duplicate_of"].ne("").sum())),
            ("Total size (MB)", round(df["size_MB"].sum(), 1)),
            ("Total pages", int(ok["pages"].sum())),
            ("Total words", int(ok["words"].sum())),
            ("Total tokens (cl100k_base)", int(ok["tokens"].sum())),
            ("Empty pages", int(ok["empty_pages"].sum())),
            ("Unique PDFs (by sha256)", int(df["sha256"].nunique())),
            ("Unique pages", int(uniq["pages"].sum())),
            ("Unique tokens (cl100k_base)", int(uniq["tokens"].sum())),
        ],
        columns=["metric", "value"],
        dtype=object,  # keep ints as ints next to the MB float
    )
    long_ = ok["pages"] > LONG_PDF_PAGES
    by_len = pd.DataFrame(
        [
            (f"<= {LONG_PDF_PAGES} pages", int((~long_).sum()),
             int(ok.loc[~long_, "pages"].sum()), int(ok.loc[~long_, "tokens"].sum())),
            (f"> {LONG_PDF_PAGES} pages", int(long_.sum()),
             int(ok.loc[long_, "pages"].sum()), int(ok.loc[long_, "tokens"].sum())),
        ],
        columns=["group", "pdfs", "total_pages", "total_tokens"],
    )
    issues = []
    for _, r in df.iterrows():
        if r["extract_error"]:
            issues.append((r["file_id"], "broken/error", r["extract_error"], r["relative_path"]))
        if r["likely_scanned"] is True:
            issues.append((r["file_id"], "likely scanned",
                           f"{r['pages']} pages, {r['chars']} chars", r["relative_path"]))
        if r["duplicate_of"]:
            issues.append((r["file_id"], "duplicate", f"same sha256 as {r['duplicate_of']}",
                           r["relative_path"]))
    flagged = pd.DataFrame(issues, columns=["file_id", "issue", "detail", "relative_path"])
    return {"totals": totals, "by_length": by_len, "flagged": flagged}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--force", action="store_true", help="overwrite existing .txt files")
    args = ap.parse_args()
    raw_dir = Path(args.raw).resolve()

    LOG_DIR.mkdir(exist_ok=True)
    INV_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"extract_pdfs_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("extract_pdfs")
    fitz.TOOLS.mupdf_display_errors(False)  # MuPDF warnings go to the log, not the terminal

    pdfs = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    log.info("found %d PDFs under %s", len(pdfs), raw_dir)

    rows = []
    for pdf in tqdm(pdfs, desc="PDFs", unit="pdf"):
        rows.append(extract_one(pdf, raw_dir, args.force, log))
        warnings = fitz.TOOLS.mupdf_warnings()
        if warnings:
            log.warning("%s mupdf: %s", pdf.relative_to(raw_dir), warnings.replace("\n", " | "))

    df = pd.DataFrame(rows)
    df.insert(0, "file_id", [f"F{i:04d}" for i in range(1, len(df) + 1)])
    first_id = df.groupby("sha256")["file_id"].transform("first")
    df["duplicate_of"] = first_id.where(first_id != df["file_id"], "")

    df.to_csv(INV_DIR / "pdf_inventory.csv", index=False)
    summary = build_summary(df)
    with pd.ExcelWriter(INV_DIR / "pdf_inventory.xlsx", engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="inventory", index=False)
        row = 0
        for title, part in summary.items():
            pd.DataFrame([[title.upper()]]).to_excel(
                xw, sheet_name="summary", startrow=row, index=False, header=False)
            part.to_excel(xw, sheet_name="summary", startrow=row + 1, index=False)
            row += len(part) + 4
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 70)

    log.info("done: %d PDFs", len(df))
    for title, part in summary.items():
        print(f"\n== {title} ==")
        print(part.to_string(index=False) if len(part) else "(none)")
    print(f"\nLog: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
