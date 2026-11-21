"""Extract text from every PDF under data/raw and build an inventory.

Outputs
  data/text/<same subfolders>/<name>.txt   text per page, with "=== Page N ===" markers
  data/inventory/pdf_inventory.csv         one row per PDF
  data/inventory/pdf_inventory.xlsx        sheets: Inventory, Summary
  logs/extract.log                         errors and MuPDF warnings

Usage: python src/extract_pdfs.py [--raw data/raw]
"""
import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd
import pymupdf
import tiktoken
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "data" / "text"
INV_DIR = ROOT / "data" / "inventory"
LOG_FILE = ROOT / "logs" / "extract.log"

LONG_PDF_PAGES = 500          # "long" means at or over this many pages
SCANNED_CHARS_PER_PAGE = 50   # average under this means likely scanned
EMPTY_PAGE_CHARS = 10         # page with fewer non-whitespace chars counts as empty

COLUMNS = [
    "file_id", "rel_path", "subfolder", "filename", "size_MB", "pages", "chars",
    "words", "tokens", "tokens_per_page", "empty_pages", "likely_scanned",
    "extract_error", "sha256", "is_duplicate",
]

ENC = tiktoken.get_encoding("cl100k_base")
log = logging.getLogger("extract")


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    # Collect MuPDF warnings ourselves instead of letting them spam stderr.
    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_display_warnings(False)


def find_pdfs(raw_dir: Path) -> list[Path]:
    return sorted(
        (p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.relative_to(raw_dir).as_posix().lower(),
    )


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_tokens(text: str) -> int:
    # disallowed_special=() so a literal "<|endoftext|>" in a PDF doesn't raise
    return len(ENC.encode(text, disallowed_special=()))


def extract_one(pdf: Path, raw_dir: Path, file_id: str) -> dict:
    rel = pdf.relative_to(raw_dir)
    row = dict.fromkeys(COLUMNS)
    row.update(
        file_id=file_id,
        rel_path=rel.as_posix(),
        subfolder="" if rel.parent == Path(".") else rel.parent.as_posix(),
        filename=pdf.name,
        size_MB=round(pdf.stat().st_size / 2**20, 3),
        extract_error="",
    )
    try:
        row["sha256"] = sha256_of(pdf)
        pymupdf.TOOLS.reset_mupdf_warnings()
        with pymupdf.open(pdf) as doc:
            if doc.needs_pass:
                raise RuntimeError("encrypted: password required")
            pages, empty, page_errors = [], 0, 0
            for i, page in enumerate(doc, start=1):
                try:
                    txt = page.get_text("text")
                except Exception as e:  # one bad page shouldn't lose the whole file
                    page_errors += 1
                    log.warning("%s page %d: %s", rel, i, e)
                    txt = ""
                if len("".join(txt.split())) < EMPTY_PAGE_CHARS:
                    empty += 1
                pages.append(txt)

        warnings = pymupdf.TOOLS.mupdf_warnings()
        if warnings:
            log.warning("%s mupdf: %s", rel, warnings.replace("\n", " | "))

        out_path = TEXT_DIR / rel.with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "".join(f"=== Page {i} ===\n{t}\n" for i, t in enumerate(pages, start=1)),
            encoding="utf-8",
        )

        body = "\n".join(pages)  # counts exclude the page markers
        n = len(pages)
        chars = len(body)
        tokens = count_tokens(body)
        row.update(
            pages=n,
            chars=chars,
            words=len(body.split()),
            tokens=tokens,
            tokens_per_page=round(tokens / n, 1) if n else 0,
            empty_pages=empty,
            likely_scanned=(chars / n < SCANNED_CHARS_PER_PAGE) if n else True,
        )
        if page_errors:
            row["extract_error"] = f"{page_errors} page(s) failed"
    except Exception as e:
        row["extract_error"] = f"{type(e).__name__}: {e}"
        log.error("%s failed: %s", rel, row["extract_error"])
    return row


def build_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    ok = df[df["extract_error"] == ""]
    short = ok[ok["pages"] < LONG_PDF_PAGES]
    long_ = ok[ok["pages"] >= LONG_PDF_PAGES]

    totals = pd.DataFrame(
        [
            ("PDFs found", len(df)),
            ("Extracted OK", len(ok)),
            ("Broken (extract_error)", int((df["extract_error"] != "").sum())),
            ("Likely scanned", int(df["likely_scanned"].eq(True).sum())),
            ("Duplicates (extra copies)", int(df["is_duplicate"].sum())),
            ("Total size MB", round(df["size_MB"].sum(), 1)),
            ("Total pages", int(ok["pages"].sum())),
            ("Total words", int(ok["words"].sum())),
            ("Total tokens (cl100k_base)", int(ok["tokens"].sum())),
            ("Empty pages", int(ok["empty_pages"].sum())),
        ],
        columns=["metric", "value"],
    )
    by_length = pd.DataFrame(
        [
            (f"< {LONG_PDF_PAGES} pages", len(short), int(short["pages"].sum()), int(short["tokens"].sum())),
            (f">= {LONG_PDF_PAGES} pages", len(long_), int(long_["pages"].sum()), int(long_["tokens"].sum())),
        ],
        columns=["group", "pdfs", "pages", "tokens"],
    )
    dup_hashes = df.loc[df["is_duplicate"], "sha256"].unique()
    lists = {
        "By length": by_length,
        "Likely scanned files": df.loc[df["likely_scanned"].eq(True),
                                       ["file_id", "rel_path", "pages", "chars", "size_MB"]],
        "Broken files": df.loc[df["extract_error"] != "", ["file_id", "rel_path", "extract_error"]],
        "Duplicate groups (same sha256)": df.loc[df["sha256"].isin(dup_hashes),
                                                 ["sha256", "file_id", "rel_path", "is_duplicate"]]
                                            .sort_values(["sha256", "rel_path"]),
    }
    return totals, lists


def clean_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v)


def write_outputs(df: pd.DataFrame, totals: pd.DataFrame, lists: dict[str, pd.DataFrame]) -> Path:
    INV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INV_DIR / "pdf_inventory.csv", index=False, encoding="utf-8-sig")
    xlsx = INV_DIR / "pdf_inventory.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        clean_for_excel(df).to_excel(xw, sheet_name="Inventory", index=False)
        totals.to_excel(xw, sheet_name="Summary", index=False, startrow=1)
        ws = xw.sheets["Summary"]
        ws.cell(row=1, column=1, value="Totals")
        row = len(totals) + 4
        for title, table in lists.items():
            ws.cell(row=row, column=1, value=f"{title} ({len(table)})" if title != "By length" else title)
            if len(table):
                clean_for_excel(table).to_excel(xw, sheet_name="Summary", index=False, startrow=row)
                row += len(table) + 3
            else:
                ws.cell(row=row + 1, column=1, value="none")
                row += 3
        inv = xw.sheets["Inventory"]
        inv.freeze_panes = "A2"
        inv.auto_filter.ref = inv.dimensions
        for ws_ in (inv, ws):
            for col in ws_.columns:
                width = max(len(str(c.value or "")) for c in col)
                ws_.column_dimensions[col[0].column_letter].width = min(max(width + 2, 8), 70)
    return xlsx


def print_summary(totals: pd.DataFrame, lists: dict[str, pd.DataFrame]) -> None:
    with pd.option_context("display.width", 200, "display.max_colwidth", 90, "display.max_rows", 500):
        print("\n== Totals ==")
        print(totals.to_string(index=False))
        for title, table in lists.items():
            print(f"\n== {title} ({len(table)}) ==")
            print(table.to_string(index=False) if len(table) else "none")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()
    raw_dir = args.raw.resolve()

    setup_logging()
    pdfs = find_pdfs(raw_dir)
    log.info("found %d PDFs under %s", len(pdfs), raw_dir)
    if not pdfs:
        raise SystemExit(f"No PDFs under {raw_dir}")

    width = max(4, len(str(len(pdfs))))
    rows = [extract_one(p, raw_dir, f"P{i:0{width}d}")
            for i, p in enumerate(tqdm(pdfs, desc="Extracting", unit="pdf"), start=1)]

    df = pd.DataFrame(rows, columns=COLUMNS)
    # First copy (by rel_path order) is the original; later identical files are duplicates.
    df["is_duplicate"] = df["sha256"].notna() & df.duplicated("sha256", keep="first")
    totals, lists = build_summary(df)
    xlsx = write_outputs(df, totals, lists)
    log.info("done: %d PDFs, %d errors", len(df), int((df["extract_error"] != "").sum()))

    print_summary(totals, lists)
    print(f"\nWrote {xlsx.relative_to(ROOT)}, pdf_inventory.csv, text in "
          f"{TEXT_DIR.relative_to(ROOT)}/, log in {LOG_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
