import csv
import logging

from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_pdf(path, row):
    meta = {
        "doc_id": row["doc_id"],
        "source": row["file"],
        "title": row["title"],
        "doc_type": row["doc_type"],
        "year": int(row["year"]),
    }
    texts = [page.extract_text() or "" for page in PdfReader(path).pages]
    return [{**meta, "page": i + 1, "text": t} for i, t in enumerate(texts)]
