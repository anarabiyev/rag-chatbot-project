import csv
import re
import sys
from pathlib import Path

from pypdf import PdfReader

DOC_TYPES = [
    ("annual", "annual_report"),
    ("ar25", "annual_report"),
    ("hy26", "half_year_report"),
    ("slavery", "modern_slavery"),
    ("suppliers", "supplier_code"),
    ("conduct", "code_of_conduct"),
    ("bribery", "anti_bribery"),
    ("human-rights", "human_rights"),
    ("sourcing", "responsible_sourcing"),
    ("financing", "financing_framework"),
    ("tax", "tax_strategy"),
    ("articles", "articles_of_association"),
    ("arlag", "farm_programme"),
    ("bbfaw", "animal_welfare"),
    ("policy-statement", "policy_statement"),
]

args = [a for a in sys.argv[1:] if a != "--force"]
corpus = Path(args[0] if args else "corpus")
if (corpus / "manifest.csv").exists() and "--force" not in sys.argv:
    sys.exit(f"{corpus / 'manifest.csv'} already exists and holds hand-corrected metadata. Pass --force to overwrite it.")
rows, used = [], set()
for path in sorted(corpus.glob("*.pdf")):
    name = path.stem.lower()
    doc_type = next((t for key, t in DOC_TYPES if key in name), "other")
    short = re.search(r"(?<![a-z])(?:ar|hy)(\d\d)(?!\d)", name)
    first_page = PdfReader(path).pages[0].extract_text() or ""
    found = re.search(r"20[12]\d", name) or re.search(r"20[12]\d", first_page)
    year = 2000 + int(short[1]) if short else int(found[0]) if found else 0

    doc_id = f"{doc_type}_{year}"
    while doc_id in used:
        doc_id += "_b"
    used.add(doc_id)
    title = re.sub(r"[-_]+", " ", path.stem).strip().title()
    rows.append({"doc_id": doc_id, "file": path.name, "title": title,
                 "doc_type": doc_type, "year": year, "url": ""})

with open(corpus / "manifest.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(rows)} files -> {corpus / 'manifest.csv'}")
