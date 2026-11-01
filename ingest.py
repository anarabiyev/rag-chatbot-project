import argparse
from pathlib import Path

from rag import chunking
from rag.cleaning import clean_pages
from rag.loader import load_pdf, read_csv
from rag.paths import CORPUS
from rag.store import add_chunks, get_collection

CHUNKERS = {"recursive": chunking.recursive, "parent": chunking.parent_child}

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", default=str(CORPUS))
parser.add_argument("--strategy", nargs="+", choices=CHUNKERS, default=list(CHUNKERS))
args = parser.parse_args()

corpus = Path(args.corpus)
rows = read_csv(corpus / "manifest.csv")
listed = {r["file"] for r in rows}
for path in corpus.glob("*.pdf"):
    if path.name not in listed:
        print(f"not in manifest, skipped: {path.name}")

pages = []
for row in rows:
    doc = clean_pages(load_pdf(corpus / row["file"], row))
    print(f"{row['doc_id']:34} {row['year']}  {len(doc)} pages")
    pages += doc

for name in args.strategy:
    chunks = CHUNKERS[name](pages)
    print(f"{name}: {len(chunks)} chunks")
    add_chunks(get_collection(f"arla_{name}", reset=True), chunks)
