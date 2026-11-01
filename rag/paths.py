from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
MANIFEST = CORPUS / "manifest.csv"
DB_PATH = ROOT / "chroma_db"
SNAPSHOTS = ROOT / "snapshots"
