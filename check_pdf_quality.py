import logging
import re
import sys
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

FOLDER = Path("corpus")

logging.getLogger("pypdf").setLevel(logging.ERROR)
sys.stdout.reconfigure(encoding="utf-8")

PATTERNS = {
    "cid": r"\(cid:\d+\)",
    "replacement": "�",
    "private_use": r"[-]",
    "mojibake": r"[ÃÂ][\x80-\xbf]|â€",
    "spaced": r"\b(?:\w ){5,}\w\b",
    "ligature": r"[ﬀ-ﬆ]",
    "hyphen_break": r"\w-\n\w",
}
BROKEN = ("cid", "replacement", "private_use", "mojibake")


def repeated_lines(pages):
    if len(pages) < 5:
        return 0
    seen = Counter()
    for page in pages:
        seen.update({l.strip() for l in page.splitlines() if 3 < len(l.strip()) < 80})
    return sum(n > 0.4 * len(pages) for n in seen.values())


def check(path):
    pages = [p.extract_text() or "" for p in PdfReader(path).pages]
    text = "\n".join(pages)
    nonspace = len(re.sub(r"\s", "", text)) or 1
    counts = {name: len(re.findall(rx, text)) for name, rx in PATTERNS.items()}
    letters = sum(c.isalpha() for c in text) / nonspace
    empty = sum(len(p.strip()) < 100 for p in pages) / len(pages)

    issues = []
    if sum(counts[k] for k in BROKEN) / nonspace > 0.001:
        issues.append("broken chars")
    if letters < 0.55:
        issues.append("low letter ratio")
    if empty > 0.1:
        issues.append("empty pages")
    if counts["spaced"] > 20:
        issues.append("spaced letters")

    return len(pages), nonspace, letters, empty, counts, repeated_lines(pages), issues


header = f"{'file':38}{'pages':>6}{'letters':>9}{'empty':>7}{'broken':>7}{'spaced':>7}{'ligat':>7}{'hyph':>6}{'rep':>5}  status"
print(header)
print("-" * len(header))
flagged = []
for path in sorted(FOLDER.glob("*.pdf")):
    try:
        pages, chars, letters, empty, c, rep, issues = check(path)
    except Exception as e:
        print(f"{path.name[:37]:38} ERROR: {e}")
        flagged.append(path.name)
        continue
    broken = sum(c[k] for k in BROKEN)
    status = "OK" if not issues else "CHECK: " + ", ".join(issues)
    if issues:
        flagged.append(path.name)
    print(f"{path.name[:37]:38}{pages:6}{letters:9.0%}{empty:7.0%}{broken:7}{c['spaced']:7}{c['ligature']:7}{c['hyphen_break']:6}{rep:5}  {status}")

print(f"\n{len(flagged)} flagged.")
