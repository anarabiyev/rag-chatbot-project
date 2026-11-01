import re
import unicodedata
from collections import Counter


def repeated_lines(texts):
    seen = Counter()
    for text in texts:
        seen.update({l.strip() for l in text.splitlines() if 3 < len(l.strip()) < 80})
    return {line for line, n in seen.items() if n > 0.4 * len(texts)}


def clean_pages(pages):
    texts = [unicodedata.normalize("NFKC", p["text"]) for p in pages]
    noise = repeated_lines(texts) if len(texts) >= 5 else set()
    cleaned = []
    for page, text in zip(pages, texts):
        text = "\n".join(l for l in text.splitlines() if l.strip() not in noise)
        text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) >= 100:
            cleaned.append({**page, "text": text})
    return cleaned
