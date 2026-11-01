import json
from functools import lru_cache

from rag.paths import SNAPSHOTS

MODEL = "ProsusAI/finbert"
CACHE = SNAPSHOTS / "sentiment_finbert.json"


@lru_cache(maxsize=1)
def classifier():
    from transformers import pipeline

    return pipeline("text-classification", model=MODEL, top_k=None)


def score(titles):
    """Signed sentiment per headline: P(positive) - P(negative), from -1 to 1."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    new = [t for t in dict.fromkeys(titles) if t not in cache]
    if new:
        for title, probs in zip(new, classifier()(new, batch_size=16, truncation=True)):
            p = {x["label"].lower(): x["score"] for x in probs}
            cache[title] = round(p["positive"] - p["negative"], 3)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return [cache[t] for t in titles]
