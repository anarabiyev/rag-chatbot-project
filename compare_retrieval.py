import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from rag.graph import REWRITE, norm, parse_json
from rag.llm import ask
from rag.loader import read_csv
from rag.retrieval import doc_types, retrieve
from rag.store import EMBED_MODEL, get_collection

sys.stdout.reconfigure(encoding="utf-8")

K = 5
REWRITES = Path("eval/rewrites.json")
# the production rewrite no longer asks for doc_type; the `full` mode still needs it
TYPED_REWRITE = (REWRITE
                 .replace("one.\n", "one. Set doc_type only if the question clearly targets one of: {types}.\n", 1)
                 .replace('null}}', 'null, "doc_type": "..." or null}}'))
STRATEGIES = ["recursive", "parent"]
# none: raw question | query: rewritten query | year: raw question + year filter
# qyear: rewritten query + year filter | full: rewritten query + year + doc_type
MODES = {
    "none": lambda q, a: (q, None, None),
    "query": lambda q, a: (a[0], None, None),
    "year": lambda q, a: (q, a[1], None),
    "qyear": lambda q, a: (a[0], a[1], None),
    "full": lambda q, a: (a[0], a[1], a[2]),
}

questions = read_csv("eval/questions.csv")

types = doc_types()


def analyze(question):
    data = parse_json(ask(TYPED_REWRITE.format(types=", ".join(types), retry="", question=question), "rewrite"))
    year = data.get("year") if isinstance(data.get("year"), int) else None
    doc_type = data.get("doc_type") if data.get("doc_type") in types else None
    return [data.get("query") or question, year, doc_type]


cache = json.loads(REWRITES.read_text(encoding="utf-8")) if REWRITES.exists() else {}
for q in questions:
    if q["question"] not in cache:
        cache[q["question"]] = analyze(q["question"])
        REWRITES.write_text(json.dumps(cache, indent=1), encoding="utf-8")
analysis = [cache[q["question"]] for q in questions]


def rank_of(docs, row):
    pages = {int(p) for p in row["pages"].split("|")}
    for i, d in enumerate(docs, 1):
        if d["source"] == row["source"] and d["page"] in pages:
            return i


def hit(ranks, n):
    return sum(r is not None and r <= n for r in ranks) / len(ranks)


def mrr(ranks):
    return sum(1 / r for r in ranks if r) / len(ranks)


print(f"embedding model: {EMBED_MODEL}")
print(f"{'strategy':10} {'mode':6} {'hit@1':>6} {'hit@3':>6} {'hit@'+str(K):>6} {'mrr':>6} {'kw@'+str(K):>6}")
for strategy in STRATEGIES:
    collection = get_collection(f"arla_{strategy}")
    for mode, plan in MODES.items():
        ranks, kw_hits, by_type, misses, seconds = [], [], defaultdict(list), [], 0.0
        for row, ana in zip(questions, analysis):
            query, year, doc_type = plan(row["question"], ana)
            start = time.perf_counter()
            docs = retrieve(collection, query, K, year, doc_type)
            seconds += time.perf_counter() - start
            rank = rank_of(docs, row)
            ranks.append(rank)
            kw_hits.append(any(norm(row["keyword"]) in norm(d["text"]) for d in docs))
            by_type[row["type"]].append(rank)
            if rank is None:
                top = f"{docs[0]['source'][:34]} p.{docs[0]['page']}" if docs else "no results"
                misses.append(f"  miss [{row['type']}] {row['question'][:58]} | year={year} type={doc_type} | top: {top}")
        print(f"{strategy:10} {mode:6} {hit(ranks, 1):6.2f} {hit(ranks, 3):6.2f} {hit(ranks, K):6.2f} {mrr(ranks):6.2f} {sum(kw_hits) / len(kw_hits):6.2f}  {1000 * seconds / len(ranks):5.0f} ms/query")
        for t, r in by_type.items():
            print(f"    {t:10} n={len(r):2}  hit@{K}={hit(r, K):.2f}  mrr={mrr(r):.2f}")
        print("\n".join(misses))
