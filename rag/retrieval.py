from rag.loader import read_csv
from rag.paths import MANIFEST


def doc_types(manifest=MANIFEST):
    return sorted({row["doc_type"] for row in read_csv(manifest)})


def retrieve(collection, query, k=5, year=None, doc_type=None):
    filters = []
    if year:
        filters.append({"year": year})
    if doc_type:
        filters.append({"doc_type": {"$in": doc_type}} if isinstance(doc_type, list) else {"doc_type": doc_type})
    where = {"$and": filters} if len(filters) > 1 else filters[0] if filters else None

    hits = collection.query(query_texts=[query], n_results=k * 2, where=where)
    docs, seen = [], set()
    for text, meta, dist in zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0]):
        parent = meta.get("parent", text)
        if parent in seen:
            continue
        seen.add(parent)
        docs.append({**{k_: v for k_, v in meta.items() if k_ != "parent"}, "text": parent, "match": text, "distance": dist})
    return docs[:k]
