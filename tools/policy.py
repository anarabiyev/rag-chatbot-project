from rag.graph import format_context, parse_json
from rag.llm import ask
from rag.retrieval import retrieve

POLICY_TYPES = [
    "anti_bribery", "code_of_conduct", "supplier_code", "human_rights", "responsible_sourcing",
    "modern_slavery", "policy_statement", "tax_strategy", "animal_welfare", "articles_of_association",
]
VERDICTS = ("consistent", "conflict", "no_precedent")
MAX_DISTANCE = 0.35

PROMPT = """You check a proposed business action against Arla's policies.
- conflict: a passage forbids or restricts the action
- consistent: a passage explicitly allows it, or it clearly follows the rules
- no_precedent: the passages do not address the action
Reply with JSON only: {{"verdict": "conflict" or "consistent" or "no_precedent", "policies": [numbers of the supporting passages], "reason": "one sentence"}}

Action: {action}

Passages:
{context}"""


def policy_docs(collection, action, k=6):
    return retrieve(collection, action, k, doc_type=POLICY_TYPES)


def latest_only(docs):
    newest = {}
    for d in docs:
        newest[d["doc_type"]] = max(newest.get(d["doc_type"], 0), d["year"])
    return [d for d in docs if d["year"] == newest[d["doc_type"]]]


def compare_policy(action, docs):
    docs = [d for d in latest_only(docs) if d["distance"] <= MAX_DISTANCE]
    if not docs:
        return {"verdict": "no_precedent", "policies": [], "reason": "No relevant policy passage found.", "docs": []}

    data = parse_json(ask(PROMPT.format(action=action, context=format_context(docs)), "policy"))
    ids = [int(i) for i in data.get("policies", []) if str(i).isdigit() and 1 <= int(i) <= len(docs)]
    verdict = data.get("verdict")
    if verdict not in VERDICTS or (verdict != "no_precedent" and not ids):
        verdict, ids = "no_precedent", []
    matched = []
    for i in ids:
        doc = docs[i - 1]
        entry = {k: doc[k] for k in ("doc_id", "title", "doc_type", "source", "page")}
        if entry not in matched:
            matched.append(entry)
    return {"verdict": verdict, "policies": matched, "reason": data.get("reason", ""), "docs": docs}
