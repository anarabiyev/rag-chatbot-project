import json
import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from rag.llm import ask
from rag.retrieval import retrieve
from rag.store import get_collection

MAX_RETRIES = 1
GOOD_DISTANCE, BAD_DISTANCE = 0.30, 0.45
CONTEXT_CHARS = 6000

REWRITE = """Rewrite the question as a short search query for Arla company documents (annual reports, policies, codes of conduct).
Extract the year if the question names one.
Reply with JSON only: {{"query": "...", "year": 2024 or null}}
{retry}
Question: {question}"""

GRADE = """Do the passages below contain information needed to answer the question? Answer only "yes" or "no".

Question: {question}

Passages:
{context}"""

ANSWER = """Answer the question using only the context. Cite sources as [1], [2]. If the context does not contain the answer, say so.
State what each figure refers to (the whole company or a segment, region or product) and for which year.
Finish with one line that starts with "Quote:" and copies, word for word, the sentence or table row from the context that supports your answer, followed by its source number, for example: Quote: "Revenue 13,770 13,674" [2]

Context:
{context}

Question: {question}
Answer:"""


class State(TypedDict, total=False):
    question: str
    query: str
    year: int | None
    docs: list[dict]
    relevant: bool
    attempts: int
    answer: str
    quote: str
    trace: Annotated[list[str], operator.add]


def window(doc, limit):
    text = doc["text"]
    if len(text) <= limit:
        return text
    at = text.find(doc.get("match", "")[:60]) if doc.get("match") else -1
    start = 0 if at < 0 else max(0, min(at - limit // 4, len(text) - limit))
    return text[start:start + limit]


def format_context(docs, budget=CONTEXT_CHARS):
    limit = budget // max(len(docs), 1)
    return "\n\n".join(f"[{i}] {d['source']} p.{d['page']} ({d['year']})\n{window(d, limit)}" for i, d in enumerate(docs, 1))


def split_quote(text):
    match = re.search(r"\n?\s*quote:\s*", text, re.I)
    if not match:
        return text.strip(), ""
    return text[:match.start()].strip(), text[match.end():].strip()


def norm(text):
    return " ".join(text.lower().split())


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(match.group())
    except (AttributeError, ValueError):
        return {}


def analyze(question, previous=None):
    retry = f'A previous search for "{previous}" found nothing useful. Use different keywords.\n' if previous else ""
    data = parse_json(ask(REWRITE.format(retry=retry, question=question), "rewrite"))
    year = data.get("year") if isinstance(data.get("year"), int) else None
    return data.get("query") or question, year


def build_rag_graph(strategy="parent", k=5, use_rewrite=True, use_grade=True):
    collection = get_collection(f"arla_{strategy}")

    def rewrite(state):
        attempts = state.get("attempts", 0) + 1
        question = state["question"]
        if not use_rewrite:
            return {"query": question, "attempts": attempts, "trace": ["rewrite: skipped"]}
        previous = state["query"] if attempts > 1 else None
        query, year = analyze(question, previous)
        if attempts > 1:
            year = None
        return {"query": query, "year": year, "attempts": attempts, "trace": [f"rewrite: '{query}' (year={year})"]}

    def retrieve_node(state):
        docs = retrieve(collection, state["query"], k, state.get("year"))
        return {"docs": docs, "trace": [f"retrieve: {len(docs)} chunks"]}

    def grade(state):
        if not use_grade:
            return {"relevant": True, "trace": ["grade: skipped"]}
        if not state["docs"]:
            return {"relevant": False, "trace": ["grade: no chunks"]}
        best = min(d["distance"] for d in state["docs"])
        if best <= GOOD_DISTANCE or best > BAD_DISTANCE:
            relevant = best <= GOOD_DISTANCE
            return {"relevant": relevant, "trace": [f"grade: {'relevant' if relevant else 'not relevant'} (best distance {best:.2f}, no LLM)"]}
        reply = ask(GRADE.format(question=state["question"], context=format_context(state["docs"])), "grade")
        relevant = reply.strip().lower().startswith("yes")
        return {"relevant": relevant, "trace": [f"grade: {'relevant' if relevant else 'not relevant'} (best distance {best:.2f}, LLM)"]}

    def generate(state):
        if not state["docs"] or not state["relevant"]:
            return {"answer": "I could not find this in the Arla documents.", "quote": "", "trace": ["generate: no evidence"]}
        prompt = ANSWER.format(context=format_context(state["docs"]), question=state["question"])
        answer, quote = split_quote(ask(prompt, "answer"))
        return {"answer": answer, "quote": quote, "trace": ["generate: answered"]}

    def route(state):
        return "generate" if state["relevant"] or state["attempts"] > MAX_RETRIES else "rewrite"

    graph = StateGraph(State)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade)
    graph.add_node("generate", generate)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route, {"generate": "generate", "rewrite": "rewrite"})
    graph.add_edge("generate", END)
    return graph.compile()
