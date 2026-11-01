import re

from agent.memory import format_history
from news.loader import COMPANIES
from rag.graph import parse_json
from rag.llm import ask

NAMES = {name.lower().replace("campina", ""): name for name in COMPANIES}  # "friesland" also matches
INTENTS = ("lookup", "market", "decision", "chat", "out_of_scope")
COMPETITORS = [name for name in NAMES.values() if name != "Arla"]

# The small local model never labelled a proposed action as a decision, so clear phrasings are decided by rules.
DECISION = re.compile(
    r"\b(should|shall) (we|i|arla)\b"
    r"|\b(can|could|may|might) (we|i)\b"
    r"|\b(is it|would it be|will it be) (ok|okay|acceptable|allowed|permitted|appropriate|fine|a problem)\b"
    r"|\b(we are|we're|i am|i'm) (considering|planning|thinking about|thinking of)\b"
    r"|\bwe (plan|propose|intend|want) to\b",
    re.I)
COMPETITOR_WORDS = re.compile(r"\b(competitors?|rivals?|competition|peers)\b", re.I)

PROMPT = """Classify the question for an Arla Foods decision-support assistant and split it if needed.
{history}
Assistant facts: it tracks news for Arla and these competitors: {competitors}. It answers from 22 Arla documents.

intent:
- lookup: what Arla's documents say, or what Arla did in the past
- market: recent news or sentiment about Arla or competitors ("latest news about Danone", "how is the market doing")
- decision: whether Arla should or can do something that is proposed
- chat: asks for something the assistant already knows, from the conversation or the assistant facts above, so nothing needs
  searching ("what was my last question?", "what can you do?", "what are the names of the competitors?", "which companies do you track?")
- out_of_scope: clearly unrelated to Arla, dairy or the food business. Use it only when sure.{follow_up}

questions: lookup and market only. If the question needs separate searches (several years or documents, a comparison),
give 2 or 3 short, self-contained questions that keep company names, years and units. Otherwise give an empty list.

Reply with JSON only: {reply}

Question: {question}"""

HISTORY = """
Conversation so far (oldest first):
{turns}

The question below is the newest message in this conversation.
"""

FOLLOW_UP = """
A question that refers to the conversation (for example "and in 2023?", "why?", "what about Danone?") is not out_of_scope.

standalone: rewrite the question so it can be understood without the conversation. Replace pronouns and
missing years, companies or topics with what the conversation says. If it is already self-contained, repeat it unchanged."""


def looks_like_decision(question):
    return bool(DECISION.search(question))


def find_companies(question):
    if COMPETITOR_WORDS.search(question):  # "Arla's competitors" names only Arla, but the answer needs the rivals
        return list(NAMES.values())
    text = question.lower()
    return [name for key, name in NAMES.items() if key in text]


def split(question, intent, questions):
    parts = [q.strip() for q in questions if isinstance(q, str) and q.strip()] if isinstance(questions, list) else []
    parts = list(dict.fromkeys(parts))
    return parts[:3] if intent in ("lookup", "market") and len(parts) >= 2 else [question]


def standalone(question, value):
    return value.strip() if isinstance(value, str) and value.strip() else question


def plan(question, history=None):
    prompt = PROMPT.format(
        history=HISTORY.format(turns=format_history(history)) if history else "",
        follow_up=FOLLOW_UP if history else "",
        competitors=", ".join(COMPETITORS),
        reply='{"intent": "lookup", "standalone": "...", "questions": []}' if history else '{"intent": "lookup", "questions": []}',
        question=question)
    if looks_like_decision(question) and not history:
        data = {"intent": "decision"}  # no LLM call needed
    else:
        data = parse_json(ask(prompt, "plan"))
    intent = data.get("intent") if data.get("intent") in INTENTS else "lookup"
    if history and intent != "chat":  # a question about the conversation must reach the chat node as written
        question = standalone(question, data.get("standalone"))
    if intent != "chat" and looks_like_decision(question):
        intent = "decision"
    return {
        "question": question,
        "intent": intent,
        "subquestions": split(question, intent, data.get("questions")),
        "companies": find_companies(question) or list(NAMES.values()),
        "action": question.strip().rstrip("?"),
    }
