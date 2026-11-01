import json
import os
import time
from collections import defaultdict

from langchain_openai import ChatOpenAI

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
MODEL = os.getenv("LLM_MODEL", "google/gemma-4-e4b")
USE_DUMMY = os.getenv("LLM_BACKEND") == "dummy"

llm = ChatOpenAI(base_url=BASE_URL, api_key="lm-studio", model=MODEL, temperature=0, timeout=180)

DUMMY_REPLIES = {
    "rewrite": lambda p: json.dumps({"query": p.split("Question:")[-1].strip(), "year": None}),
    "plan": lambda p: json.dumps({"intent": "lookup", "questions": []}),
    "chat": lambda p: "[dummy LLM] chat reply.",
    "summary": lambda p: "[dummy LLM] summary of the evidence below.",
    "grade": lambda p: "yes",
    "policy": lambda p: json.dumps({"verdict": "no_precedent", "policies": [], "reason": "dummy LLM"}),
    "answer": lambda p: "[dummy LLM] " + p.split("Context:")[-1].strip()[:300],
}


STATS = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "in": 0, "out": 0})


def ask(prompt, task):
    start = time.perf_counter()
    if USE_DUMMY:
        content, usage = DUMMY_REPLIES[task](prompt), {}
    else:
        try:
            reply = llm.invoke(prompt)
        except Exception as exc:
            detail = getattr(exc, "body", None) or getattr(exc, "message", str(exc))
            raise RuntimeError(f"LLM call failed for task '{task}' ({len(prompt)} chars): {detail}") from exc
        content, usage = reply.content, reply.usage_metadata or {}
    stat = STATS[task]
    stat["calls"] += 1
    stat["seconds"] += time.perf_counter() - start
    stat["in"] += usage.get("input_tokens", 0)
    stat["out"] += usage.get("output_tokens", 0)
    return content
