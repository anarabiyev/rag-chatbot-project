import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    intent: str
    action: str
    subquestions: list[str]
    companies: list[str]
    docs: list[dict]
    rag_results: Annotated[list[dict], operator.add]
    rag_answer: str
    quote: str
    rag_relevant: bool
    rag_attempts: int
    news: dict
    policy: dict
    risk: dict
    warnings: list[str]
    answer: str
    critic_ok: bool
    retries: int
    trace: Annotated[list[str], operator.add]
    timings: Annotated[list[tuple], operator.add]
