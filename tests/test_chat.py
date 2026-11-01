from langgraph.graph import END

from agent import nodes
from agent.planner import COMPETITORS


def test_competitors_exclude_arla():
    assert "Arla" not in COMPETITORS and {"Lactalis", "FrieslandCampina", "Danone"} <= set(COMPETITORS)


def test_chat_intent_skips_rag_and_news():
    assert nodes.route_after_plan({"intent": "chat", "subquestions": ["q"]}) == "chat"
    assert nodes.route_after_plan({"intent": "out_of_scope"}) == END


def test_chat_prompt_has_history_and_competitors(monkeypatch):
    seen = {}
    monkeypatch.setattr(nodes, "ask", lambda prompt, task: seen.update(prompt=prompt, task=task) or " Your last question was X. ")
    history = [{"question": "What was Arla's revenue in 2024?", "answer": "EUR 13,770 million."}]
    update = nodes.chat_node({"question": "what was my last question?", "history": history})
    assert update["answer"] == "Your last question was X."
    assert seen["task"] == "chat"
    assert "What was Arla's revenue in 2024?" in seen["prompt"] and "Danone" in seen["prompt"]


def test_chat_without_history_says_so(monkeypatch):
    seen = {}
    monkeypatch.setattr(nodes, "ask", lambda prompt, task: seen.update(prompt=prompt) or "ok")
    nodes.chat_node({"question": "what was my last question?"})
    assert "(no earlier messages)" in seen["prompt"]


def test_summary_facts_name_the_tracked_companies():
    from agent.report import facts

    news = {"companies": {"Arla": {}, "Danone": {}}, "fetched_at": "2026-10-30", "origin": "snapshot", "days": 30}
    news["companies"] = {c: {"stories": 1, "avg_sentiment": None, "negative_share": None, "events": 0, "latest": [], "most_negative": []}
                         for c in ("Arla", "Lactalis", "Danone")}
    text = facts({"question": "Who are the competitors?", "rag_answer": "Not in the documents.", "news": news})
    assert "Companies in the market data: Arla, Lactalis, Danone" in text


def test_summary_prompt_answers_the_question_from_market_data():
    assert "Answer the question first" in nodes.SUMMARY and "market data" in nodes.SUMMARY
