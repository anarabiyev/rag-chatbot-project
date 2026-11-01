import re

from langgraph.graph import END
from langgraph.types import Send
from rapidfuzz import fuzz

from agent.memory import format_history
from agent.planner import COMPETITORS, plan
from agent.report import evidence_line, facts, market_section, policy_section, risk_section, sources_section
from news.tool import get_market_news
from rag.graph import norm
from rag.llm import USE_DUMMY, ask
from tools.policy import POLICY_TYPES, compare_policy, policy_docs
from tools.risk import evidence_summary, score_risk

MAX_RETRIES = 1
NUMBER = re.compile(r"\d[\d,.]*\d")
NOT_FOUND = re.compile(r"\b(does not|doesn't|do not|did not) (contain|provide|include|mention|name|specify)\b|could not find|no information", re.I)
OPEN_ISSUE = re.compile(r"\b(pending|expected to|is expected|are expected|planned|proposed|not yet|yet to|subject to approval)\b", re.I)

SUMMARY = """Write a 3-4 sentence answer for a new Arla employee. Use only the facts below.
Answer the question first. If the internal evidence does not contain the answer, use the market data: it names the companies
that are tracked. Do not change the policy verdict or the risk level.

{facts}"""

CHAT = """You are the Arla decision-support assistant. Answer the user's message using only the facts and the conversation below.
If the answer is not there, say so briefly. The user's newest message is the last line, so "my last question" means the user message before it.

Facts about this assistant:
- It answers from 22 public Arla documents (annual reports, policies, codes of conduct), with citations.
- It tracks recent news for Arla and its competitors: {competitors}.
- It checks a proposed decision against Arla's policies and gives a risk score.

Conversation so far (oldest first):
{history}

User: {question}
Answer:"""

OUT_OF_SCOPE = "This question is not about Arla, dairy or the food business, so I did not search the documents or the news."


def planner_node(state):
    result = plan(state["question"], state.get("history"))
    parts = len(result["subquestions"])
    rewritten = f", asked as '{result['question']}'" if result["question"] != state["question"] else ""
    update = {**result, "retries": 0,
              "trace": [f"planner: {result['intent']}{rewritten}" + (f", split into {parts} questions" if parts > 1 else "")]}
    if result["intent"] == "out_of_scope":
        update["answer"] = OUT_OF_SCOPE
    return update


def chat_node(state):
    history = format_history(state.get("history") or []) or "(no earlier messages)"
    prompt = CHAT.format(competitors=", ".join(COMPETITORS), history=history, question=state["question"])
    return {"answer": ask(prompt, "chat").strip(), "trace": ["chat: answered from the conversation"]}


def rag_inputs(state):
    parts = state["subquestions"]
    return [Send("rag", {"question": q, "index": i, "parts": len(parts), "retries": state.get("retries", 0)})
            for i, q in enumerate(parts)]


def route_after_plan(state):
    if state["intent"] == "out_of_scope":
        return END
    if state["intent"] == "chat":
        return "chat"
    return rag_inputs(state) + ([] if state["intent"] == "lookup" else ["web"])


def route_after_retry(state):
    return rag_inputs(state)


def rag_node(state, graph, wide_graph):
    result = (wide_graph if state["retries"] else graph).invoke({"question": state["question"]})
    entry = {"round": state["retries"], "index": state["index"], "question": state["question"], "docs": result["docs"],
             "answer": result["answer"], "quote": result["quote"], "relevant": result["relevant"], "attempts": result["attempts"]}
    label = f"rag[{state['index'] + 1}]" if state["parts"] > 1 else "rag"
    return {"rag_results": [entry], "trace": [f"{label}: {step}" for step in result["trace"]]}


def web_node(state):
    companies = list(dict.fromkeys(["Arla"] + state["companies"]))
    news = get_market_news(companies)
    stories = sum(s["stories"] for s in news["companies"].values())
    why = " (escalated from lookup)" if state["intent"] == "lookup" else ""
    return {"news": news, "trace": [f"web: {stories} stories from {news['origin']}{why}"]}


def shift(text, offset):
    return re.sub(r"\[(\d+)\]", lambda m: f"[{int(m[1]) + offset}]", text)


def merge_results(results):
    results = sorted(results, key=lambda r: r["index"])
    docs, answers, quotes, offset = [], [], [], 0
    for r in results:
        answers.append(f"**{r['question']}**\n{shift(r['answer'], offset)}" if len(results) > 1 else r["answer"])
        if r["quote"]:
            quotes.append(shift(r["quote"], offset))
        docs += r["docs"]
        offset += len(r["docs"])
    return {"docs": docs, "rag_answer": "\n\n".join(answers), "quote": " | ".join(quotes),
            "rag_relevant": all(r["relevant"] for r in results), "rag_attempts": max(r["attempts"] for r in results)}


def current_results(state):
    return [r for r in state["rag_results"] if r["round"] == state.get("retries", 0)]


def open_issue(state):
    return bool(OPEN_ISSUE.search(f"{state.get('quote', '')} {state.get('rag_answer', '')}"))


def gather_node(state, newest_year):
    merged = merge_results(current_results(state))
    notes = []
    top = (merged["docs"] or [None])[0]
    if top and top["doc_type"] in POLICY_TYPES and 0 < top["year"] < newest_year.get(top["doc_type"], 0):
        notes.append(f"The top source ('{top['title']}', {top['year']}) may be superseded "
                     f"by a {newest_year[top['doc_type']]} version.")
    flags = (", superseded warning" if notes else "") + (", open issue in evidence" if open_issue(merged) else "")
    return {**merged, "warnings": notes,
            "trace": [f"gather: {len(current_results(state))} rag result(s), news={'yes' if state.get('news') else 'no'}{flags}"]}


def policy_node(state, collection):
    k = 6 + 4 * state.get("retries", 0)
    policy = compare_policy(state["action"], policy_docs(collection, state["action"], k))
    return {"policy": policy, "trace": [f"policy: {policy['verdict']} ({len(policy['policies'])} cited)"]}


def risk_node(state):
    evidence = evidence_summary(state["policy"]["docs"], state["rag_relevant"], state["rag_attempts"])
    risk = score_risk(state["action"], state["policy"], evidence, state.get("news"))
    return {"risk": risk, "trace": [f"risk: {risk['level']} ({risk['total']})"]}


def synthesize_node(state):
    notes = "".join(f"> {w}\n" for w in state.get("warnings", []))
    evidence = f"{state['rag_answer']}{evidence_line(state['quote'])}"
    if state["intent"] == "lookup" and len(state["subquestions"]) == 1:
        answer = f"{notes}{evidence}\n\n**Sources**\n{sources_section(state['docs'])}"
        if state.get("news"):
            answer += "\n\n**Market now**\n" + market_section(state["news"])
        return {"answer": answer, "trace": ["synthesize: lookup"]}

    summary = ask(SUMMARY.format(facts=facts(state)), "summary").strip()
    parts = [f"{notes}## Summary\n{summary}"]
    if state["intent"] == "decision":
        parts += ["## Policy check\n" + policy_section(state["policy"]), "## Risk assessment\n" + risk_section(state["risk"])]
    parts.append("## Internal evidence\n" + evidence)
    if state.get("news"):
        parts.append("## Market now\n" + market_section(state["news"]))
    parts.append("## Sources\n" + sources_section(state["docs"]))
    return {"answer": "\n\n".join(parts), "trace": [f"synthesize: {state['intent']}"]}


def check_result(result):
    if not result["relevant"]:
        return ["no relevant evidence found"]
    notes, docs, text = [], result["docs"], result["answer"]
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    if not cited:
        notes.append("answer has no citations")
    elif cited[0] < 1 or cited[-1] > len(docs):
        notes.append("answer cites a missing source")
    cited = [n for n in cited if 1 <= n <= len(docs)]
    sources = [docs[n - 1]["text"] for n in cited] or [d["text"] for d in docs]

    body = re.sub(r"\[\d+\]", "", text)
    figures = set(NUMBER.findall(body)) - set(NUMBER.findall(result["question"]))
    missing = sorted(n for n in figures if n not in " ".join(sources))
    if missing:
        notes.append("figures not found in the cited sources: " + ", ".join(missing[:3]))

    if not USE_DUMMY:
        quote = re.sub(r"\[\d+\]", "", result["quote"]).strip(' "“”')
        if not quote:
            notes.append("answer has no supporting quote")
        elif max(fuzz.partial_ratio(norm(quote), norm(s)) for s in sources) < 85:
            notes.append("quote not found in the cited sources")
    return notes


def critic_node(state):
    results = current_results(state)
    notes = []
    for r in results:
        if state.get("news") and (not r["relevant"] or NOT_FOUND.search(r["answer"])):
            continue  # the documents had nothing to cite, so the market data carries the answer
        prefix = f"[{r['index'] + 1}] " if len(results) > 1 else ""
        notes += [prefix + n for n in check_result(r)]
    if state.get("risk", {}).get("needs_more_evidence"):
        notes.append("risk is high but the evidence is weak")
    update = {"critic_ok": not notes, "trace": [f"critic: {'passed' if not notes else '; '.join(notes)}"]}
    if notes and state.get("retries", 0) >= MAX_RETRIES:
        update["answer"] = state["answer"] + "\n\n> Warning: this answer could not be fully verified: " + "; ".join(notes)
    return update


def retry_node(state):
    return {"retries": state.get("retries", 0) + 1, "trace": ["retry: broader retrieval"]}


def route_after_gather(state):
    if state["intent"] == "lookup" and open_issue(state) and not state.get("news"):
        return "web"
    return "policy" if state["intent"] == "decision" else "synthesize"
