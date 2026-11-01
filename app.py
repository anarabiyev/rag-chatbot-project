import time
from collections import defaultdict

import pandas as pd
import streamlit as st

from agent.memory import recent_turns
from rag.llm import BASE_URL, MODEL, STATS

st.set_page_config(page_title="Arla decision support", layout="wide")


EXAMPLES = {
    "Lookup": "What was Arla's revenue in 2024?",
    "Compare": "Compare Arla's revenue in 2023 and 2024.",
    "Market": "What is the latest news about Arla?",
    "Decision": "Should we accept a EUR 300 gift voucher from a supplier during an ongoing tender?",
}


@st.cache_resource(show_spinner="Loading models and indexes...")
def load_graph():
    from agent.graph import build_agent_graph

    return build_agent_graph()


def run_agent(question, history):
    graph = load_graph()
    before = {task: dict(v) for task, v in STATS.items()}
    start = time.perf_counter()
    state = {}
    with st.status("Running the agent...", expanded=True) as status:
        for mode, chunk in graph.stream({"question": question, "history": history}, stream_mode=["updates", "values"]):
            if mode == "values":
                state = chunk
                continue
            for update in chunk.values():
                for line in (update or {}).get("trace", []):
                    status.write(line)
        status.update(label=f"Done in {time.perf_counter() - start:.0f}s", state="complete", expanded=False)
    usage = {task: {key: v[key] - before.get(task, {}).get(key, 0) for key in ("calls", "seconds", "in", "out")}
             for task, v in STATS.items()}
    return {"state": state, "seconds": time.perf_counter() - start, "usage": usage}


def show_rag(state):
    results = [r for r in state["rag_results"] if r["round"] == state.get("retries", 0)]
    for r in sorted(results, key=lambda r: r["index"]):
        st.markdown(f"**Question:** {r['question']}")
        st.markdown(r["answer"])
        if r["quote"]:
            st.caption(f"Quote: {r['quote']}")
    st.subheader("Retrieved chunks")
    for i, d in enumerate(state["docs"], 1):
        with st.expander(f"[{i}] {d['title']}, p.{d['page']} ({d['year']}) | distance {d['distance']:.2f}"):
            st.caption("Matched passage")
            st.text(d.get("match", ""))
            st.caption("Parent chunk used as context")
            st.text(d["text"])


def show_news(news):
    st.caption(f"{news['origin']} | fetched {news['fetched_at'][:10]} | last {news['days']} days")
    table = pd.DataFrame([
        {"company": name, "stories": s["stories"], "avg sentiment": s["avg_sentiment"],
         "negative share": s["negative_share"], "event stories": s["events"]}
        for name, s in news["companies"].items()
    ])
    st.dataframe(table, hide_index=True)
    arla = news["companies"].get("Arla")
    if arla:
        st.markdown("**Arla headlines**")
        seen = set()
        for story in arla["latest"] + arla["most_negative"]:
            if story["title"] not in seen:
                seen.add(story["title"])
                flags = f" | {', '.join(story['events'])}" if story["events"] else ""
                st.markdown(f"- {story['published'][:10]} | [{story['title']}]({story['url']}) | sentiment {story['sentiment']:+.2f}{flags}")


def show_policy_risk(state):
    policy, risk = state["policy"], state["risk"]
    verdict, level, total = st.columns(3)
    verdict.metric("Policy verdict", policy["verdict"].upper())
    level.metric("Risk level", risk["level"])
    total.metric("Risk score", f"{risk['total']}/100")
    st.caption(policy["reason"])
    st.markdown("**Score per dimension**")
    st.bar_chart(pd.Series(risk["scores"], name="score"))
    st.markdown("**Drivers**")
    for d in risk["drivers"]:
        st.markdown(f"- {d['text']}")
    for m in risk["mitigations"]:
        st.markdown(f"- Mitigation: {m}")
    for cited in policy["policies"]:
        for d in policy["docs"]:
            if d["source"] == cited["source"] and d["page"] == cited["page"]:
                with st.expander(f"Cited: {cited['title']}, p.{cited['page']}"):
                    st.text(d["text"])
                break


def show_trace(result):
    state = result["state"]
    steps = [line.split(": ", 1) if ": " in line else [line, ""] for line in state["trace"]]
    st.dataframe(pd.DataFrame(steps, columns=["node", "detail"]), hide_index=True)
    seconds = defaultdict(float)
    for node, value in state["timings"]:
        seconds[node] += value
    st.markdown("**Seconds per node**")
    st.bar_chart(pd.Series(seconds, name="seconds"))
    st.markdown("**LLM usage for this question**")
    rows = [{"task": t, "calls": v["calls"], "seconds": round(v["seconds"], 1), "tokens in": v["in"], "tokens out": v["out"]}
            for t, v in result["usage"].items() if v["calls"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def show_metrics(result):
    state = result["state"]
    calls = sum(v["calls"] for v in result["usage"].values())
    verified = "could not be fully verified" not in state["answer"]
    intent, seconds, llm, critic = st.columns(4)
    intent.metric("Intent", state["intent"])
    seconds.metric("Time", f"{result['seconds']:.0f}s")
    llm.metric("LLM calls", calls)
    critic.metric("Critic", "verified" if verified else "warning")


def show_details(result):
    state = result["state"]
    names = ["Agent route"] + (["RAG output"] if state.get("rag_results") else []) + (["Market news"] if state.get("news") else []) + (["Policy and risk"] if state.get("policy") else [])
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["Agent route"]:
        show_trace(result)
    if "RAG output" in tabs:
        with tabs["RAG output"]:
            show_rag(state)
    if "Market news" in tabs:
        with tabs["Market news"]:
            show_news(state["news"])
    if "Policy and risk" in tabs:
        with tabs["Policy and risk"]:
            show_policy_risk(state)


with st.sidebar:
    st.header("Setup")
    st.write(f"LLM: `{MODEL}`")
    st.caption(f"served by {BASE_URL}")
    st.write("Embeddings: `bge-small-en-v1.5`")
    st.write("Index: parent-child chunks (Chroma)")

LLM_FILL, CODE_FILL = "#bfdbfe", "#e5e7eb"
# (node id, label, uses the LLM)
GRAPH_NODES = [
    ("question", "question", False),
    ("planner", "Planner\nintent, sub-questions, follow-up rewrite (1 LLM call)", True),
    ("fixed", "Fixed reply\nout_of_scope, ends here", False),
    ("news", "News pipeline\nFinBERT + event flags, no LLM", False),
    ("gather", "Gather\nmerge, renumber [n], warnings", False),
    ("policy", "Policy comparator\nretrieve policy docs + 1 LLM verdict", True),
    ("risk", "Risk scorer\nrules; reads news + RAG evidence", False),
    ("synth", "Synthesize\ncode builds sections, 1 LLM summary", True),
    ("critic", "Critic\nrules: citations, figures, quote", False),
    ("retry", "Retry\nk 5 to 8, wider policy search", False),
    ("end", "END", False),
]
RAG_NODES = [
    ("rewrite", "rewrite\nLLM", True),
    ("retrieve", "retrieve\nvector search", False),
    ("grade", "grade\ndistance rules", False),
    ("generate", "generate\nLLM + quote", True),
]
# (from, to, label, loops back)
GRAPH_EDGES = [
    ("question", "planner", "", False),
    ("planner", "fixed", "out_of_scope", False),
    ("planner", "rewrite", "lookup / market / decision", False),
    ("planner", "news", "market / decision", False),
    ("rewrite", "retrieve", "", False),
    ("retrieve", "grade", "", False),
    ("grade", "generate", "relevant", False),
    ("grade", "rewrite", "not relevant: retry once", True),
    ("generate", "gather", "", False),
    ("news", "gather", "", False),
    ("gather", "news", "open issue in a lookup: add news", True),
    ("gather", "policy", "decision", False),
    ("policy", "risk", "", False),
    ("risk", "synth", "", False),
    ("gather", "synth", "lookup / market", False),
    ("synth", "critic", "", False),
    ("critic", "end", "passed, or retry used", False),
    ("critic", "retry", "failed, retry left", False),
    ("retry", "rewrite", "re-run RAG with 8 chunks", True),
]


def agent_graph_dot():
    def node(key, label, llm):
        label = label.replace("\n", "\\n")  # DOT line break
        return f'{key} [label="{label}", fillcolor="{LLM_FILL if llm else CODE_FILL}"];'

    lines = [
        "digraph agent {",
        'rankdir=TB; nodesep=0.5; ranksep=0.55; bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, fontcolor="#1f2328", color="#6b7280", margin="0.15,0.08"];',
        'edge [color="#6b7280", fontcolor="#9ca3af", fontname="Helvetica", fontsize=10];',
        *[node(*n) for n in GRAPH_NODES],
        'subgraph cluster_rag { label="RAG subgraph, one per sub-question (run in parallel)"; labeljust=l;'
        ' fontname="Helvetica"; fontsize=12; fontcolor="#9ca3af"; color="#3b82f6"; style="dashed,rounded";',
        *[node(*n) for n in RAG_NODES],
        "}",
        *[f'{a} -> {b} [label="{label}"{", style=dashed" if loop else ""}];' for a, b, label, loop in GRAPH_EDGES],
        "}",
    ]
    return "\n".join(lines)


def show_diagram():
    st.graphviz_chart(agent_graph_dot(), width="stretch")
    st.caption("Blue boxes use the LLM, grey boxes are rules or code only, dashed arrows loop back.")


def show_info():
    st.markdown(
        "A new employee at Arla Foods needs to know how the company acted before, what is happening in the market, "
        "and whether a proposed decision fits company policy. This agent answers all three, with sources.")
    st.subheader("What you can ask")
    st.dataframe(pd.DataFrame([
        {"intent": "lookup", "example": EXAMPLES["Lookup"], "what runs": "RAG over 22 Arla documents, critic"},
        {"intent": "market", "example": EXAMPLES["Market"], "what runs": "RAG and news pipeline in parallel, critic"},
        {"intent": "decision", "example": EXAMPLES["Decision"], "what runs": "RAG and news, then policy comparator and risk scorer, critic"},
        {"intent": "out_of_scope", "example": "Who won the football match yesterday?", "what runs": "Fixed reply, nothing else"},
    ]), hide_index=True)
    st.subheader("How a question moves through the agent")
    show_diagram()
    st.subheader("Building blocks")
    st.markdown(
        "- **RAG subgraph:** rewrite the question, retrieve parent-child chunks (with a year filter), grade by distance, generate a cited answer with a verbatim quote.\n"
        "- **News pipeline:** Google News headlines for Arla and three competitors, FinBERT sentiment and event flags. No LLM.\n"
        "- **Policy comparator:** checks a proposed action against Arla's policies. One LLM call picks `consistent`, `conflict` or `no_precedent`, and rules validate it.\n"
        "- **Risk scorer:** rule-based score from policy, regulatory, external and evidence dimensions.\n"
        "- **Critic:** rule-based check of citations, figures and quote. It can trigger one retry.\n\n"
        "Everything runs locally: an open-source LLM in LM Studio, local embeddings, Chroma and FinBERT.")


def show_message(message, detailed):
    """Only the newest answer shows the metrics and detail tabs, so a long chat stays short."""
    with st.chat_message(message["role"]):
        result = message.get("result")
        if result and detailed:
            show_metrics(result)
        st.markdown(message["content"])
        if result and detailed:
            show_details(result)
        elif result:
            st.caption(f"{result['state']['intent']} | {result['seconds']:.0f}s")


def show_ask():
    messages = st.session_state.setdefault("messages", [])
    st.caption("Examples")
    for column, (label, example) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
        column.button(label, help=example, width="stretch", on_click=lambda q=example: st.session_state.update(pending=q))
    if messages:
        turns = len(recent_turns(messages))
        hint, reset = st.columns([5, 1], vertical_alignment="center")
        hint.info(f"Conversation in progress, the agent remembers the last {turns} "
                  f"question{'s' if turns != 1 else ''}. Ask a follow-up in the box below.",
                  icon=":material/forum:")
        reset.button("New chat", icon=":material/add_comment:", width="stretch",
                     on_click=lambda: st.session_state.update(messages=[]), help="Forget the conversation so far")

    last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    for i, message in enumerate(messages):
        show_message(message, i == last)

    typed = st.chat_input("Ask a question, or a follow-up to the last answer")
    question = (st.session_state.pop("pending", None) or typed or "").strip()
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = run_agent(question, recent_turns(messages))
        except Exception as exc:
            st.error(f"The agent failed: {exc}\n\nIs the LM Studio server running with the model loaded?")
            return
    messages += [{"role": "user", "content": question},
                 {"role": "assistant", "content": result["state"]["answer"], "result": result}]
    st.rerun()


st.title("Arla decision-support agent")
st.caption("Ask what Arla's documents say, what is happening in the market, or whether a proposed action is a good idea. "
           "The agent remembers the conversation, so you can ask follow-ups.")
info_tab, ask_tab = st.tabs(["Info", "Ask a question"])
with info_tab:
    show_info()
with ask_tab:
    show_ask()
