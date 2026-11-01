import time

from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.state import AgentState
from rag.graph import build_rag_graph
from rag.loader import read_csv
from rag.paths import MANIFEST
from rag.store import get_collection


def newest_years(manifest=MANIFEST):
    newest = {}
    for row in read_csv(manifest):
        newest[row["doc_type"]] = max(newest.get(row["doc_type"], 0), int(row["year"]))
    return newest


def timed(name, fn):
    def run(state):
        start = time.perf_counter()
        update = fn(state)
        return {**update, "timings": [(name, round(time.perf_counter() - start, 3))]}
    return run


def route_after_critic(state):
    return "retry" if not state["critic_ok"] and state.get("retries", 0) < nodes.MAX_RETRIES else END


def build_agent_graph(strategy="parent"):
    rag_graph, wide_graph = build_rag_graph(strategy), build_rag_graph(strategy, k=8)
    collection = get_collection(f"arla_{strategy}")
    newest = newest_years()

    graph = StateGraph(AgentState)
    graph.add_node("planner", timed("planner", nodes.planner_node))
    graph.add_node("rag", timed("rag", lambda state: nodes.rag_node(state, rag_graph, wide_graph)))
    graph.add_node("chat", timed("chat", nodes.chat_node))
    graph.add_node("web", timed("web", nodes.web_node))
    graph.add_node("gather", timed("gather", lambda state: nodes.gather_node(state, newest)))
    graph.add_node("policy", timed("policy", lambda state: nodes.policy_node(state, collection)))
    graph.add_node("risk", timed("risk", nodes.risk_node))
    graph.add_node("synthesize", timed("synthesize", nodes.synthesize_node))
    graph.add_node("critic", timed("critic", nodes.critic_node))
    graph.add_node("retry", timed("retry", nodes.retry_node))

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", nodes.route_after_plan, ["rag", "web", "chat", END])
    graph.add_edge("chat", END)
    graph.add_edge("rag", "gather")
    graph.add_edge("web", "gather")
    graph.add_conditional_edges("gather", nodes.route_after_gather, ["web", "policy", "synthesize"])
    graph.add_edge("policy", "risk")
    graph.add_edge("risk", "synthesize")
    graph.add_edge("synthesize", "critic")
    graph.add_conditional_edges("critic", route_after_critic, ["retry", END])
    graph.add_conditional_edges("retry", nodes.route_after_retry, ["rag"])
    return graph.compile()
