import sys
import time
from collections import defaultdict

from agent.graph import build_agent_graph
from rag.llm import STATS

sys.stdout.reconfigure(encoding="utf-8")

question = " ".join(sys.argv[1:]) or "Should we accept a EUR 300 gift voucher from a supplier during an ongoing tender?"
graph = build_agent_graph()

start = time.perf_counter()
result = graph.invoke({"question": question})
elapsed = time.perf_counter() - start

print("\n".join(result["trace"]))
nodes = defaultdict(float)
for name, seconds in result["timings"]:
    nodes[name] += seconds
print(f"\nintent: {result['intent']} | total {elapsed:.1f}s")
print("nodes:  " + ", ".join(f"{n} {s:.1f}s" for n, s in nodes.items()))
print("llm:    " + ", ".join(f"{t} {v['calls']}x {v['seconds']:.1f}s ({v['in']} in/{v['out']} out tokens)" for t, v in STATS.items()))
print(f"\n{result['answer']}")
