import argparse
import sys

from rag.graph import build_rag_graph

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--strategy", default="parent", choices=["recursive", "parent"])
args = parser.parse_args()

result = build_rag_graph(args.strategy).invoke({"question": args.question})
print("\n".join(result["trace"]))
for i, d in enumerate(result["docs"], 1):
    print(f"[{i}] {d['source']} p.{d['page']} ({d['doc_type']}, {d['year']}) dist={d['distance']:.3f}")
print(f"\n{result['answer']}")
print(f"quote: {result['quote']}")
