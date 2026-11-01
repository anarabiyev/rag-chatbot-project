import sys

from rag.loader import read_csv
from rag.store import get_collection
from tools.policy import compare_policy, policy_docs

sys.stdout.reconfigure(encoding="utf-8")

cases = read_csv("eval/policy_cases.csv")

collection = get_collection("arla_parent")
correct = source_ok = with_source = 0
for case in cases:
    result = compare_policy(case["action"], policy_docs(collection, case["action"]))
    cited = {p["source"] for p in result["policies"]}
    ok = result["verdict"] == case["expected"]
    correct += ok
    if case["source"]:
        with_source += 1
        source_ok += case["source"] in cited
    print(f"{'ok  ' if ok else 'MISS'} expected={case['expected']:12} got={result['verdict']:12} {case['action'][:60]}")
    if not ok or case["source"] and case["source"] not in cited:
        print(f"       cited: {[(p['doc_id'], p['page']) for p in result['policies']]} | {result['reason'][:100]}")

print(f"\nverdict accuracy: {correct}/{len(cases)} | expected source cited: {source_ok}/{with_source}")
