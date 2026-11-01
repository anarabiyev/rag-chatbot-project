import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from agent.graph import build_agent_graph
from rag.graph import norm
from rag.llm import STATS
from rag.loader import read_csv

sys.stdout.reconfigure(encoding="utf-8")

CASES = Path("eval/agent_cases.csv")
RESULTS = Path("eval/agent_results.csv")
FIELDS = ["question", "intent", "got_intent", "intent_ok", "answer_ok", "verdict", "got_verdict", "verdict_ok",
          "verified", "seconds", "llm_calls", "answer"]


def contains(answer, spec):
    text = norm(answer)
    return all(any(alt.strip().lower() in text for alt in part.split("|")) for part in spec.split("&"))


def score(case, got_intent, answer, got_verdict):
    return {"intent": case["intent"], "intent_ok": got_intent == case["intent"], "answer_ok": contains(answer, case["keyword"]),
            "verdict": case["verdict"], "verdict_ok": got_verdict == case["verdict"]}


def summary(rows):
    flag = lambda row, key: str(row[key]) == "True"
    print(f"\n{len(rows)} cases")
    print(f"routing accuracy:    {mean(flag(r, 'intent_ok') for r in rows):.2f}")
    print(f"answer correctness:  {mean(flag(r, 'answer_ok') for r in rows):.2f}")
    decisions = [r for r in rows if r["verdict"]]
    if decisions:
        print(f"policy verdict:      {mean(flag(r, 'verdict_ok') for r in decisions):.2f} ({len(decisions)} decisions)")
    print(f"critic verified:     {mean(flag(r, 'verified') for r in rows):.2f}")
    print(f"\n{'intent':10} {'n':>3} {'correct':>8} {'mean s':>8} {'LLM calls':>10}")
    by_intent = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)
    for intent, group in by_intent.items():
        print(f"{intent:10} {len(group):3} {mean(flag(r, 'answer_ok') for r in group):8.2f} "
              f"{mean(float(r['seconds']) for r in group):8.1f} {mean(int(r['llm_calls']) for r in group):10.1f}")
    for r in rows:
        if not flag(r, "verified"):
            note = r["answer"].split("could not be fully verified:")[-1].strip()
            print(f"\nUNVERIFIED {r['question'][:80]}\n  critic: {note[:300]}")
    for r in rows:
        if not (flag(r, "intent_ok") and flag(r, "answer_ok") and (not r["verdict"] or flag(r, "verdict_ok"))):
            print(f"\nMISS {r['question'][:80]}\n  intent {r['got_intent']} (expected {r['intent']}), verdict {r['got_verdict'] or '-'}"
                  f"\n  answer: {' '.join(r['answer'].split())[:260]}")


parser = argparse.ArgumentParser()
parser.add_argument("--only", nargs="+", metavar="TEXT", help="run only the cases whose question contains one of these texts")
args = parser.parse_args()

cases = read_csv(CASES)
if args.only:
    cases = [c for c in cases if any(t.lower() in c["question"].lower() for t in args.only)]
    print(f"--only: {len(cases)} case(s) selected")
done = {r["question"] for r in read_csv(RESULTS)} if RESULTS.exists() else set()
graph = build_agent_graph()

failed = 0
with open(RESULTS, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, FIELDS)
    if not done:
        writer.writeheader()
    for case in cases:
        if case["question"] in done:
            continue
        calls_before = sum(v["calls"] for v in STATS.values())
        start = time.perf_counter()
        try:
            result = graph.invoke({"question": case["question"]})
        except Exception as exc:
            failed += 1
            print(f"FAIL {case['question'][:75]}\n     {str(exc)[:300]}")
            continue
        seconds = time.perf_counter() - start
        got_verdict = result.get("policy", {}).get("verdict", "")
        row = {
            "question": case["question"], "got_intent": result["intent"], "got_verdict": got_verdict,
            **score(case, result["intent"], result["answer"], got_verdict),
            "verified": "could not be fully verified" not in result["answer"], "seconds": round(seconds, 1),
            "llm_calls": sum(v["calls"] for v in STATS.values()) - calls_before, "answer": result["answer"],
        }
        writer.writerow(row)
        f.flush()
        print(f"{'ok  ' if row['answer_ok'] and row['intent_ok'] else 'MISS'} {seconds:6.1f}s {case['question'][:75]}")

by_question = {c["question"]: c for c in cases}
summary([{**r, **score(by_question[r["question"]], r["got_intent"], r["answer"], r["got_verdict"])}
         for r in read_csv(RESULTS) if r["question"] in by_question])
if failed:
    print(f"\n{failed} case(s) failed with an error and were not saved. Run the script again to retry them.")
