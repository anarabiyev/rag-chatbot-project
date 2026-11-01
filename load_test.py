"""Load test for the agent.

    python load_test.py a    system overhead: 100 queries, dummy LLM, concurrency 1 and 8
    python load_test.py b    real model: a sample of 12 queries, stops after --budget-min minutes
    python load_test.py c    extrapolate 50 / 100 / 200 real-model queries from the results of run b

Results go to eval/load/ (one CSV per run, plus a text summary).
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("run", choices=["a", "b", "c"])
parser.add_argument("--queries", type=int, default=100, help="run a: number of queries per pass")
parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8], help="run a: concurrency of each pass")
parser.add_argument("--sample", type=int, default=12, help="run b: number of queries")
parser.add_argument("--budget-min", type=float, default=55, help="run b: no new query is started after this many minutes")
args = parser.parse_args()

if args.run == "a":
    os.environ["LLM_BACKEND"] = "dummy"  # must be set before rag.llm is imported

from agent import nodes  # noqa: E402
from agent.graph import build_agent_graph  # noqa: E402
from rag import llm  # noqa: E402
from rag.loader import read_csv  # noqa: E402

CASES = Path("eval/agent_cases.csv")
OUT = Path("eval/load")
ROUTES = ["lookup", "market", "decision", "out_of_scope"]
FIELDS = ["run", "question", "expected_intent", "got_intent", "seconds", "retries", "llm_calls", "llm_seconds",
          "llm_tasks", "nodes", "error"]


def pct(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1)]


def llm_totals():
    return {task: dict(v) for task, v in llm.STATS.items()}


def llm_diff(before):
    out = {}
    for task, v in llm.STATS.items():
        d = {k: v[k] - before.get(task, {}).get(k, 0) for k in ("calls", "seconds", "in", "out")}
        if d["calls"]:
            out[task] = {k: round(x, 2) for k, x in d.items()}
    return out


def run_one(graph, case, run, track_llm):
    before = llm_totals() if track_llm else None
    start = time.perf_counter()
    row = {"run": run, "question": case["question"], "expected_intent": case["intent"], "error": ""}
    try:
        result = graph.invoke({"question": case["question"]})
    except Exception as exc:
        row.update(seconds=round(time.perf_counter() - start, 3), error=str(exc)[:200])
        return row
    nodes = defaultdict(float)
    for name, seconds in result["timings"]:
        nodes[name] += seconds
    row.update(got_intent=result["intent"], seconds=round(time.perf_counter() - start, 3),
               retries=result.get("retries", 0), nodes=json.dumps({k: round(v, 3) for k, v in nodes.items()}))
    if track_llm:
        tasks = llm_diff(before)
        row.update(llm_tasks=json.dumps(tasks), llm_calls=sum(t["calls"] for t in tasks.values()),
                   llm_seconds=round(sum(t["seconds"] for t in tasks.values()), 2))
    return row


def summarize(title, rows, wall, out):
    def say(line=""):
        print(line)
        out.append(line)

    ok = [r for r in rows if not r["error"]]
    say(f"\n=== {title} ===")
    say(f"{len(rows)} queries, {len(rows) - len(ok)} errors, wall time {wall:.1f}s, "
        f"throughput {60 * len(rows) / wall:.1f} queries/min")
    say(f"\n{'route':13} {'n':>4} {'mean s':>8} {'p50 s':>8} {'p95 s':>8} {'max s':>8} {'retry %':>8} {'routed ok':>10}")
    for route in ROUTES + ["all"]:
        group = [r for r in ok if route == "all" or r["expected_intent"] == route]
        if not group:
            continue
        s = [float(r["seconds"]) for r in group]
        retried = sum(int(r["retries"]) > 0 for r in group) / len(group)
        routed = sum(r["got_intent"] == r["expected_intent"] for r in group) / len(group)
        say(f"{route:13} {len(group):4} {sum(s) / len(s):8.2f} {pct(s, 50):8.2f} {pct(s, 95):8.2f} {max(s):8.2f} "
            f"{retried:8.0%} {routed:10.0%}")

    node_total = defaultdict(float)
    for r in ok:
        for name, seconds in json.loads(r["nodes"]).items():
            node_total[name] += seconds
    all_nodes = sum(node_total.values()) or 1
    say("\nShare of node time (summed over queries, parallel branches overlap):")
    for name, seconds in sorted(node_total.items(), key=lambda kv: -kv[1]):
        say(f"  {name:11} {seconds:9.1f}s {seconds / all_nodes:6.1%}")

    tasks = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "in": 0, "out": 0})
    for r in ok:
        for task, v in json.loads(r.get("llm_tasks") or "{}").items():
            for key in tasks[task]:
                tasks[task][key] += v[key]
    if tasks:
        llm_seconds = sum(v["seconds"] for v in tasks.values())
        say(f"\nLLM calls: {sum(v['calls'] for v in tasks.values())}, {llm_seconds:.1f}s = "
            f"{llm_seconds / sum(float(r['seconds']) for r in ok):.1%} of total query time")
        say(f"  {'task':9} {'calls':>6} {'seconds':>9} {'tokens in':>10} {'tokens out':>11} {'s/call':>7} {'out tok/s':>10}")
        for task, v in sorted(tasks.items(), key=lambda kv: -kv[1]["seconds"]):
            say(f"  {task:9} {v['calls']:6} {v['seconds']:9.1f} {v['in']:10} {v['out']:11} "
                f"{v['seconds'] / v['calls']:7.1f} {v['out'] / v['seconds'] if v['seconds'] else 0:10.1f}")
    for r in rows:
        if r["error"]:
            say(f"ERROR {r['question'][:70]}: {r['error']}")


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def warm_up(cases):
    start = time.perf_counter()
    graph = build_agent_graph()
    built = time.perf_counter() - start
    run_one(graph, cases[0], "warmup", track_llm=False)
    print(f"cold start: graph built in {built:.1f}s, first query done after {time.perf_counter() - start:.1f}s in total")
    llm.STATS.clear()
    return graph, built


def run_a(cases):
    labels = {c["question"]: c["intent"] for c in cases}
    # the dummy planner always says "lookup"; give it the labelled intent so every route is exercised
    llm.DUMMY_REPLIES["plan"] = lambda p: json.dumps({"intent": labels.get(p.split("Question:")[-1].strip(), "lookup"), "questions": []})
    nodes.MAX_RETRIES = 0  # canned answers never pass the critic, so every query would run twice
    graph, built = warm_up(cases)
    pool = (cases * math.ceil(args.queries / len(cases)))[:args.queries]
    random.Random(0).shuffle(pool)
    out = [f"Run A: dummy LLM, {args.queries} queries from eval/agent_cases.csv, planner stubbed with the labelled intent, critic retry disabled",
           f"cold start: graph built in {built:.1f}s"]
    for workers in args.concurrency:
        start = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool_:
            rows = list(pool_.map(lambda c: run_one(graph, c, f"a_c{workers}", track_llm=workers == 1), pool))
        wall = time.perf_counter() - start
        write_rows(OUT / f"a_c{workers}.csv", rows)
        summarize(f"A: dummy LLM, concurrency {workers}", rows, wall, out)
    (OUT / "a_summary.txt").write_text("\n".join(out), encoding="utf-8")


def sample_cases(cases, size):
    """Evenly spaced cases per route, interleaved, so stopping early still leaves a mix of routes."""
    shares = {r: [c for c in cases if c["intent"] == r] for r in ROUTES}
    take = {"lookup": 4, "market": 3, "decision": 3, "out_of_scope": 2}
    picked = {r: [shares[r][i * len(shares[r]) // take[r]] for i in range(take[r])] for r in ROUTES}
    order = [c for i in range(max(take.values())) for r in ROUTES for c in picked[r][i:i + 1]]
    return order[:size]


def run_b(cases):
    import requests

    try:
        requests.get(llm.BASE_URL + "/models", timeout=5).raise_for_status()
    except Exception as exc:
        sys.exit(f"LM Studio is not reachable at {llm.BASE_URL}: {exc}")
    path = OUT / "b.csv"
    rows = read_csv(path) if path.exists() else []
    done = {r["question"] for r in rows}
    queue = [c for c in sample_cases(cases, args.sample) if c["question"] not in done]
    print(f"{len(rows)} already done, {len(queue)} to run, budget {args.budget_min:.0f} min")

    graph, built = warm_up(cases)
    ask_start = time.perf_counter()
    llm.ask("Reply with OK.", "grade")  # makes sure the model is loaded before timing
    print(f"model warm-up call: {time.perf_counter() - ask_start:.1f}s")
    llm.STATS.clear()

    start = time.perf_counter()
    for case in queue:
        if time.perf_counter() - start > args.budget_min * 60:
            print(f"budget of {args.budget_min:.0f} min reached, stopping")
            break
        row = run_one(graph, case, "b", track_llm=True)
        rows.append(row)
        write_rows(path, rows)
        print(f"{row['expected_intent']:13} {row['seconds']:7.1f}s retries={row.get('retries', '-')} "
              f"llm_calls={row.get('llm_calls', '-')} {case['question'][:60]} {row['error']}", flush=True)
    wall = time.perf_counter() - start
    out = [f"Run B: real model {llm.MODEL}, {len(rows)} queries, sequential",
           f"cold start: graph built in {built:.1f}s"]
    summarize("B: real model, concurrency 1", rows, wall, out)
    (OUT / "b_summary.txt").write_text("\n".join(out), encoding="utf-8")


def run_c(cases):
    rows = [r for r in read_csv(OUT / "b.csv") if not r["error"]]
    mix = {r: sum(c["intent"] == r for c in cases) / len(cases) for r in ROUTES}
    stats = {}
    for route in ROUTES:
        s = [float(r["seconds"]) for r in rows if r["expected_intent"] == route]
        if s:
            stats[route] = (len(s), sum(s) / len(s), max(s))
    missing = [r for r in ROUTES if r not in stats and mix[r]]
    out = ["Run C: extrapolation from run B (sequential, one local model)",
           f"query mix = mix of eval/agent_cases.csv: " + ", ".join(f"{r} {mix[r]:.0%}" for r in ROUTES),
           f"\n{'route':13} {'n in B':>7} {'mean s':>8} {'max s':>8}"]
    out += [f"{r:13} {n:7} {m:8.1f} {mx:8.1f}" for r, (n, m, mx) in stats.items()]
    if missing:
        out.append(f"no measurements for {', '.join(missing)}; their share is excluded and the mix is renormalised")
    share = sum(mix[r] for r in stats) or 1
    mean_s = sum(mix[r] / share * stats[r][1] for r in stats)
    slow_s = sum(mix[r] / share * stats[r][2] for r in stats)
    out.append(f"\nexpected seconds per query: {mean_s:.1f} (mean per route), {slow_s:.1f} (slowest observed per route)")
    out.append(f"\n{'queries':>8} {'estimate (mean)':>18} {'upper bound (slowest)':>24}")
    for n in (50, 100, 200):
        out.append(f"{n:8} {n * mean_s / 3600:15.1f} h {n * slow_s / 3600:21.1f} h")
    out.append("\nThe estimate assumes queries run one after another against one local model. "
               "Run B has only a few queries per route, so treat it as an order of magnitude.")
    print("\n".join(out))
    (OUT / "c_summary.txt").write_text("\n".join(out), encoding="utf-8")


OUT.mkdir(parents=True, exist_ok=True)
cases = read_csv(CASES)
{"a": run_a, "b": run_b, "c": run_c}[args.run](cases)
