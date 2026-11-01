import json
import re
from pathlib import Path
from statistics import mean

CONFIG = json.loads((Path(__file__).resolve().parent.parent / "config" / "risk.json").read_text(encoding="utf-8"))
LEVELS = ["LOW", "MEDIUM", "HIGH"]


def level_for(total, cfg=CONFIG):
    if total >= cfg["levels"]["high"]:
        return "HIGH"
    return "MEDIUM" if total >= cfg["levels"]["medium"] else "LOW"


def evidence_summary(docs, relevant=True, attempts=1):
    distances = [d["distance"] for d in docs]
    return {"best_distance": min(distances) if distances else None, "n_chunks": len(docs),
            "relevant": relevant, "attempts": attempts}


def regulatory_score(action, policy, cfg):
    text = action.lower()
    types = {p["doc_type"] for p in policy["policies"]}
    areas = {name: rule["weight"] for name, rule in cfg["regulatory_areas"].items()
             if any(k in text for k in rule["keywords"]) or types & set(rule["doc_types"])}
    return min(100, sum(areas.values())), list(areas)


def evidence_score(evidence, cfg):
    rules, penalty = cfg["evidence"], cfg["evidence"]["penalties"]
    found = []
    if evidence["best_distance"] is None or evidence["best_distance"] > rules["weak_distance"]:
        found.append(("weak match", penalty["distance"]))
    if evidence["n_chunks"] < rules["min_chunks"]:
        found.append(("few passages", penalty["few_chunks"]))
    if not evidence["relevant"]:
        found.append(("graded not relevant", penalty["not_relevant"]))
    if evidence["attempts"] > 1:
        found.append(("needed a retry", penalty["retried"]))
    return min(100, sum(p for _, p in found)), [name for name, _ in found]


def words(text):
    return set(re.findall(r"[a-z]{5,}", text.lower()))


def event_is_relevant(story, action, areas, cfg):
    event_areas = cfg["external"]["event_areas"]
    if set(areas) & {a for event in story["events"] for a in event_areas.get(event, [])}:
        return True
    return len(words(story["title"]) & words(action)) >= 2


def external_score(news, action, areas, cfg):
    rules = cfg["external"]
    if not news:
        return rules["missing_score"], "news unavailable"
    companies = news["companies"]
    arla = companies.get("Arla") or {}
    arla_negative = arla.get("negative_share") or 0
    rivals = [c["negative_share"] for name, c in companies.items() if name != "Arla" and c["negative_share"] is not None]
    negative = rules["arla_share"] * arla_negative + (1 - rules["arla_share"]) * (mean(rivals) if rivals else arla_negative)
    if "stories" in news:
        events = sum(1 for s in news["stories"]
                     if s["company"] == "Arla" and s["events"] and event_is_relevant(s, action, areas, cfg))
    else:
        events = arla.get("events", 0)
    score = 100 * (rules["negative_weight"] * negative + rules["event_weight"] * min(1, events / rules["events_cap"]))
    return score, f"{arla_negative:.0%} of Arla headlines negative, {events} event stories relevant to this action"


def score_risk(action, policy, evidence, news=None, cfg=CONFIG):
    regulatory, areas = regulatory_score(action, policy, cfg)
    evidence_points, weak = evidence_score(evidence, cfg)
    external, external_note = external_score(news, action, areas, cfg)
    scores = {"policy": cfg["policy_scores"][policy["verdict"]], "regulatory": regulatory,
              "external": external, "evidence": evidence_points}
    total = sum(cfg["weights"][k] * v for k, v in scores.items())

    level = level_for(total, cfg)
    floor = cfg["hard_floor"]
    if policy["verdict"] == "conflict" and {p["doc_type"] for p in policy["policies"]} & set(floor["doc_types"]):
        level = max(level, floor["level"], key=LEVELS.index)

    cited = ", ".join(f"{p.get('title', p['doc_id'])} p.{p['page']}" for p in policy["policies"])
    text = {
        "policy": f"Policy verdict: {policy['verdict']}" + (f" ({cited})" if cited else ""),
        "regulatory": "Regulatory exposure: " + ", ".join(areas),
        "external": f"Market signals: {external_note}",
        "evidence": "Weak evidence: " + ", ".join(weak),
    }
    mitigation = {"policy": f"policy_{policy['verdict']}", "regulatory": "regulatory",
                  "external": "external", "evidence": "evidence"}
    drivers = sorted(
        ({"dimension": d, "contribution": round(cfg["weights"][d] * scores[d], 1), "text": text[d],
          "mitigation": cfg["mitigations"].get(mitigation[d])} for d in scores if scores[d] >= cfg["driver_min_score"]),
        key=lambda d: -d["contribution"],
    )[:3]
    return {
        "level": level,
        "total": round(total, 1),
        "scores": {k: round(v) for k, v in scores.items()},
        "drivers": drivers,
        "mitigations": [d["mitigation"] for d in drivers if d["mitigation"]],
        "needs_more_evidence": level == "HIGH" and evidence_points >= cfg["needs_more_evidence"]["evidence_score"],
        "news_available": bool(news),
    }
