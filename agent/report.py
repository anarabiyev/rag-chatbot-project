def sources_section(docs):
    groups = {}
    for i, d in enumerate(docs, 1):
        groups.setdefault((d["title"], d["page"], d["year"]), []).append(i)
    return "\n".join("".join(f"[{i}]" for i in ids) + f" {title}, p.{page}" + (f" ({year})" if year else "")
                     for (title, page, year), ids in groups.items())


def evidence_line(quote):
    return f"\n\n> Evidence: {quote}" if quote else ""


def outlet(name):
    return name.replace("https://", "").replace("http://", "").rstrip("/")


def market_section(news):
    lines = [f"As of {news['fetched_at'][:10]} ({news['origin']}), last {news['days']} days:"]
    for company, s in news["companies"].items():
        if s["avg_sentiment"] is None:
            lines.append(f"- {company}: {s['stories']} stories, no rated headlines")
        else:
            lines.append(f"- {company}: {s['stories']} stories, average sentiment {s['avg_sentiment']:+.2f}, "
                         f"{s['negative_share']:.0%} negative, {s['events']} event stories")
    arla = news["companies"].get("Arla") or {}
    seen, headlines = set(), []
    for story in arla.get("latest", [])[:3] + arla.get("most_negative", [])[:2]:
        if story["title"] not in seen:
            seen.add(story["title"])
            headlines.append(f"- {story['published'][:10]}, {outlet(story['publisher'])}: [{story['title']}]({story['url']}) "
                             f"(sentiment {story['sentiment']:+.2f})")
    return "\n".join(lines + (["", "Key Arla headlines:"] + headlines if headlines else []))


def policy_section(policy):
    cited = "; ".join(f"{p['title']}, p.{p['page']}" for p in policy["policies"]) or "none"
    return f"Verdict: **{policy['verdict'].upper()}**. {policy['reason']}\n\nCited: {cited}"


def risk_section(risk):
    scores = ", ".join(f"{k} {v}" for k, v in risk["scores"].items())
    lines = [f"**{risk['level']}** (score {risk['total']}/100; {scores})", ""]
    lines += [f"- {d['text']}" for d in risk["drivers"]]
    if not risk["news_available"]:
        lines.append("- News unavailable, so the external dimension is neutral.")
    if risk["mitigations"]:
        lines += ["", "Suggested mitigations:"] + [f"- {m}" for m in risk["mitigations"]]
    return "\n".join(lines)


def facts(state):
    lines = [f"Question: {state['question']}", f"Internal evidence: {state['rag_answer'][:800]}"]
    if state.get("policy"):
        lines.append(f"Policy verdict: {state['policy']['verdict']}. {state['policy']['reason']}")
    if state.get("risk"):
        risk = state["risk"]
        lines.append(f"Risk level: {risk['level']}. Drivers: " + "; ".join(d["text"] for d in risk["drivers"]))
    if state.get("news"):
        lines.append("Companies in the market data: " + ", ".join(state["news"]["companies"]))
        lines.append("Market:\n" + market_section(state["news"]))
    return "\n".join(lines)
