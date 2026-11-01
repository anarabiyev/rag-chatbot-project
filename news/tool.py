from collections import Counter
from statistics import mean

from news.clean import clean
from news.events import flag_events
from news.loader import COMPANIES, load_news
from news.sentiment import score

NEGATIVE = -0.25


def summarise(stories, top_n):
    rated = [s for s in stories if not s["corporate"]]
    summary = {"stories": len(stories), "rated": len(rated), "avg_sentiment": None, "negative_share": None,
               "events": sum(bool(s["events"]) for s in stories)}
    if rated:
        scores = [s["sentiment"] for s in rated]
        summary["avg_sentiment"] = round(mean(scores), 2)
        summary["negative_share"] = round(sum(x < NEGATIVE for x in scores) / len(scores), 2)
    summary["most_negative"] = sorted(rated, key=lambda s: s["sentiment"])[:top_n]
    summary["latest"] = sorted(stories, key=lambda s: s["published"], reverse=True)[:top_n]
    return summary


def get_market_news(companies=None, days=30, topic=None, live=False, top_n=3):
    items, fetched_at, origin = load_news(live)
    companies = companies or list(COMPANIES)
    stories = [s for s in clean(items, fetched_at, days) if s["company"] in companies]
    if topic:
        stories = [s for s in stories if topic.lower() in s["title"].lower()]
    for story, value in zip(stories, score([s["title"] for s in stories])):
        story["sentiment"] = value
        story["events"] = flag_events(story["title"])

    summary = {c: summarise([s for s in stories if s["company"] == c], top_n) for c in companies}
    rivals = [v["avg_sentiment"] for c, v in summary.items() if c != "Arla" and v["avg_sentiment"] is not None]
    return {
        "origin": origin,
        "fetched_at": fetched_at,
        "days": days,
        "raw_items": dict(Counter(i["company"] for i in items)),
        "companies": summary,
        "competitor_sentiment": round(mean(rivals), 2) if rivals else None,
        "stories": stories,
    }
