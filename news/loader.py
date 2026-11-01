import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

from rag.paths import SNAPSHOTS

COMPANIES = {
    "Arla": '"Arla Foods"',
    "Lactalis": "Lactalis",
    "FrieslandCampina": "FrieslandCampina",
    "Danone": "Danone",
}


def load_snapshot(folder=SNAPSHOTS):
    path = max(Path(folder).glob("news_snapshot_*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [i for i in data["items"] if i["source_type"] == "google_news"]
    return items, data["fetched_at"], f"snapshot {path.name}"


def fetch_live(days=30):
    import feedparser
    import requests

    items = []
    for company, phrase in COMPANIES.items():
        url = f"https://news.google.com/rss/search?q={quote_plus(f'{phrase} when:{days}d')}&hl=en-GB&gl=GB&ceid=GB:en"
        feed = feedparser.parse(requests.get(url, timeout=20).content)
        for e in feed.entries:
            published = e.get("published_parsed")
            items.append({
                "company": company,
                "title": e.get("title", ""),
                "url": e.get("link", ""),
                "published": datetime(*published[:6], tzinfo=timezone.utc).isoformat() if published else None,
                "publisher": e.get("source", {}).get("title", ""),
            })
    return items, datetime.now(timezone.utc).isoformat(), "live"


def load_news(live=False):
    if live:
        try:
            return fetch_live()
        except Exception:  # offline or blocked, use the snapshot
            pass
    return load_snapshot()
