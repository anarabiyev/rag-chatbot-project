import re
from datetime import datetime, timedelta

from rapidfuzz import fuzz


def strip_publisher(title, publisher):
    suffix = f" - {publisher}"
    if publisher and title.endswith(suffix):
        return title[: -len(suffix)]
    return re.sub(r" - [^-]{2,40}$", "", title)


def clean(items, fetched_at, days=30, similarity=70):
    since = datetime.fromisoformat(fetched_at) - timedelta(days=days)
    stories = []
    for item in sorted(items, key=lambda i: i["published"] or ""):
        title = strip_publisher(item["title"], item["publisher"])
        if not item["published"] or datetime.fromisoformat(item["published"]) < since:
            continue
        if item["company"].lower() not in title.lower():
            continue
        same = next((s for s in stories
                     if s["company"] == item["company"] and fuzz.token_set_ratio(s["title"], title) >= similarity), None)
        if same:
            same["outlets"] += 1
            continue
        stories.append({
            "company": item["company"],
            "title": title,
            "url": item["url"],
            "published": item["published"],
            "publisher": item["publisher"],
            "outlets": 1,
            "corporate": item["company"].lower() in item["publisher"].lower(),
        })
    return stories
