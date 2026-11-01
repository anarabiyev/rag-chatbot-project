import sys

from news.tool import get_market_news

sys.stdout.reconfigure(encoding="utf-8")

result = get_market_news()
print(f"source: {result['origin']} | fetched: {result['fetched_at'][:10]} | window: {result['days']} days\n")
print(f"{'company':18} {'raw':>5} {'stories':>8} {'rated':>6} {'avg':>6} {'neg':>5} {'events':>7}")
for company, s in result["companies"].items():
    print(f"{company:18} {result['raw_items'][company]:5} {s['stories']:8} {s['rated']:6} {s['avg_sentiment']!s:>6} {s['negative_share']!s:>5} {s['events']:7}")
print(f"\ncompetitor sentiment: {result['competitor_sentiment']}")

for company, s in result["companies"].items():
    print(f"\n{company}: most negative / latest")
    for tag, rows in (("neg", s["most_negative"]), ("new", s["latest"])):
        for r in rows:
            print(f"  {tag} {r['sentiment']:+.2f} x{r['outlets']} {r['published'][:10]} {r['publisher'][:22]:22} {','.join(r['events']):13} {r['title'][:80]}")
