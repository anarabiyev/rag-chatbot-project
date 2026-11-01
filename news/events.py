import re

EVENTS = {
    "recall": r"recall|withdraw",
    "contamination": r"contaminat|glass fragments|salmonella|listeria",
    "price_cut": r"cuts?\b[\w\- ]{0,20}prices?|price cuts?",
    "labour": r"\bstrikes?\b|walkout",
    "legal": r"\bfined?\b|investigat|lawsuit|fraud|\bprobe\b|antitrust",
    "disruption": r"shortage|plant clos|to close|job cuts|layoffs?|axes|terminat",
}


def flag_events(title):
    return [name for name, pattern in EVENTS.items() if re.search(pattern, title, re.I)]
