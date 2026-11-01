import re

MAX_TURNS = 4
MAX_ANSWER_CHARS = 600


def compact(answer, limit=MAX_ANSWER_CHARS):
    """Short version of an answer for the planner: no source list, headings or warnings, cut to `limit`."""
    text = re.split(r"\n\s*(?:\*\*Sources\*\*|## Sources)", answer)[0]
    text = re.sub(r"^>.*$|^#+\s*", "", text, flags=re.M)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def recent_turns(messages, max_turns=MAX_TURNS):
    """Last question and answer pairs from chat messages ({'role': 'user' | 'assistant', 'content': str})."""
    turns, question = [], None
    for m in messages:
        if m["role"] == "user":
            question = m["content"]
        elif question is not None:
            turns.append({"question": question, "answer": compact(m["content"])})
            question = None
    return turns[-max_turns:]


def format_history(history):
    return "\n".join(f"User: {t['question']}\nAssistant: {t['answer']}" for t in history[-MAX_TURNS:])
