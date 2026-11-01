from rag.graph import format_context, window


def doc(text, match=""):
    return {"text": text, "match": match, "source": "a.pdf", "page": 1, "year": 2024}


def test_short_text_is_kept_whole():
    assert window(doc("short text"), 100) == "short text"


def test_window_is_centered_near_the_match():
    text = "x" * 2000 + "THE ANSWER ROW 13,770" + "y" * 2000
    cut = window(doc(text, "THE ANSWER ROW 13,770"), 500)
    assert len(cut) == 500
    assert "THE ANSWER ROW 13,770" in cut


def test_window_at_the_end_stays_inside_the_text():
    text = "x" * 3000 + "FINAL ROW"
    assert "FINAL ROW" in window(doc(text, "FINAL ROW"), 400)


def test_window_without_a_match_takes_the_start():
    assert window(doc("a" * 100 + "b" * 100, ""), 50) == "a" * 50


def test_context_stays_within_the_budget():
    docs = [doc("z" * 1500, "z" * 60) for _ in range(8)]
    context = format_context(docs, budget=6000)
    assert len(context) < 6000 + 8 * 60
    assert context.count("[") == 8
