from agent.memory import MAX_TURNS, compact, format_history, recent_turns


def test_compact_drops_sources_headings_and_warnings():
    answer = "## Summary\nRevenue was 13,770 [1].\n\n> Warning: not verified\n\n**Sources**\n[1] Annual report, p.112"
    assert compact(answer) == "Summary Revenue was 13,770 [1]."


def test_compact_truncates_on_a_word_boundary():
    text = compact("word " * 500, limit=50)
    assert text.endswith("...") and len(text) <= 53


def msgs(n):
    out = []
    for i in range(n):
        out += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    return out


def test_recent_turns_keeps_the_last_pairs_in_order():
    turns = recent_turns(msgs(MAX_TURNS + 2))
    assert [t["question"] for t in turns] == [f"q{i}" for i in range(2, MAX_TURNS + 2)]


def test_unanswered_question_is_not_a_turn():
    assert recent_turns([{"role": "user", "content": "q"}]) == []


def test_format_history_labels_both_sides():
    assert format_history([{"question": "q", "answer": "a"}]) == "User: q\nAssistant: a"
