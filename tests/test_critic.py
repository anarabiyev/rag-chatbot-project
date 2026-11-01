from agent.nodes import check_result, merge_results, open_issue

DOC = {"text": "Revenue 13,770 13,674 13,793 11,202 EBITDA 1,109", "title": "T", "page": 8, "year": 2024, "doc_type": "annual_report"}


def result(**changes):
    base = {"question": "What was Arla's revenue in 2024?", "docs": [DOC], "answer": "Revenue was EUR 13,770 million [1].",
            "quote": '"Revenue 13,770 13,674" [1]', "relevant": True, "attempts": 1, "index": 0}
    return {**base, **changes}


def test_grounded_answer_passes():
    assert check_result(result()) == []


def test_not_relevant_is_flagged():
    assert check_result(result(relevant=False)) == ["no relevant evidence found"]


def test_missing_citation_and_missing_source():
    assert "answer has no citations" in check_result(result(answer="Revenue was EUR 13,770 million."))
    assert "answer cites a missing source" in check_result(result(answer="Revenue was EUR 13,770 million [3]."))


def test_figure_must_be_in_the_cited_source():
    notes = check_result(result(answer="Revenue was EUR 2,471 million [1]."))
    assert any("2,471" in n for n in notes)


def test_figure_from_the_question_is_not_flagged():
    assert check_result(result(question="Did revenue reach 15.1 billion?", answer="No, 13,770 million [1]. Not 15.1 billion.")) == []


def test_quote_must_come_from_the_cited_source():
    assert "quote not found in the cited sources" in check_result(result(quote='"Net profit was 401 million and rising fast" [1]'))
    assert "answer has no supporting quote" in check_result(result(quote=""))


def test_quote_matching_tolerates_spacing_and_case():
    assert check_result(result(quote='"revenue  13,770\n13,674" [1]')) == []


def test_merge_renumbers_citations_across_parts():
    a = result(index=0, question="Q1", answer="First [1].", quote="q [1]", docs=[DOC])
    b = result(index=1, question="Q2", answer="Second [1][2].", quote="q [2]", docs=[DOC, DOC], attempts=2)
    merged = merge_results([b, a])
    assert len(merged["docs"]) == 3
    assert "Second [2][3]." in merged["rag_answer"]
    assert merged["quote"] == "q [1] | q [3]"
    assert merged["rag_attempts"] == 2


def test_merge_of_a_single_result_keeps_the_answer_as_is():
    assert merge_results([result()])["rag_answer"] == "Revenue was EUR 13,770 million [1]."


def test_open_issue_detection():
    assert open_issue({"quote": "The merger is still pending", "rag_answer": ""})
    assert not open_issue({"quote": "Revenue was 13,770", "rag_answer": "Revenue grew."})


from agent.nodes import critic_node


def state(**changes):
    base = {"rag_results": [{**result(), "round": 0}], "retries": 0, "answer": "Final answer"}
    return {**base, **changes}


def not_found(**changes):
    return {**result(answer="The context does not name specific competitors.", quote=""), "round": 0, **changes}


def test_not_found_rag_answer_is_not_flagged_when_news_carries_the_answer():
    assert critic_node(state(rag_results=[not_found()], news={"companies": {}}))["critic_ok"]
    assert critic_node(state(rag_results=[not_found(relevant=False)], news={"companies": {}}))["critic_ok"]


def test_not_found_rag_answer_is_still_flagged_without_news():
    assert not critic_node(state(rag_results=[not_found()]))["critic_ok"]


def test_grounded_check_still_applies_when_news_is_present():
    bad = {**result(answer="Revenue was EUR 2,471 million [1]."), "round": 0}
    assert not critic_node(state(rag_results=[bad], news={"companies": {}}))["critic_ok"]
