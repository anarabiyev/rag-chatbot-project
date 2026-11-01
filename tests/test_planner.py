import json

import pytest

from agent.planner import plan


def reply(monkeypatch, intent="lookup", questions=None, raw=None):
    text = raw if raw is not None else json.dumps({"intent": intent, "questions": questions})
    monkeypatch.setattr("agent.planner.ask", lambda prompt, task: text)


@pytest.mark.parametrize("intent", ["lookup", "market", "decision", "out_of_scope"])
def test_valid_intent_is_kept(monkeypatch, intent):
    reply(monkeypatch, intent)
    assert plan("Any question")["intent"] == intent


@pytest.mark.parametrize("raw", ["no idea", '{"intent": "weather"}', '{"questions": []}'])
def test_invalid_answer_defaults_to_lookup(monkeypatch, raw):
    reply(monkeypatch, raw=raw)
    assert plan("Any question")["intent"] == "lookup"


def test_companies_are_detected_with_a_default(monkeypatch):
    reply(monkeypatch)
    assert plan("What is Danone doing?")["companies"] == ["Danone"]
    assert len(plan("What is the latest news?")["companies"]) == 4


def test_action_is_the_question_without_the_question_mark(monkeypatch):
    reply(monkeypatch, "decision")
    assert plan("Should we pay a bribe?")["action"] == "Should we pay a bribe"


@pytest.mark.parametrize("intent", ["lookup", "market"])
def test_compound_question_is_split(monkeypatch, intent):
    reply(monkeypatch, intent, ["Revenue in 2023?", "Revenue in 2024?"])
    assert plan("Compare revenue in 2023 and 2024")["subquestions"] == ["Revenue in 2023?", "Revenue in 2024?"]


def test_split_is_capped_at_three_and_deduplicated(monkeypatch):
    reply(monkeypatch, questions=["a", "a", "b", "c", "d"])
    assert plan("Question")["subquestions"] == ["a", "b", "c"]


@pytest.mark.parametrize("questions", [[], ["only one"], "not a list", [1, None]])
def test_unusable_split_keeps_the_original_question(monkeypatch, questions):
    reply(monkeypatch, questions=questions)
    assert plan("Question")["subquestions"] == ["Question"]


@pytest.mark.parametrize("intent", ["decision", "out_of_scope"])
def test_decision_and_out_of_scope_are_not_split(monkeypatch, intent):
    reply(monkeypatch, intent, ["a", "b"])
    assert plan("Question")["subquestions"] == ["Question"]


HISTORY = [{"question": "What was Arla's revenue in 2024?", "answer": "Revenue was EUR 13,770 million [1]."}]


def capture(monkeypatch, **fields):
    seen = {}

    def fake_ask(prompt, task):
        seen["prompt"] = prompt
        return json.dumps({"intent": "lookup", "questions": [], **fields})

    monkeypatch.setattr("agent.planner.ask", fake_ask)
    return seen


def test_follow_up_is_rewritten_using_the_history(monkeypatch):
    seen = capture(monkeypatch, standalone="What was Arla's revenue in 2023?")
    result = plan("and in 2023?", HISTORY)
    assert result["question"] == "What was Arla's revenue in 2023?"
    assert result["subquestions"] == ["What was Arla's revenue in 2023?"]
    assert "Arla's revenue in 2024" in seen["prompt"] and "and in 2023?" in seen["prompt"]


@pytest.mark.parametrize("value", [None, "", "   ", 5])
def test_missing_standalone_keeps_the_original_question(monkeypatch, value):
    capture(monkeypatch, standalone=value)
    assert plan("and in 2023?", HISTORY)["question"] == "and in 2023?"


def test_first_message_has_no_history_in_the_prompt_and_ignores_standalone(monkeypatch):
    seen = capture(monkeypatch, standalone="something else")
    result = plan("What was Arla's revenue in 2024?")
    assert result["question"] == "What was Arla's revenue in 2024?"
    assert "Conversation so far" not in seen["prompt"] and "standalone" not in seen["prompt"]


def test_rewritten_question_is_used_for_companies_and_action(monkeypatch):
    capture(monkeypatch, intent="decision", standalone="Should we sign a supplier contract with Danone?")
    result = plan("what about Danone?", HISTORY)
    assert result["companies"] == ["Danone"]
    assert result["action"] == "Should we sign a supplier contract with Danone"


def test_chat_question_is_not_rewritten(monkeypatch):
    capture(monkeypatch, intent="chat", standalone="What was Arla's revenue in 2024?")
    result = plan("what was my last question?", HISTORY)
    assert result["intent"] == "chat" and result["question"] == "what was my last question?"


DECISION_QUESTIONS = [
    "Should we accept a EUR 300 gift voucher from a supplier during an ongoing tender?",
    "Can we pay a small facilitation payment to speed up customs clearance?",
    "Should we take a customer to a normal business lunch costing EUR 80 per head?",
    "Should we move the company headquarters to another Danish city?",
    "We are considering a recruitment fee charged to workers at a supplier plant",
    "Would it be acceptable to schedule 16-year-olds for night shifts?",
    "Is it OK to accept a gift from a supplier?",
]


@pytest.mark.parametrize("question", DECISION_QUESTIONS)
def test_clear_decision_phrasing_is_decided_without_the_llm(monkeypatch, question):
    def no_llm(prompt, task):
        raise AssertionError("the LLM must not be called")

    monkeypatch.setattr("agent.planner.ask", no_llm)
    result = plan(question)
    assert result["intent"] == "decision" and result["subquestions"] == [question]


@pytest.mark.parametrize("question", ["What was Arla's revenue in 2024?", "What can you do?", "What is the latest news about Arla?",
                                      "Who conducts the on-farm audits?", "What is the best pizza topping?"])
def test_other_questions_still_go_to_the_llm(monkeypatch, question):
    reply(monkeypatch, "lookup")
    assert plan(question)["intent"] == "lookup"


def test_rule_overrides_the_llm_label(monkeypatch):
    reply(monkeypatch, "lookup")
    assert plan("Should we sign the contract?")["intent"] == "decision"


def test_decision_phrasing_in_a_follow_up_is_found_after_the_rewrite(monkeypatch):
    capture(monkeypatch, standalone="Should we accept a EUR 500 gift voucher from a supplier?")
    assert plan("what about EUR 500?", HISTORY)["intent"] == "decision"


@pytest.mark.parametrize("question", ["Who are Arla's main competitors in the dairy market right now?",
                                      "How is the competition doing?", "What are the rivals up to?"])
def test_competitor_questions_fetch_every_company(monkeypatch, question):
    reply(monkeypatch, "market")
    assert len(plan(question)["companies"]) == 4
