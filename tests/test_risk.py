import copy

from tools.risk import CONFIG, level_for, score_risk

STRONG = {"best_distance": 0.2, "n_chunks": 5, "relevant": True, "attempts": 1}
WEAK = {"best_distance": 0.4, "n_chunks": 1, "relevant": False, "attempts": 2}
CALM = {"companies": {"Arla": {"negative_share": 0.0, "events": 0}, "Danone": {"negative_share": 0.0, "events": 0}}}
HOSTILE = {"companies": {"Arla": {"negative_share": 0.6, "events": 4}, "Danone": {"negative_share": 0.5, "events": 1}}}


def policy(verdict, *doc_types):
    return {"verdict": verdict, "policies": [{"doc_id": f"{t}_2025", "doc_type": t, "page": 1} for t in doc_types]}


def test_weights_sum_to_one():
    assert abs(sum(CONFIG["weights"].values()) - 1) < 1e-9


def test_calm_consistent_case_is_low():
    result = score_risk("take a customer to lunch", policy("consistent", "code_of_conduct"), STRONG, CALM)
    assert result["level"] == "LOW"
    assert result["total"] < 35


def test_conflict_with_weak_evidence_and_bad_news_is_high():
    result = score_risk("accept a gift voucher during a tender", policy("conflict", "anti_bribery"), WEAK, HOSTILE)
    assert result["level"] == "HIGH"
    assert result["needs_more_evidence"]


def test_hard_floor_lifts_bribery_conflict_to_medium():
    cfg = copy.deepcopy(CONFIG)
    cfg["weights"] = {"policy": 0.1, "regulatory": 0, "external": 0, "evidence": 0}
    result = score_risk("pay a facilitation payment", policy("conflict", "anti_bribery"), STRONG, CALM, cfg)
    assert result["total"] < cfg["levels"]["medium"]
    assert result["level"] == "MEDIUM"


def test_no_precedent_is_not_floored():
    result = score_risk("move headquarters", policy("no_precedent"), STRONG, CALM)
    assert result["level"] == "LOW"


def test_level_thresholds():
    assert level_for(34.9) == "LOW"
    assert level_for(35) == "MEDIUM"
    assert level_for(64.9) == "MEDIUM"
    assert level_for(65) == "HIGH"


def test_missing_news_is_neutral_and_flagged():
    result = score_risk("x", policy("consistent"), STRONG, None)
    assert result["scores"]["external"] == CONFIG["external"]["missing_score"]
    assert not result["news_available"]


def test_changing_a_weight_changes_the_total():
    cfg = copy.deepcopy(CONFIG)
    cfg["weights"]["policy"] = 0.0
    args = ("x", policy("conflict", "code_of_conduct"), STRONG, CALM)
    assert score_risk(*args, cfg)["total"] < score_risk(*args)["total"]


def test_regulatory_score_is_capped():
    action = "gift voucher, tax planning, child labour, palm oil and a sanction"
    assert score_risk(action, policy("no_precedent"), STRONG, CALM)["scores"]["regulatory"] == 100


def test_drivers_are_sorted_and_carry_mitigations():
    result = score_risk("accept a gift voucher", policy("conflict", "anti_bribery"), WEAK, HOSTILE)
    contributions = [d["contribution"] for d in result["drivers"]]
    assert contributions == sorted(contributions, reverse=True)
    assert len(result["drivers"]) <= 3
    assert any("Legal" in m for m in result["mitigations"])


def story(title, *events):
    return {"company": "Arla", "title": title, "events": list(events)}


NEWS_WITH_RECALLS = {
    "companies": {"Arla": {"negative_share": 0.2, "events": 3}},
    "stories": [story("Arla recalls protein pudding over glass fragments", "recall", "contamination")] * 3,
}


def test_unrelated_events_do_not_raise_external_score():
    voucher = score_risk("accept a gift voucher during a tender", policy("no_precedent"), STRONG, NEWS_WITH_RECALLS)
    recall = score_risk("withdraw a protein pudding batch after a recall", policy("no_precedent"), STRONG, NEWS_WITH_RECALLS)
    assert voucher["scores"]["external"] < 20
    assert recall["scores"]["external"] > 60


def test_headline_word_overlap_makes_an_event_relevant():
    news = {"companies": {"Arla": {"negative_share": 0.0, "events": 3}},
            "stories": [story("Arla Sweden axes contracts with 52 farms", "disruption")] * 3}
    result = score_risk("cut organic milk contracts with farms in Sweden", policy("no_precedent"), STRONG, news)
    assert result["scores"]["external"] == 60
