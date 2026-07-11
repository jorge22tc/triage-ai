"""Unit tests for the triage engine (heuristic tier — deterministic)."""
from app.classifier import CATEGORIES, PRIORITIES, SENTIMENTS, _classify_heuristic, classify


def test_billing_ticket_is_categorized():
    t = _classify_heuristic("Charged twice", "My invoice shows a duplicate charge, I want a refund")
    assert t.category == "billing"
    assert t.engine == "heuristic"


def test_technical_outage_is_critical():
    t = _classify_heuristic("Production down", "Our production API is down, this is urgent")
    assert t.category == "technical"
    assert t.priority == "critical"


def test_account_lockout_detected():
    t = _classify_heuristic("Locked out", "I forgot my password and my account is locked")
    assert t.category == "account"


def test_negative_sentiment():
    t = _classify_heuristic(
        "Terrible service", "This is unacceptable, I am frustrated and want to cancel")
    assert t.sentiment == "negative"
    assert t.priority in ("critical", "high")


def test_positive_sentiment():
    t = _classify_heuristic("Thanks", "Thank you, the tool is excellent and I love it")
    assert t.sentiment == "positive"


def test_unknown_ticket_falls_to_other_low():
    t = _classify_heuristic("Hello", "Just wanted to say hi")
    assert t.category == "other"
    assert t.priority == "low"


def test_labels_always_in_contract():
    for subject, body in [
        ("Refund please", "charge on my invoice"),
        ("URGENT server down", "production outage now"),
        ("hi", "random text with no keywords"),
    ]:
        t = classify(subject, body)  # no API key in test env -> heuristic
        assert t.category in CATEGORIES
        assert t.priority in PRIORITIES
        assert t.sentiment in SENTIMENTS
        assert 0.0 <= t.confidence <= 1.0


def test_heuristic_confidence_capped_below_llm_levels():
    t = _classify_heuristic(
        "URGENT billing error", "invoice charge refund payment urgent emergency production")
    assert t.confidence <= 0.75
