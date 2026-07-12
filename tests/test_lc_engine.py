"""LangChain/LangGraph tier tests — fully offline via FakeListChatModel."""
import json

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.classifier import classify
from app.lc_engine import classify_lc

VALID = json.dumps({
    "category": "technical",
    "priority": "critical",
    "sentiment": "negative",
    "confidence": 0.93,
    "summary": "Production API is down for all users.",
})


def test_lc_valid_output_returns_langchain_engine():
    model = FakeListChatModel(responses=[VALID])
    res = classify_lc("API DOWN", "production outage, all users affected", model=model)
    assert res is not None
    assert res.engine == "langchain"
    assert res.category == "technical"
    assert res.priority == "critical"
    assert res.sentiment == "negative"
    assert 0.0 <= res.confidence <= 1.0


def test_lc_malformed_json_yields_none():
    model = FakeListChatModel(responses=["sorry, I cannot classify this"])
    assert classify_lc("hi", "hello", model=model) is None


def test_lc_out_of_contract_labels_yield_none():
    bad = json.dumps({"category": "pizza", "priority": "yesterday"})
    model = FakeListChatModel(responses=[bad])
    assert classify_lc("x", "y", model=model) is None


def test_lc_bad_sentiment_is_normalized():
    odd = json.dumps({"category": "billing", "priority": "low", "sentiment": "confused"})
    model = FakeListChatModel(responses=[odd])
    res = classify_lc("invoice", "question about invoice", model=model)
    assert res is not None and res.sentiment == "neutral"


def test_service_never_fails_when_lc_tier_fails(monkeypatch):
    """End-to-end guarantee: langchain tier down -> heuristic still triages."""
    monkeypatch.setenv("TRIAGE_ENGINE", "langchain")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = classify("URGENT: production down", "error 500, cannot work, losing money")
    assert res.engine == "heuristic"
    assert res.priority == "critical"
