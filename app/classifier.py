"""Ticket triage engine.

Two-tier design, mirroring how AI is deployed responsibly in production:

1. **LLM tier** — if an API key is configured (`ANTHROPIC_API_KEY` or
   `GEMINI_API_KEY`), tickets are classified by a language model with a
   strict JSON contract.
2. **Heuristic tier** — a deterministic, dependency-free keyword engine.
   It is the automatic fallback when no key is set, the LLM errors out,
   or the LLM returns malformed output. The service NEVER fails to
   triage a ticket because of an upstream AI outage.

Every result carries `engine` ("llm" | "heuristic") and `confidence`
so downstream consumers can decide how much to trust the label.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger("triageai.classifier")

CATEGORIES = ("billing", "technical", "account", "sales", "other")
PRIORITIES = ("critical", "high", "medium", "low")
SENTIMENTS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class Triage:
    category: str
    priority: str
    sentiment: str
    confidence: float
    engine: str
    summary: str


# --------------------------------------------------------------------- LLM
_PROMPT = """You are a support-ticket triage system. Classify the ticket below.

Respond with ONLY a JSON object, no markdown, exactly this shape:
{{"category": "billing|technical|account|sales|other",
  "priority": "critical|high|medium|low",
  "sentiment": "negative|neutral|positive",
  "confidence": 0.0-1.0,
  "summary": "<one sentence, max 25 words>"}}

Ticket subject: {subject}
Ticket body: {body}
"""


def _classify_llm(subject: str, body: str) -> Triage | None:
    """Classify with Anthropic or Gemini. Returns None on any failure."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    prompt = _PROMPT.format(subject=subject[:200], body=body[:2000])

    try:
        if anthropic_key:
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": anthropic_key,
                         "anthropic-version": "2023-06-01"},
                json={"model": os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
                      "max_tokens": 200,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=20,
            )
            r.raise_for_status()
            text = r.json()["content"][0]["text"]
        elif gemini_key:
            model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
            r = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": gemini_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=20,
            )
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return None

        raw = re.search(r"\{.*\}", text, re.S)
        data = json.loads(raw.group(0) if raw else text)
        if data.get("category") not in CATEGORIES or data.get("priority") not in PRIORITIES:
            raise ValueError(f"LLM returned out-of-contract labels: {data}")
        return Triage(
            category=data["category"],
            priority=data["priority"],
            sentiment=data.get("sentiment", "neutral"),
            confidence=min(max(float(data.get("confidence", 0.8)), 0.0), 1.0),
            engine="llm",
            summary=str(data.get("summary", ""))[:300],
        )
    except Exception as exc:  # noqa: BLE001 — any LLM failure falls back to heuristics
        logger.warning("LLM triage failed, falling back to heuristics: %s", exc)
        return None


# --------------------------------------------------------------- heuristics
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": ("invoice", "billing", "charge", "charged", "refund", "payment",
                "factura", "cobro", "pago", "reembolso", "price", "subscription"),
    "technical": ("error", "bug", "crash", "down", "outage", "broken", "fail",
                  "slow", "timeout", "500", "404", "falla", "caido", "lento",
                  "not working", "no funciona", "api", "login error"),
    "account": ("password", "login", "account", "access", "locked", "2fa",
                "contraseña", "cuenta", "acceso", "bloqueada", "email change"),
    "sales": ("quote", "pricing", "upgrade", "plan", "demo", "purchase",
              "cotizacion", "comprar", "contratar", "enterprise plan"),
}

_URGENCY_KEYWORDS = ("urgent", "asap", "immediately", "critical", "emergency",
                     "production", "outage", "down", "urgente", "emergencia",
                     "caido", "produccion", "cannot work", "losing money")

_NEGATIVE_KEYWORDS = ("angry", "terrible", "awful", "worst", "unacceptable",
                      "frustrated", "disappointed", "cancel", "molesto",
                      "terrible", "inaceptable", "frustrado", "cancelar")

_POSITIVE_KEYWORDS = ("thanks", "thank you", "great", "love", "excellent",
                      "gracias", "excelente", "encanta")


def _score(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for kw in keywords if kw in text)


def _classify_heuristic(subject: str, body: str) -> Triage:
    """Deterministic keyword-based triage — the always-available fallback."""
    text = f"{subject} {body}".lower()

    scores = {cat: _score(text, kws) for cat, kws in _CATEGORY_KEYWORDS.items()}
    best_cat, best_score = max(scores.items(), key=lambda kv: kv[1])
    category = best_cat if best_score > 0 else "other"

    urgency = _score(text, _URGENCY_KEYWORDS)
    negative = _score(text, _NEGATIVE_KEYWORDS)
    positive = _score(text, _POSITIVE_KEYWORDS)

    if urgency >= 2 or (urgency >= 1 and category == "technical"):
        priority = "critical"
    elif urgency >= 1 or negative >= 2:
        priority = "high"
    elif best_score >= 1:
        priority = "medium"
    else:
        priority = "low"

    sentiment = ("negative" if negative > positive
                 else "positive" if positive > negative
                 else "neutral")

    # Confidence grows with keyword evidence but is capped below LLM levels.
    confidence = round(min(0.35 + 0.15 * best_score + 0.05 * urgency, 0.75), 2)

    summary = subject.strip()[:140]
    return Triage(category, priority, sentiment, confidence, "heuristic", summary)


# ------------------------------------------------------------------ public
def classify(subject: str, body: str) -> Triage:
    """Triage a ticket: try the LLM tier, fall back to heuristics."""
    return _classify_llm(subject, body) or _classify_heuristic(subject, body)
