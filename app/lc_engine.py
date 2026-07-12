"""LangChain / LangGraph triage tier (optional).

This is a third implementation of the LLM tier built on the LangChain
ecosystem, orchestrated as a small LangGraph state machine:

    classify ──► guard ──► END
        │          │
        │          └─ validates the structured output against the triage
        │             contract; anything out-of-contract yields None so the
        │             caller falls back to the deterministic heuristic tier.
        └─ prompt | chat-model | PydanticOutputParser

Design notes
------------
* **Same philosophy as the core engine**: the LLM (and LangChain itself)
  is a tier, not a dependency. This module is imported lazily and only
  when ``TRIAGE_ENGINE=langchain``; if the extras are not installed the
  service keeps running on the built-in tiers.
* **Dependency injection**: ``build_graph(model)`` accepts any
  ``BaseChatModel`` — production uses Anthropic/Gemini, tests inject a
  ``FakeListChatModel`` so the whole graph runs offline.
* **Structured output**: parsing is handled by ``PydanticOutputParser``
  instead of hand-rolled regex; the guard node re-validates labels so a
  hallucinated category can never poison the queue.

Enable with::

    pip install -r requirements-langchain.txt
    export TRIAGE_ENGINE=langchain
"""
from __future__ import annotations

import logging
import os
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .classifier import CATEGORIES, PRIORITIES, SENTIMENTS, Triage

logger = logging.getLogger("triageai.lc_engine")


class TriageOutput(BaseModel):
    """Structured contract the model must fill."""

    category: str = Field(description="one of: billing, technical, account, sales, other")
    priority: str = Field(description="one of: critical, high, medium, low")
    sentiment: str = Field(default="neutral", description="negative, neutral or positive")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    summary: str = Field(default="", description="one sentence, max 25 words")


class _State(TypedDict):
    subject: str
    body: str
    result: TriageOutput | None


_parser = PydanticOutputParser(pydantic_object=TriageOutput)

_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a support-ticket triage system. Classify the ticket.\n"
     "{format_instructions}"),
    ("human", "Ticket subject: {subject}\nTicket body: {body}"),
]).partial(format_instructions=_parser.get_format_instructions())


def _default_model() -> BaseChatModel | None:
    """Pick a chat model from the configured provider keys (None if none)."""
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=200, timeout=20,
        )
    if os.getenv("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    return None


def build_graph(model: BaseChatModel):
    """Compile the classify→guard LangGraph for the given chat model."""
    chain = _prompt | model | _parser

    def classify_node(state: _State) -> dict:
        try:
            out = chain.invoke(
                {"subject": state["subject"][:200], "body": state["body"][:2000]}
            )
            return {"result": out}
        except Exception as exc:  # noqa: BLE001 — any failure means "no result"
            logger.warning("LangChain tier failed to classify: %s", exc)
            return {"result": None}

    def guard_node(state: _State) -> dict:
        out = state["result"]
        if out is None:
            return {"result": None}
        if out.category not in CATEGORIES or out.priority not in PRIORITIES:
            logger.warning("LangChain tier returned out-of-contract labels: %s", out)
            return {"result": None}
        if out.sentiment not in SENTIMENTS:
            out = out.model_copy(update={"sentiment": "neutral"})
        return {"result": out}

    graph = StateGraph(_State)
    graph.add_node("classify", classify_node)
    graph.add_node("guard", guard_node)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "guard")
    graph.add_edge("guard", END)
    return graph.compile()


def classify_lc(subject: str, body: str,
                model: BaseChatModel | None = None) -> Triage | None:
    """Triage via the LangGraph pipeline. None on any failure (caller falls back)."""
    model = model or _default_model()
    if model is None:
        return None
    final = build_graph(model).invoke({"subject": subject, "body": body, "result": None})
    out = final.get("result")
    if out is None:
        return None
    return Triage(
        category=out.category,
        priority=out.priority,
        sentiment=out.sentiment,
        confidence=min(max(out.confidence, 0.0), 1.0),
        engine="langchain",
        summary=out.summary[:300],
    )
