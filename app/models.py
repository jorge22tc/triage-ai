"""ORM models."""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ticket(Base):
    """A support ticket, enriched with AI triage results."""

    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    customer_email: Mapped[str] = mapped_column(String(120), default="anonymous@example.com")

    # Triage results
    category: Mapped[str] = mapped_column(String(40), index=True)      # billing | technical | account | sales | other
    priority: Mapped[str] = mapped_column(String(10), index=True)      # critical | high | medium | low
    sentiment: Mapped[str] = mapped_column(String(10))                 # negative | neutral | positive
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    engine: Mapped[str] = mapped_column(String(20), default="heuristic")  # llm | heuristic
    summary: Mapped[str] = mapped_column(String(300), default="")

    status: Mapped[str] = mapped_column(String(15), default="open", index=True)  # open | resolved
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
