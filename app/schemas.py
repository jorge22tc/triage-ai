"""Pydantic request/response schemas."""
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class TicketIn(BaseModel):
    """Incoming ticket, as a customer or upstream system would submit it."""

    subject: str = Field(min_length=3, max_length=200)
    body: str = Field(min_length=3, max_length=10_000)
    customer_email: EmailStr = "anonymous@example.com"


class TriageResult(BaseModel):
    category: str
    priority: str
    sentiment: str
    confidence: float
    engine: str
    summary: str


class TicketOut(TriageResult):
    id: int
    subject: str
    body: str
    customer_email: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class StatsOut(BaseModel):
    total: int
    open: int
    resolved: int
    by_category: dict[str, int]
    by_priority: dict[str, int]
    llm_share: float
