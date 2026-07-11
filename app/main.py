"""TriageAI — FastAPI application entrypoint."""
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import __version__
from .classifier import classify
from .db import Base, engine, get_db
from .models import Ticket
from .schemas import StatsOut, TicketIn, TicketOut

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="TriageAI",
    version=__version__,
    description=(
        "Intelligent support-ticket triage: "
        "LLM classification with a deterministic fallback."
    ),
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.post("/api/tickets", response_model=TicketOut, status_code=201)
def create_ticket(payload: TicketIn, db: Session = Depends(get_db)) -> Ticket:
    """Submit a ticket; it is triaged synchronously on ingestion."""
    triage = classify(payload.subject, payload.body)
    ticket = Ticket(
        subject=payload.subject,
        body=payload.body,
        customer_email=payload.customer_email,
        category=triage.category,
        priority=triage.priority,
        sentiment=triage.sentiment,
        confidence=triage.confidence,
        engine=triage.engine,
        summary=triage.summary,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


@app.get("/api/tickets", response_model=list[TicketOut])
def list_tickets(
    category: str | None = None,
    priority: str | None = None,
    status: str | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[Ticket]:
    """List tickets, newest first, with optional filters."""
    q = select(Ticket).order_by(Ticket.created_at.desc()).limit(min(limit, 500))
    if category:
        q = q.where(Ticket.category == category)
    if priority:
        q = q.where(Ticket.priority == priority)
    if status:
        q = q.where(Ticket.status == status)
    return list(db.scalars(q))


@app.patch("/api/tickets/{ticket_id}/resolve", response_model=TicketOut)
def resolve_ticket(ticket_id: int, db: Session = Depends(get_db)) -> Ticket:
    """Mark a ticket as resolved."""
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    ticket.status = "resolved"
    db.commit()
    db.refresh(ticket)
    return ticket


@app.get("/api/stats", response_model=StatsOut)
def stats(db: Session = Depends(get_db)) -> StatsOut:
    """Aggregate metrics for the dashboard."""
    total = db.scalar(select(func.count(Ticket.id))) or 0
    open_ = db.scalar(select(func.count()).where(Ticket.status == "open")) or 0
    llm = db.scalar(select(func.count()).where(Ticket.engine == "llm")) or 0

    by_cat = dict(db.execute(
        select(Ticket.category, func.count()).group_by(Ticket.category)).all())
    by_pri = dict(db.execute(
        select(Ticket.priority, func.count()).group_by(Ticket.priority)).all())

    return StatsOut(
        total=total, open=open_, resolved=total - open_,
        by_category=by_cat, by_priority=by_pri,
        llm_share=round(llm / total, 3) if total else 0.0,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
