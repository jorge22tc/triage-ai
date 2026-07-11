# TriageAI

> Intelligent support-ticket triage: LLM classification with a deterministic fallback — because production AI must never be a single point of failure.

[![CI](https://github.com/jorge22tc/triage-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/jorge22tc/triage-ai/actions)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

TriageAI ingests support tickets and instantly classifies **category** (billing / technical / account / sales / other), **priority** (critical → low), and **sentiment**, then exposes a live operations dashboard. It is a public, from-scratch implementation of the AI-triage patterns I work with in enterprise telecom environments.

## Why this design is interesting

**The LLM is a tier, not a dependency.** Every ticket is triaged by a two-tier engine:

```
            ┌──────────────────────────┐
 ticket ──► │ 1. LLM tier              │ ──► JSON contract validated ──► result (engine: "llm")
            │  (Anthropic or Gemini)   │
            └────────────┬─────────────┘
                         │ no key · timeout · malformed output · out-of-contract labels
                         ▼
            ┌──────────────────────────┐
            │ 2. Heuristic tier        │ ──► always succeeds ──► result (engine: "heuristic")
            │  (deterministic, 0 deps) │
            └──────────────────────────┘
```

- The service **never fails to triage** because an AI provider is down — it degrades gracefully.
- Every result carries `engine` and `confidence`, so consumers know how much to trust each label. Heuristic confidence is deliberately capped below LLM levels.
- The LLM output is validated against a strict contract; anything malformed falls through to heuristics instead of poisoning the queue.

## Features

- **REST API** (FastAPI + SQLAlchemy 2.0, typed end-to-end) — submit, list/filter, resolve, stats
- **Live dashboard** — queue with priority/category/status filters, real-time stats, LLM-share metric
- **Bilingual heuristics** — keyword engine understands English and Spanish tickets
- **PostgreSQL in production, SQLite by default** — clone and run with zero setup
- **CI on every push** — ruff + pytest on Python 3.11/3.12 + Docker build (GitHub Actions)

## Quickstart

```bash
git clone https://github.com/jorge22tc/triage-ai.git
cd triage-ai
pip install -r requirements.txt
uvicorn app.main:app --reload
# Dashboard: http://localhost:8000  ·  API docs: http://localhost:8000/docs
```

With Docker (PostgreSQL included):

```bash
docker compose up --build
```

Enable the LLM tier (optional):

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or GEMINI_API_KEY=...
uvicorn app.main:app
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/tickets` | Submit a ticket — triaged synchronously on ingest |
| `GET` | `/api/tickets?category=&priority=&status=` | List tickets, newest first, filterable |
| `PATCH` | `/api/tickets/{id}/resolve` | Mark resolved |
| `GET` | `/api/stats` | Aggregates: totals, by category/priority, LLM share |
| `GET` | `/health` | Liveness probe |

Example:

```bash
curl -X POST localhost:8000/api/tickets \
  -H 'Content-Type: application/json' \
  -d '{"subject": "Production API down", "body": "urgent outage, customers affected"}'
```

```json
{
  "category": "technical",
  "priority": "critical",
  "sentiment": "neutral",
  "confidence": 0.65,
  "engine": "heuristic",
  "status": "open",
  "...": "..."
}
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v          # 15 tests: engine contract, API integration, edge cases
ruff check app tests
```

## Project structure

```
app/
├── main.py         # FastAPI app + routes
├── classifier.py   # two-tier triage engine (LLM + heuristic fallback)
├── models.py       # SQLAlchemy 2.0 ORM
├── schemas.py      # Pydantic request/response contracts
└── db.py           # engine/session, env-driven database URL
static/index.html   # operations dashboard (vanilla JS, zero build step)
tests/              # unit + integration
```

## License

MIT © [Jorge Martes](https://github.com/jorge22tc) · [jorge22tc.github.io](https://jorge22tc.github.io)
