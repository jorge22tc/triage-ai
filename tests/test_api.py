"""API integration tests (file-based SQLite via TestClient)."""
import contextlib
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_triage.db"

from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def setup_module():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def teardown_module():
    Base.metadata.drop_all(bind=engine)
    engine.dispose()  # release the SQLite file handle (Windows locks it otherwise)
    with contextlib.suppress(OSError):
        os.remove("test_triage.db")


def _create(subject="Production API down", body="urgent outage in production"):
    return client.post("/api/tickets", json={
        "subject": subject, "body": body, "customer_email": "qa@example.com"})


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_ticket_triages_on_ingest():
    r = _create()
    assert r.status_code == 201
    data = r.json()
    assert data["category"] == "technical"
    assert data["priority"] == "critical"
    assert data["status"] == "open"
    assert data["engine"] in ("llm", "heuristic")


def test_validation_rejects_short_subject():
    r = client.post("/api/tickets", json={"subject": "ab", "body": "valid body here"})
    assert r.status_code == 422


def test_list_and_filter():
    _create("Invoice question", "duplicate charge on my invoice")
    all_ = client.get("/api/tickets").json()
    assert len(all_) >= 2
    billing = client.get("/api/tickets", params={"category": "billing"}).json()
    assert all(t["category"] == "billing" for t in billing)


def test_resolve_flow():
    tid = _create().json()["id"]
    r = client.patch(f"/api/tickets/{tid}/resolve")
    assert r.status_code == 200
    assert r.json()["status"] == "resolved"


def test_resolve_missing_returns_404():
    assert client.patch("/api/tickets/99999/resolve").status_code == 404


def test_stats_shape():
    s = client.get("/api/stats").json()
    assert s["total"] == s["open"] + s["resolved"]
    assert isinstance(s["by_category"], dict)
    assert 0.0 <= s["llm_share"] <= 1.0
