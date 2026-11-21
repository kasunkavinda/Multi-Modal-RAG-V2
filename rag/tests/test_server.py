"""E5.3: API contract, artefact path guard, distinct configuration errors, health degraded state."""
import json

from fastapi.testclient import TestClient

from app.schemas import Provenance, Unit
from app.server import create_app
from app.store import MemoryStore, save_catalog
from tests.conftest import FakeClient, make_units


def _app(settings, figure_png, extra=None):
    units = make_units(settings, figure_png) + (extra or [])
    fc = FakeClient()
    store = MemoryStore(settings)
    store.reset(64)
    store.write(units, fc.embed_texts([u.search_text for u in units]))
    save_catalog(settings, units)
    settings.index_report_path.write_text(json.dumps({
        "embedding": {"model": settings.embed_model, "dimensions": None, "figure_mode": "caption"},
        "vector_backend": "memory", "units_by_type": store.count()}))
    return TestClient(create_app(settings))


def test_health_degraded_without_key(settings, figure_png):
    body = _app(settings, figure_png).get("/health").json()
    assert body["status"] == "degraded"
    assert any("OPENROUTER_API_KEY" in p["message"] for p in body["problems"])


def test_query_without_key_is_configuration_error(settings, figure_png):
    r = _app(settings, figure_png).post("/query", json={"question": "relay wiring"})
    assert r.status_code == 503 and r.json()["error"]["kind"] == "configuration"


def test_artefact_served_and_path_guard(settings, figure_png):
    evil = Unit(unit_id="evil:p1:fig1", type="figure", image_path="../../../../etc/passwd", search_text="evil x",
                provenance=Provenance(document_id="evil", page=1, unit_id="evil:p1:fig1"))
    c = _app(settings, figure_png, [evil])
    ok = c.get("/artefact/docA:p3:fig1")
    assert ok.status_code == 200 and ok.headers["x-citation"] == "docA > p.3 > docA:p3:fig1"
    assert c.get("/artefact/docA:p4:tab1").json()["table"]["rows"][0][1] == "bolt M8"
    assert c.get("/artefact/evil:p1:fig1").status_code == 403
    assert c.get("/artefact/nope").status_code == 404


def test_examples(settings, figure_png):
    assert len(_app(settings, figure_png).get("/examples").json()["examples"]) >= 3


def test_fallback_ui_served(settings, figure_png):
    r = _app(settings, figure_png).get("/ui")
    assert r.status_code == 200 and "Manual Answers" in r.text
