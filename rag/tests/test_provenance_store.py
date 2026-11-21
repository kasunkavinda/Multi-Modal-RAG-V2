"""E4.2: provenance enforced at write time; embedding model recorded and checked."""
import json

import pytest

from app.config import ConfigError
from app.retrieve import check_index
from app.schemas import Provenance, Unit
from app.store import MemoryStore, validate_for_write
from tests.conftest import FakeClient, make_units


def test_unit_without_page_is_refused(settings, figure_png):
    store = MemoryStore(settings)
    store.reset(64)
    bad = Unit(unit_id="x:1", type="text", text="t", search_text="torque 25 Nm",
               provenance=Provenance(document_id="x", page=None, unit_id="x:1"))
    good = make_units(settings, figure_png)[2]
    fc = FakeClient()
    rejected = store.write([bad, good], fc.embed_texts([bad.search_text, good.search_text]))
    assert [r["unit_id"] for r in rejected] == ["x:1"]
    assert "page" in rejected[0]["reason"]
    assert store.count() == {"text": 1}
    assert all(h.unit_id != "x:1" for h in store.search(fc._vec("torque 25 Nm"), 5))


@pytest.mark.parametrize("prov", [
    Provenance(document_id="", page=1, unit_id="u"),
    Provenance(document_id="d", page=0, unit_id="u"),
    Provenance(document_id="d", page=1, unit_id=""),
])
def test_each_missing_field_rejected(prov):
    u = Unit(unit_id="u", type="text", search_text="s", provenance=prov)
    assert validate_for_write(u) is not None


def test_memory_store_type_filter_and_persistence(settings, figure_png):
    fc = FakeClient()
    units = make_units(settings, figure_png)
    store = MemoryStore(settings)
    store.reset(64)
    store.write(units, fc.embed_texts([u.search_text for u in units]))
    hits = store.search(fc._vec("relay wiring"), 3, types=["table"])
    assert hits and all(h.payload["type"] == "table" for h in hits)
    reopened = MemoryStore(settings)
    assert reopened.count() == {"figure": 1, "table": 1, "text": 1}
    p = reopened.search(fc._vec("relay control wiring"), 1)[0].payload
    assert p["document_id"] == "docA" and p["page"] == 3 and p["unit_id"] == "docA:p3:fig1"


def test_mismatched_embedding_model_refused(settings):
    rep = {"embedding": {"model": "other-model", "dimensions": None, "figure_mode": "caption"},
           "vector_backend": "memory"}
    settings.index_report_path.write_text(json.dumps(rep))
    with pytest.raises(ConfigError) as e:
        check_index(settings)
    assert "other-model" in str(e.value) and "rebuild" in e.value.fix.lower()


def test_missing_key_stops_before_any_call(settings):
    from app.openrouter import OpenRouterClient

    with pytest.raises(ConfigError) as e:
        OpenRouterClient(settings)
    assert "OPENROUTER_API_KEY" in str(e.value)
