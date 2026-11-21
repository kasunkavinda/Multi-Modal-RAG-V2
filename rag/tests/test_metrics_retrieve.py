"""E8.1 metrics + retrieval end to end with the fake embedder (and BM25 fusion on an exact code)."""
import json

from app.retrieve import Retriever, tokenize
from app.store import MemoryStore, save_catalog
from eval.metrics import aggregate, first_relevant_rank, is_relevant, per_type_ranks, recall_at, reciprocal_rank
from tests.conftest import FakeClient, make_units


def test_relevance_rules():
    assert is_relevant({"unit_id": "a"}, {"unit_id": "a", "type": "figure"})
    tgt = {"document_id": "d", "page": 3, "type": "table"}
    assert is_relevant({"document_id": "d", "page": 3, "type": "table"}, tgt)
    assert not is_relevant({"document_id": "d", "page": 3, "type": "text"}, tgt), "page-only match must not count"


def test_rank_metrics():
    hits = [{"unit_id": "x"}, {"unit_id": "g"}]
    assert first_relevant_rank(hits, [{"unit_id": "g"}]) == 2
    assert recall_at(2, 1) == 0 and recall_at(2, 3) == 1 and reciprocal_rank(2) == 0.5
    ranks = per_type_ranks([{"unit_id": "t"}, {"unit_id": "f"}],
                           [{"unit_id": "f", "type": "figure"}, {"unit_id": "t", "type": "table"}])
    assert ranks == {"figure": 2, "table": 1}
    agg = aggregate([{"ranks": ranks}, {"ranks": {"figure": None}}])
    assert agg["figure"]["recall@5"] == 0.5 and agg["table"]["mrr"] == 1.0


def test_tokenize_keeps_codes():
    t = tokenize("NSN 5340-00-530-5543 bolt")
    assert "5340-00-530-5543" in t and "5340005305543" in t


def _build(settings, figure_png):
    fc = FakeClient()
    units = make_units(settings, figure_png)
    store = MemoryStore(settings)
    store.reset(64)
    store.write(units, fc.embed_texts([u.search_text for u in units]))
    save_catalog(settings, units)
    settings.index_report_path.write_text(json.dumps({
        "embedding": {"model": settings.embed_model, "dimensions": settings.embed_dimensions,
                      "figure_mode": "caption"},
        "vector_backend": "memory"}))
    return fc, store


def test_retrieve_with_type_filter(settings, figure_png):
    fc, store = _build(settings, figure_png)
    r = Retriever(settings, fc, store)
    res = r.retrieve("relay control wiring", k=3)
    assert res.hits[0].unit.unit_id == "docA:p3:fig1"
    assert res.hits[0].citation == "docA > p.3 > docA:p3:fig1"
    only_tables = r.retrieve("relay control wiring", k=3, types=["table"])
    assert all(h.unit.type == "table" for h in only_tables.hits)


def test_fusion_surfaces_exact_identifier(settings, figure_png):
    fc, store = _build(settings, figure_png)
    settings.fusion = True
    settings.bm25_weight = 0.6
    r = Retriever(settings, fc, store)
    res = r.retrieve("M8", k=3)
    assert res.hits[0].unit.unit_id == "docA:p4:tab1"
    assert res.hits[0].bm25_score is not None
