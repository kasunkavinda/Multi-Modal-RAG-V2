"""E3: captions cached and non-empty; tables linearised with structure kept; rejects visible."""
from app.enrich import enrich_units, is_usable_caption, linearise_table
from app.ingest import IngestReport
from app.openrouter import KVCache
from app.schemas import Provenance, TableData, Unit
from tests.conftest import FakeClient

GOOD_CAPTION = ("Exploded view of a diesel engine cylinder block assembly showing callouts 1-31, "
                "expansion plugs, pipe plugs, cylinder liner with O-rings items 28 and 29.")


def _fig(uid="d:p1:fig1"):
    return Unit(unit_id=uid, type="figure", image_path="artefacts/figures/f.png",
                provenance=Provenance(document_id="d", page=1, unit_id=uid))


def test_linearise_keeps_structure():
    t = TableData(columns=["item_no", "part_no", "description"], rows=[["1", "8923317", "ENGINE BLOCK,DIESEL"]])
    before = t.model_dump()
    s = linearise_table(t, caption="Parts list")
    assert "part_no=8923317" in s and "Parts list" in s
    assert t.model_dump() == before


def test_caption_cached_and_rejects(settings, figure_png):
    cache = KVCache(settings.cache_dir / "c.sqlite")
    fc = FakeClient(answer=GOOD_CAPTION)
    out = enrich_units([_fig()], settings, IngestReport(), fc, cache)
    assert out[0].search_text.startswith("Exploded view")
    calls = sum(fc.stats.values())
    enrich_units([_fig()], settings, IngestReport(), fc, cache)
    assert sum(fc.stats.values()) == calls, "second run must be served from cache"

    rep2 = IngestReport()
    out2 = enrich_units([_fig("d:p1:fig2")], settings, rep2, FakeClient(answer="A technical diagram."),
                        KVCache(settings.cache_dir / "c2.sqlite"))
    assert out2 == [] and "unusable caption" in rep2.rejected[0]["reason"]


def test_no_provenance_rejected_before_paying(settings, figure_png):
    fc = FakeClient(answer=GOOD_CAPTION)
    u = Unit(unit_id="d:x", type="figure", image_path="artefacts/figures/f.png",
             provenance=Provenance(document_id="d", page=None, unit_id="d:x"))
    rep = IngestReport()
    assert enrich_units([u], settings, rep, fc, KVCache(settings.cache_dir / "c.sqlite")) == []
    assert sum(fc.stats.values()) == 0 and "provenance" in rep.rejected[0]["reason"]


def test_generic_caption_detection():
    assert not is_usable_caption("")[0]
    assert not is_usable_caption("A technical diagram.")[0]
    assert is_usable_caption(GOOD_CAPTION)[0]
