"""E6: answer with the real artefact; drop unsourced; abstain when nothing sourced; never invent citations."""
from app.generate import answer_question
from app.openrouter import UpstreamError
from app.schemas import Hit, Provenance, Timings, Unit
from tests.conftest import FakeClient, make_units


def _hits(units):
    return [Hit(unit=u, rank=i + 1, score=0.9 - i * 0.1) for i, u in enumerate(units)]


def test_figure_answer_returns_image_and_citation(settings, figure_png):
    units = make_units(settings, figure_png)
    fc = FakeClient(answer="It is the relay wiring [docA:p3:fig1].")
    r = answer_question("show the relay control wiring", _hits(units), settings, fc, Timings())
    assert r.status == "answered" and r.answer_type == "figure"
    assert r.artefact.image_url == "/artefact/docA:p3:fig1"
    assert r.citation.text == "docA > p.3 > docA:p3:fig1"
    assert r.cited_unit_ids == ["docA:p3:fig1"] and r.sources[0].cited


def test_table_answer_returns_exact_rows(settings, figure_png):
    units = make_units(settings, figure_png)
    r = answer_question("torque for bolt M8", _hits([units[1], units[0]]), settings,
                        FakeClient(answer="25 Nm [docA:p4:tab1]."), Timings())
    assert r.answer_type == "table"
    assert r.artefact.table.rows[0] == ["1", "bolt M8", "25 Nm"]
    assert 0 in r.artefact.highlight_rows


def test_unsourced_hits_dropped_and_abstain(settings, figure_png):
    bad = Unit(unit_id="z", type="text", text="Torque 99 Nm", search_text="torque",
               provenance=Provenance(document_id="z", page=None, unit_id="z"))
    fc = FakeClient()
    r = answer_question("torque?", _hits([bad]), settings, fc, Timings())
    assert r.status == "abstained" and r.citation is None and r.artefact is None
    assert "provenance" in r.abstain_reason and not r.sources[0].sourced
    assert sum(fc.stats.values()) == 0, "no generation without a source"


def test_unsourced_top_hit_skipped(settings, figure_png):
    units = make_units(settings, figure_png)
    bad = Unit(unit_id="z", type="figure", image_path="missing.png", search_text="relay",
               provenance=Provenance(document_id="z", page=2, unit_id="z"))
    r = answer_question("relay", _hits([bad, units[0]]), settings,
                        FakeClient(answer="ok [docA:p3:fig1]"), Timings())
    assert r.citation.unit_id == "docA:p3:fig1"
    assert r.sources[0].drop_reason == "figure artefact missing"


def test_invented_citation_removed(settings, figure_png):
    units = make_units(settings, figure_png)
    r = answer_question("relay", _hits(units), settings,
                        FakeClient(answer="See [docQ:p9:fig7] and [docA:p3:fig1]."), Timings())
    assert "docQ" not in r.answer and r.cited_unit_ids == ["docA:p3:fig1"]


def test_generation_failure_still_shows_sourced_artefact(settings, figure_png):
    units = make_units(settings, figure_png)
    fc = FakeClient(fail=UpstreamError("vision-x", "no endpoints (404)", 404))
    r = answer_question("relay", _hits(units), settings, fc, Timings())
    assert r.status == "partial" and r.artefact is not None and "vision-x" in r.generation_error


def test_model_says_unrelated_abstains(settings, figure_png):
    r = answer_question("wine pairing?", _hits(make_units(settings, figure_png)), settings,
                        FakeClient(answer="NOT_IN_SOURCES"), Timings())
    assert r.status == "abstained" and r.citation is None


def test_min_score_abstains(settings, figure_png):
    settings.min_score = 0.95
    r = answer_question("relay", _hits(make_units(settings, figure_png)), settings, FakeClient(), Timings())
    assert r.status == "abstained" and "MIN_SCORE" in r.abstain_reason
