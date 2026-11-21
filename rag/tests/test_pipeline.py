"""End-to-end wiring: Enginuity-shaped corpus -> build_index (fake models, memory store) -> query."""
import json

from PIL import Image

from app.pipeline import RagService, build_index
from tests.conftest import FakeClient

CAPTION = ("Exploded view of a diesel engine cylinder block with expansion plugs, pipe plugs, "
           "cylinder liner and O-rings, callouts 1 to 31.")


def _corpus(settings):
    base = settings.raw_dir / "enginuity" / "task-2-question-answering"
    (base / "figures").mkdir(parents=True)
    (base / "tables").mkdir()
    (base / "table_data").mkdir()
    Image.new("RGB", (300, 200), "white").save(base / "figures" / "figure_01.png")
    Image.new("RGB", (300, 200), "white").save(base / "tables" / "table_01_1.png")
    (base / "table_data" / "table_01_1.tsv").write_text(
        "item_no\tpart_no\tdescription\tuoc\tquantity\n1\t8923317\tENGINE BLOCK,DIESEL\t\t1\n2\t444618\tPLUG,PIPE\t\t7\n",
        encoding="utf-8")


def test_build_and_query(settings, monkeypatch):
    _corpus(settings)
    settings.corpora = ["enginuity_t2"]
    fc = FakeClient(answer=CAPTION)
    rep = build_index(settings, client=fc, inject_broken_unit=True)
    assert rep["units_by_type"] == {"figure": 1, "table": 1}
    assert any(r["unit_id"] == "demo:broken-unit" and "page" in r["reason"] for r in rep["rejected"])
    assert rep["embedding"]["model"] == settings.embed_model

    svc = RagService(settings)
    svc._client = FakeClient(answer="Order separately [enginuity-t2-fig01:table1].")
    out = svc.query("plug pipe part number 444618", types=["table"])
    assert out.status == "answered" and out.answer_type == "table"
    assert out.citation.text == "enginuity-t2-fig01 > p.2 > enginuity-t2-fig01:table1"
    assert out.artefact.table.rows[1][1] == "444618"
    assert 1 in out.artefact.highlight_rows
    fig = svc.query("cylinder block exploded view", types=["figure"])
    assert fig.artefact.image_url.endswith("enginuity-t2-fig01:figure")
    assert json.loads(settings.index_report_path.read_text())["units_total"] == 2
