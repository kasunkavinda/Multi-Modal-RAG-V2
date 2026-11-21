"""E1/E2: routing, OCR gate, figure + table extraction with provenance on a synthetic 2-page PDF."""
import pymupdf
import pytest
from PIL import Image, ImageDraw

from app.ingest import (IngestContext, IngestReport, OcrResult, PageSignals, PdfLoader, classify_page,
                        garble_ratio, ocr_gate)

NOISE = "~~ |l| ,,; rnnrnw .:. ^^ ~ ]] [[ xqzpt ;; ,, ;; ~~ || %% ##"


@pytest.fixture
def synthetic_pdf(settings, tmp_path):
    img = tmp_path / "diagram.png"
    im = Image.new("RGB", (800, 600), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([50, 50, 750, 550], outline="black", width=6)
    d.line([50, 50, 750, 550], fill="black", width=6)
    im.save(img)
    scan = tmp_path / "scan.png"
    Image.new("RGB", (1200, 1600), (250, 250, 250)).save(scan)

    doc = pymupdf.open()
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((72, 72), "SECTION 1 PUMP MAINTENANCE", fontsize=12)
    y = 100
    for i in range(12):
        p1.insert_text((72, y), f"Step {i+1}. Disconnect the battery and drain the hydraulic reservoir fully.",
                       fontsize=9)
        y += 14
    p1.insert_image(pymupdf.Rect(100, 300, 500, 600), filename=str(img))
    p1.insert_text((200, 620), "Figure 1. Pump assembly", fontsize=10)
    p2 = doc.new_page(width=612, height=792)
    p2.insert_image(p2.rect, filename=str(scan))  # image-only page: a scan with no text layer
    out = settings.raw_dir / "test.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out


def test_classify_page():
    assert classify_page(PageSignals(1, chars=2000, images=0, image_coverage=0, drawings=0)) == "native"
    assert classify_page(PageSignals(1, chars=10, images=1, image_coverage=0.95, drawings=0)) == "scanned"
    assert classify_page(PageSignals(1, chars=900, images=1, image_coverage=0.95, drawings=0)) == "scanned_text_layer"
    assert classify_page(PageSignals(1, chars=400, images=1, image_coverage=0.4, drawings=0)) == "mixed"
    assert classify_page(PageSignals(1, chars=0, images=0, image_coverage=0, drawings=0)) == "blank"


def test_garble_and_gate(settings):
    clean = "Remove the four bolts securing the pump housing and lift the cover."
    assert garble_ratio(clean) < 0.1
    assert garble_ratio(NOISE) > 0.5
    assert ocr_gate(OcrResult(clean, 91.0, "tesseract"), settings)[0]
    assert not ocr_gate(OcrResult(clean, 30.0, "tesseract"), settings)[0]
    assert not ocr_gate(OcrResult(NOISE, None, "vision"), settings)[0]


def test_pdf_routing_and_provenance(settings, synthetic_pdf):
    rep = IngestReport()
    ocr_calls = []

    def bad_ocr(path):  # garbled OCR must go to the review queue, never the index
        ocr_calls.append(path)
        return OcrResult(NOISE, 20.0, "fake")

    units = list(PdfLoader("t", [synthetic_pdf]).load(IngestContext(settings=settings, report=rep, ocr=bad_ocr)))
    classes = {p["page"]: p["class"] for p in rep.pages}
    assert classes[1] in ("mixed", "native") and classes[2] == "scanned"
    assert len(ocr_calls) == 1 and rep.review_queue and rep.review_queue[0]["page"] == 2
    assert all(u.provenance.page == 1 for u in units), "nothing from the failed scan may be emitted"
    figs = [u for u in units if u.type == "figure"]
    assert len(figs) == 1 and figs[0].caption == "Figure 1. Pump assembly"
    assert (settings.data_dir / figs[0].image_path).exists()
    texts = [u for u in units if u.type == "text"]
    assert texts and "battery" in texts[0].text
    for u in units:
        assert u.provenance.is_resolvable() and u.provenance.unit_id == u.unit_id
        assert u.provenance.document_id == "test"


def test_scan_without_ocr_backend_goes_to_review(settings, synthetic_pdf):
    rep = IngestReport()
    list(PdfLoader("t", [synthetic_pdf]).load(IngestContext(settings=settings, report=rep, ocr=None)))
    assert any(r["page"] == 2 and "no OCR" in r["reason"] for r in rep.review_queue)
