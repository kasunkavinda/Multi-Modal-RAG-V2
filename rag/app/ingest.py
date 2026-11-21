"""Loaders per corpus behind one interface, the content-type router, the OCR gate,
and figure/table extraction with provenance (E1, E2).

Every unit leaves this module stamped document > page > unit id. Search text is NOT
produced here (that is enrichment): we only extract the artefact and its local context.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol

import pymupdf

from .config import Settings
from .schemas import Provenance, TableData, Unit

log = logging.getLogger(__name__)
PARSER_VERSION = "pdf-v3"


# =================================================================== report
@dataclass
class IngestReport:
    pages: list[dict] = field(default_factory=list)          # per-page signals + class
    review_queue: list[dict] = field(default_factory=list)   # pages failing the OCR gate (not indexed)
    rejected: list[dict] = field(default_factory=list)       # units rejected at any stage, with reason
    notes: list[str] = field(default_factory=list)
    ocr_backend: str = "none"
    ocr_pages: int = 0

    def pages_by_class(self) -> dict[str, int]:
        return dict(Counter(p["class"] for p in self.pages))

    def reject(self, unit_id: str, reason: str, stage: str, **extra) -> None:
        self.rejected.append({"unit_id": unit_id, "stage": stage, "reason": reason, **extra})


@dataclass
class IngestContext:
    settings: Settings
    report: IngestReport
    ocr: Callable[[Path], "OcrResult"] | None = None   # page-image -> OCR result
    cache: object | None = None                          # KVCache (parse cache)


class Loader(Protocol):
    name: str

    def load(self, ctx: IngestContext) -> Iterator[Unit]: ...


def rel(settings: Settings, path: Path) -> str:
    """Path relative to the data folder (portable, and what the artefact guard checks)."""
    return path.resolve().relative_to(settings.data_dir.resolve()).as_posix()


# ============================================================ OCR + quality gate
@dataclass
class OcrResult:
    text: str
    mean_confidence: float | None   # None when the backend has no confidence (vision OCR)
    backend: str


_PUNCT = "()[]{}.,:;'\"-/\\|*#+=_<>!?&%$@~`"


def garble_ratio(text: str) -> float:
    """Share of tokens that look like OCR noise: mostly non-alphanumeric, or stray single glyphs."""
    tokens = text.split()
    if not tokens:
        return 1.0
    bad = 0
    for tok in tokens:
        core = tok.strip(_PUNCT)
        if not core:
            # punctuation-only token: noise, except a lone dash/bullet or leader dots ("....")
            bad += 0 if len(tok) == 1 or set(tok) <= {".", " "} else 1
            continue
        alnum = sum(ch.isalnum() for ch in core)
        if alnum / len(core) < 0.6:
            bad += 1
        elif len(core) == 1 and core not in "aAI&" and not core.isdigit():
            bad += 1
        elif len(core) > 3 and not re.search(r"[aeiouyAEIOUY0-9]", core):
            bad += 1  # consonant soup ("rnnrnwq")
    return bad / len(tokens)


def ocr_gate(result: OcrResult, settings: Settings) -> tuple[bool, str]:
    """Returns (passed, reason). Sub-threshold pages go to the review queue, never the index."""
    g = garble_ratio(result.text)
    if result.mean_confidence is not None and result.mean_confidence < settings.ocr_min_confidence:
        return False, f"mean OCR confidence {result.mean_confidence:.0f} < {settings.ocr_min_confidence:.0f}"
    if g > settings.ocr_max_garble:
        return False, f"garbled-token ratio {g:.2f} > {settings.ocr_max_garble:.2f}"
    return True, f"ok (garble {g:.2f}, conf {result.mean_confidence if result.mean_confidence is not None else 'n/a'})"


def tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def tesseract_ocr(image_path: Path) -> OcrResult:
    import pytesseract
    from PIL import Image

    with Image.open(image_path) as im:
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for w, c in zip(data["text"], data["conf"]):
        if w.strip():
            words.append(w)
            try:
                if float(c) >= 0:
                    confs.append(float(c))
            except ValueError:
                pass
    return OcrResult(" ".join(words), sum(confs) / len(confs) if confs else 0.0, "tesseract")


# ============================================================ Enginuity Task 2
class EnginuityTask2Loader:
    """Tier A: 12 figures + their parts-list tables (PNG + TSV), no source page numbers.

    Provenance: each figure group is a document ("enginuity-t2-figNN"); the figure is page 1
    and its parts-list tables are pages 2.. in order. The dataset ships crops without the
    source manual page, so this is the resolvable page *within the figure group*.
    """

    name = "enginuity_t2"

    def base(self, s: Settings) -> Path:
        return s.raw_dir / "enginuity" / "task-2-question-answering"

    def load(self, ctx: IngestContext) -> Iterator[Unit]:
        s = ctx.settings
        base = self.base(s)
        if not base.exists():
            raise FileNotFoundError(f"Enginuity Task 2 not found at {base}; run scripts/download_data.py")
        for fig in sorted((base / "figures").glob("figure_*.*")):
            nn = re.search(r"figure_(\d+)", fig.stem).group(1)  # type: ignore[union-attr]
            doc = f"enginuity-t2-fig{nn}"
            fid = f"{doc}:figure"
            yield Unit(
                unit_id=fid,
                type="figure",
                image_path=rel(s, fig),
                provenance=Provenance(document_id=doc, page=1, unit_id=fid, source_path=rel(s, fig)),
                meta={"corpus": self.name, "dataset_path": f"figures/{fig.name}"},
            )
            for tpng in sorted((base / "tables").glob(f"table_{nn}_*.png")):
                m = int(re.search(r"table_\d+_(\d+)", tpng.stem).group(1))  # type: ignore[union-attr]
                tid = f"{doc}:table{m}"
                tsv = base / "table_data" / f"{tpng.stem}.tsv"
                table = read_tsv(tsv) if tsv.exists() else None
                yield Unit(
                    unit_id=tid,
                    type="table",
                    table=table,
                    image_path=rel(s, tpng),  # rendered table image, shown alongside the rows
                    provenance=Provenance(document_id=doc, page=1 + m, unit_id=tid,
                                          source_path=rel(s, tsv if tsv.exists() else tpng)),
                    meta={"corpus": self.name, "dataset_path": f"tables/{tpng.name}",
                          "needs_transcription": table is None},
                )


def read_tsv(path: Path) -> TableData:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [[c.strip() for c in r] for r in csv.reader(fh, delimiter="\t")]
    rows = [r for r in rows if any(r)]
    header, body = rows[0], rows[1:]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in body]
    return TableData(columns=header, rows=body)


# ==================================================================== PDFs
FIG_CAPTION_RE = re.compile(r"^\s*(figure|fig\.)\s*[A-Z]?-?\d+", re.I)
TABLE_CAPTION_RE = re.compile(r"^\s*table\s*[A-Z]?-?\d+", re.I)
LEADER_RE = re.compile(r"^[\s.,\-·_]+$")


def _norm_line(t: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", t.strip().lower()))


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersect(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return _area((x0, y0, x1, y1)) if x1 > x0 and y1 > y0 else 0.0


def _inside(inner, outer, frac: float = 0.6) -> bool:
    a = _area(inner)
    return a > 0 and _intersect(inner, outer) / a >= frac


def _merge_boxes(boxes: list[tuple], gap: float = 4.0) -> list[tuple]:
    """Union overlapping/touching rectangles (tiled scans, split rasters)."""
    boxes = [tuple(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out: list[tuple] = []
        while boxes:
            b = boxes.pop()
            for i, o in enumerate(out):
                if not (b[2] + gap < o[0] or o[2] + gap < b[0] or b[3] + gap < o[1] or o[3] + gap < b[1]):
                    out[i] = (min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3]))
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


@dataclass
class PageSignals:
    page: int
    chars: int
    images: int
    image_coverage: float       # union of image boxes / page area
    drawings: int
    has_table: bool = False
    cls: str = "native"


def classify_page(sig: PageSignals) -> str:
    """native text | scanned (no text layer + page-filling image) | scanned_text_layer | mixed | blank."""
    if sig.chars < 20 and sig.images == 0 and sig.drawings < 5:
        return "blank"
    page_filling = sig.image_coverage >= 0.8
    if page_filling and sig.chars < 100:
        return "scanned"
    if page_filling:
        return "scanned_text_layer"   # an invisible OCR layer under a page image: gate it too
    if sig.image_coverage >= 0.15 or sig.has_table or sig.drawings >= 50:
        return "mixed"
    return "native"


def _page_signals(page: pymupdf.Page, pno: int) -> tuple[PageSignals, list[tuple]]:
    A = _area(page.rect)
    img_boxes = [tuple(i["bbox"]) for i in page.get_image_info() if _area(i["bbox"]) > 0]
    merged = _merge_boxes(img_boxes)
    cov = min(1.0, sum(_area(b) for b in merged) / A) if A else 0.0
    sig = PageSignals(
        page=pno,
        chars=len(page.get_text().strip()),
        images=len(img_boxes),
        image_coverage=round(cov, 3),
        drawings=len(page.get_drawings()),
    )
    return sig, merged


def _repeated_lines(doc: pymupdf.Document, max_pages: int) -> set[str]:
    """Header/footer lines: in the top/bottom 7% band and repeated on >= 30% of pages."""
    seen: Counter[str] = Counter()
    n = min(doc.page_count, max_pages)
    for pno in range(n):
        page = doc[pno]
        h = page.rect.height
        lines = set()
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            if b[3] <= 0.07 * h or b[1] >= 0.93 * h:
                for ln in b[4].splitlines():
                    if ln.strip():
                        lines.add(_norm_line(ln))
        seen.update(lines)
    return {ln for ln, c in seen.items() if c >= max(3, 0.3 * n)}


def table_to_data(page: pymupdf.Page, tb) -> TableData | None:
    """Rebuild rows from word geometry inside the table box.

    PyMuPDF's line-based finder often returns one cell per column holding every row joined
    by newlines (parts lists have vertical rules only). We take the column boundaries from
    the finder and re-cut rows by text line, merging wrapped continuation lines.
    """
    rows_with_cells = [r for r in tb.rows if r.cells]
    if not rows_with_cells:
        return None
    best = max(rows_with_cells, key=lambda r: sum(c is not None for c in r.cells))
    col_bounds = sorted({(round(c[0], 1), round(c[2], 1)) for c in best.cells if c})
    if len(col_bounds) < 2:
        return None

    header_y1 = tb.header.bbox[3] if getattr(tb, "header", None) is not None and not tb.header.external else tb.bbox[1]
    names = []
    if getattr(tb, "header", None) is not None and tb.header.names and len(tb.header.names) == len(col_bounds):
        names = [re.sub(r"\s+", " ", (n or "").replace("\n", " ")).strip() for n in tb.header.names]
    if not names or not any(names):
        names = [f"col{i+1}" for i in range(len(col_bounds))]

    words = [w for w in page.get_text("words") if _inside(w[:4], tb.bbox, 0.5) and w[1] >= header_y1 - 1]
    words = [w for w in words if not LEADER_RE.match(w[4])]
    if not words:
        return None
    # cluster into lines by vertical centre
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines: list[list] = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if lines and abs(((lines[-1][0][1] + lines[-1][0][3]) / 2) - yc) <= 3.0:
            lines[-1].append(w)
        else:
            lines.append([w])

    def col_of(w) -> int:
        xc = (w[0] + w[2]) / 2
        for i, (x0, x1) in enumerate(col_bounds):
            if x0 - 2 <= xc <= x1 + 2:
                return i
        return min(range(len(col_bounds)), key=lambda i: abs((col_bounds[i][0] + col_bounds[i][1]) / 2 - xc))

    rows: list[list[str]] = []
    for ln in lines:
        cells = [""] * len(col_bounds)
        for w in sorted(ln, key=lambda w: w[0]):
            i = col_of(w)
            cells[i] = (cells[i] + " " + w[4]).strip()
        anchor = cells[0] or cells[1] if len(cells) > 1 else cells[0]
        if rows and not anchor:
            # continuation of a wrapped row
            rows[-1] = [(a + " " + b).strip() if b else a for a, b in zip(rows[-1], cells)]
        else:
            rows.append(cells)
    rows = [[re.sub(r"\s*[.,]\s*(?:[.,]\s*){2,}", " ", c).strip() for c in r] for r in rows]
    # drop empty columns
    keep = [i for i in range(len(col_bounds)) if any(r[i] for r in rows)]
    rows = [[r[i] for i in keep] for r in rows]
    names = [names[i] for i in keep]
    if len(rows) < 2 or len(keep) < 2:
        return None
    return TableData(columns=names, rows=rows)


class PdfLoader:
    """Tier B/C: real manuals. Routes each page, OCRs scans behind the gate, emits
    text / figure / table units with document > page > unit id."""

    def __init__(self, name: str, pdfs: list[Path]):
        self.name = name
        self.pdfs = pdfs

    def load(self, ctx: IngestContext) -> Iterator[Unit]:
        for pdf in self.pdfs:
            if not pdf.exists():
                ctx.report.notes.append(f"{self.name}: missing PDF {pdf} (skipped)")
                continue
            yield from self._load_pdf(ctx, pdf)

    # ------------------------------------------------------------------ pdf
    def _load_pdf(self, ctx: IngestContext, pdf: Path) -> Iterator[Unit]:
        s, report = ctx.settings, ctx.report
        cache = ctx.cache
        key = None
        if cache is not None:
            key = cache.key(PARSER_VERSION, hashlib.sha256(pdf.read_bytes()).hexdigest(), str(s.max_pages_per_pdf),
                            str(s.min_figure_px), str(s.min_figure_area), report.ocr_backend,
                            str(s.ocr_min_confidence), str(s.ocr_max_garble))
            hit = cache.get("parse", key)
            if hit:
                report.pages.extend(hit["pages"])
                report.review_queue.extend(hit["review"])
                report.rejected.extend(hit["rejected"])
                report.ocr_pages += hit.get("ocr_pages", 0)
                for u in hit["units"]:
                    yield Unit.model_validate(u)
                return

        start_pages, start_review, start_rej, start_ocr = (
            len(report.pages), len(report.review_queue), len(report.rejected), report.ocr_pages)
        units = list(self._parse_pdf(ctx, pdf))
        if cache is not None and key is not None:
            cache.put("parse", key, {
                "pages": report.pages[start_pages:],
                "review": report.review_queue[start_review:],
                "rejected": report.rejected[start_rej:],
                "ocr_pages": report.ocr_pages - start_ocr,
                "units": [u.model_dump() for u in units],
            })
        yield from units

    def _parse_pdf(self, ctx: IngestContext, pdf: Path) -> Iterator[Unit]:
        s, report = ctx.settings, ctx.report
        doc_id = pdf.stem
        doc = pymupdf.open(pdf)
        n_pages = min(doc.page_count, s.max_pages_per_pdf or doc.page_count)
        repeated = _repeated_lines(doc, n_pages)
        fig_dir = s.artefact_dir / "figures"
        page_dir = s.artefact_dir / "pages"
        fig_dir.mkdir(parents=True, exist_ok=True)
        page_dir.mkdir(parents=True, exist_ok=True)
        log.info("parsing %s (%d pages)", pdf.name, n_pages)

        for pno in range(n_pages):
            page = doc[pno]
            pnum = pno + 1
            A = _area(page.rect)
            sig, img_regions = _page_signals(page, pnum)

            # ---------------- tables (only where a table is plausible: cheap gate)
            tables: list[tuple[tuple, TableData, str | None]] = []
            if sig.drawings >= 10 or sig.chars >= 1500:
                try:
                    for tb in page.find_tables().tables:
                        data = table_to_data(page, tb)
                        if data is not None:
                            tables.append((tuple(tb.bbox), data, None))
                except Exception as e:  # table finder can fail on odd pages
                    report.notes.append(f"{doc_id} p{pnum}: table finder error {e}")
            sig.has_table = bool(tables)
            sig.cls = classify_page(sig)
            page_rec = {"document_id": doc_id, "page": pnum, "class": sig.cls, "chars": sig.chars,
                        "images": sig.images, "image_coverage": sig.image_coverage, "drawings": sig.drawings,
                        "tables": len(tables), "figures": 0, "text_units": 0}
            report.pages.append(page_rec)
            if sig.cls == "blank":
                continue

            blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0 and b[4].strip()]
            h = page.rect.height
            blocks = [b for b in blocks
                      if not ((b[3] <= 0.07 * h or b[1] >= 0.93 * h)
                              and all(_norm_line(ln) in repeated for ln in b[4].splitlines() if ln.strip()))]

            # ---------------- scanned pages: OCR behind the quality gate
            page_text_blocks: list[tuple[tuple, str]] = [(tuple(b[:4]), b[4]) for b in blocks]
            scan_is_figure = False
            if sig.cls in ("scanned", "scanned_text_layer"):
                layer_text = " ".join(t for _, t in page_text_blocks)
                use_layer = sig.cls == "scanned_text_layer" and garble_ratio(layer_text) <= s.ocr_max_garble
                # A full-page image whose text layer is only a "Figure N. Title" line is a scanned
                # diagram, not scanned prose: route it to the figure path, no OCR needed.
                fig_titled = any(FIG_CAPTION_RE.match(re.sub(r"\s+", " ", t)) or
                                 re.search(r"\bfigure\s*[A-Z]?\d+", re.sub(r"\s+", " ", t), re.I)
                                 for _, t in page_text_blocks)
                if fig_titled and sig.chars < 200:
                    scan_is_figure = True
                    use_layer = True
                    page_text_blocks = [(bb, re.sub(r"\s+", " ", t)) for bb, t in page_text_blocks]
                if not use_layer:
                    img_path = page_dir / f"{doc_id}_p{pnum}.png"
                    page.get_pixmap(dpi=200).save(img_path)
                    if ctx.ocr is None:
                        report.review_queue.append({"document_id": doc_id, "page": pnum,
                                                    "reason": "scanned page but no OCR backend available",
                                                    "image": rel(s, img_path)})
                        continue
                    try:
                        res = ctx.ocr(img_path)
                    except Exception as e:
                        report.review_queue.append({"document_id": doc_id, "page": pnum,
                                                    "reason": f"OCR failed: {e}", "image": rel(s, img_path)})
                        continue
                    report.ocr_pages += 1
                    n_words = len(res.text.split())
                    if n_words < 8:
                        scan_is_figure = True        # image page with almost no text: a diagram
                        page_text_blocks = []
                    else:
                        ok, why = ocr_gate(res, s)
                        page_rec["ocr"] = {"passed": ok, "reason": why, "words": n_words,
                                           "confidence": res.mean_confidence, "garble": round(garble_ratio(res.text), 3)}
                        if not ok:
                            report.review_queue.append({"document_id": doc_id, "page": pnum, "reason": why,
                                                        "image": rel(s, img_path), "backend": res.backend})
                            continue
                        page_text_blocks = [(tuple(page.rect), res.text)]
                        scan_is_figure = n_words < 80 and sig.image_coverage >= 0.8

            # ---------------- figures: raster regions + vector drawing clusters
            fig_boxes: list[tuple] = []
            for b in img_regions:
                if sig.cls in ("scanned", "scanned_text_layer") and not scan_is_figure and _area(b) / A > 0.8:
                    continue  # the scan itself is a text page, not a figure
                fig_boxes.append(b)
            if sig.drawings >= 20 and not tables:
                try:
                    for r in page.cluster_drawings():
                        fig_boxes.append(tuple(r))
                except Exception:
                    pass
            fig_boxes = _merge_boxes(fig_boxes)
            zoom = 2.0
            fig_boxes = [b for b in fig_boxes
                         if (b[2] - b[0]) * zoom >= s.min_figure_px and (b[3] - b[1]) * zoom >= s.min_figure_px
                         and _area(b) / A >= s.min_figure_area
                         and not any(_inside(b, t[0], 0.5) for t in tables)]

            fig_captions = [(bb, t.strip()) for bb, t in page_text_blocks
                            if FIG_CAPTION_RE.match(t) or (scan_is_figure and re.search(r"\bfigure\b", t, re.I))]
            used_blocks: set[int] = set()
            for i, fb in enumerate(fig_boxes):
                fid = f"{doc_id}:p{pnum}:fig{i+1}"
                out = fig_dir / f"{doc_id}_p{pnum}_fig{i+1}.png"
                clip = pymupdf.Rect(fb) & page.rect
                page.get_pixmap(clip=clip, matrix=pymupdf.Matrix(zoom, zoom)).save(out)
                caption = None
                if fig_captions:
                    caption = min(fig_captions, key=lambda c: abs(c[0][1] - fb[3]))[1]
                labels = []
                for j, (bb, t) in enumerate(page_text_blocks):
                    if _inside(bb, fb, 0.8) and _area(bb) < 0.5 * _area(fb):
                        labels.append(t.strip())
                        used_blocks.add(j)
                page_rec["figures"] += 1
                yield Unit(
                    unit_id=fid, type="figure", image_path=rel(s, out), caption=caption,
                    provenance=Provenance(document_id=doc_id, page=pnum, unit_id=fid,
                                          bbox=[round(x, 1) for x in fb], source_path=rel(s, pdf)),
                    meta={"corpus": self.name, "page_class": sig.cls,
                          "labels": " ".join(labels)[:500], "from_scan": scan_is_figure},
                )

            # ---------------- tables
            for i, (tbbox, data, _) in enumerate(tables):
                tid = f"{doc_id}:p{pnum}:tab{i+1}"
                cap = None
                above = [(bb, t) for bb, t in page_text_blocks if bb[3] <= tbbox[1] + 2 and tbbox[1] - bb[3] < 60]
                for bb, t in reversed(above):
                    if TABLE_CAPTION_RE.match(t) or len(t) < 120:
                        cap = t.strip()
                        break
                for j, (bb, t) in enumerate(page_text_blocks):
                    if _inside(bb, tbbox, 0.6):
                        used_blocks.add(j)
                yield Unit(
                    unit_id=tid, type="table", table=data, caption=cap,
                    provenance=Provenance(document_id=doc_id, page=pnum, unit_id=tid,
                                          bbox=[round(x, 1) for x in tbbox], source_path=rel(s, pdf)),
                    meta={"corpus": self.name, "page_class": sig.cls},
                )

            # ---------------- text chunks (reading order, headers/footers removed)
            heading = None
            buf: list[str] = []
            bbs: list[tuple] = []
            chunks: list[tuple[str, tuple, str | None]] = []

            def flush():
                if buf:
                    txt = re.sub(r"[ \t]+", " ", "\n".join(buf)).strip()
                    box = (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))
                    chunks.append((txt, box, heading))
                buf.clear()
                bbs.clear()

            for j, (bb, t) in enumerate(page_text_blocks):
                if j in used_blocks:
                    continue
                t = t.strip()
                if len(t) < 80 and (t.isupper() or re.match(r"^(section|chapter|\d+(\.\d+)*\s+[A-Z])", t, re.I)):
                    flush()
                    heading = re.sub(r"\s+", " ", t)
                    continue
                buf.append(t)
                bbs.append(bb)
                if sum(len(x) for x in buf) >= 1000:
                    flush()
            flush()
            k = 0
            for txt, box, head in chunks:
                if len(txt) < 40 or garble_ratio(txt) > s.ocr_max_garble:
                    if len(txt) >= 40:
                        report.reject(f"{doc_id}:p{pnum}:text", "garbled text chunk", "ingest")
                    continue
                k += 1
                uid = f"{doc_id}:p{pnum}:t{k}"
                page_rec["text_units"] += 1
                yield Unit(
                    unit_id=uid, type="text", text=txt, heading=head,
                    provenance=Provenance(document_id=doc_id, page=pnum, unit_id=uid,
                                          bbox=[round(x, 1) for x in box], source_path=rel(s, pdf)),
                    meta={"corpus": self.name, "page_class": sig.cls,
                          "ocr": sig.cls in ("scanned", "scanned_text_layer")},
                )
        doc.close()


# ================================================================= factory
def get_loaders(settings: Settings) -> list[Loader]:
    loaders: list[Loader] = []
    for name in settings.corpora:
        if name == "enginuity_t2":
            loaders.append(EnginuityTask2Loader())
        elif name == "enginuity_pdf":
            base = settings.raw_dir / "enginuity" / "task-1-component-identification" / "pdfs"
            loaders.append(PdfLoader(name, [base / p for p in settings.enginuity_pdfs]))
        elif name == "vidore":
            base = settings.raw_dir / "vidore" / "pdfs"
            loaders.append(PdfLoader(name, [base / p for p in settings.vidore_pdfs]))
        else:
            raise ValueError(f"unknown corpus {name!r} (expected enginuity_t2, enginuity_pdf, vidore)")
    return loaders


def ingest(ctx: IngestContext) -> list[Unit]:
    units: list[Unit] = []
    for loader in get_loaders(ctx.settings):
        before = len(units)
        units.extend(loader.load(ctx))
        log.info("%s: %d units", loader.name, len(units) - before)
    # unit ids must be unique: a duplicate would make citations ambiguous
    seen: dict[str, int] = defaultdict(int)
    out = []
    for u in units:
        seen[u.unit_id] += 1
        if seen[u.unit_id] > 1:
            ctx.report.reject(u.unit_id, "duplicate unit id", "ingest")
            continue
        out.append(u)
    return out
