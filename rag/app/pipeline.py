"""build_index: ingest > enrich > embed > upsert (fail closed), plus the query service shared
by the API server and the evaluator (same code path for both)."""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .generate import answer_question
from .ingest import IngestContext, IngestReport, OcrResult, ingest, tesseract_available, tesseract_ocr
from .openrouter import KVCache, OpenRouterClient
from .retrieve import Retriever, figure_mode
from .schemas import Provenance, QueryResponse, Timings, Unit
from .store import get_store, save_catalog

log = logging.getLogger(__name__)

OCR_PROMPT = ("Transcribe all text on this scanned manual page exactly, in reading order, as plain text. "
              "No commentary. If there is no readable text, reply with nothing.")


def make_ocr(settings: Settings, client: OpenRouterClient | None, cache: KVCache, report: IngestReport):
    backend = settings.ocr_backend
    if backend == "auto":
        backend = "tesseract" if tesseract_available() else ("vision" if client else "none")
    report.ocr_backend = backend
    if backend == "tesseract":
        if not tesseract_available():
            report.notes.append("OCR_BACKEND=tesseract but the tesseract binary was not found; scans go to review")
            report.ocr_backend = "none"
            return None
        return tesseract_ocr
    if backend == "vision" and client is not None:
        budget = {"left": settings.max_vision_ocr_pages}

        def vision_ocr(path: Path) -> OcrResult:
            key = KVCache.key("ocr-v1", path.read_bytes())
            hit = cache.get("ocr", key)
            if hit is not None:
                return OcrResult(hit, None, "vision")
            if budget["left"] <= 0:
                raise RuntimeError(f"vision OCR budget exhausted (MAX_VISION_OCR_PAGES={settings.max_vision_ocr_pages})")
            budget["left"] -= 1
            text, _ = client.vision_with_fallback(OCR_PROMPT, [path], role="ocr", max_tokens=2500)
            cache.put("ocr", key, text)
            return OcrResult(text, None, "vision")

        return vision_ocr
    report.ocr_backend = "none"
    return None


def broken_unit() -> Unit:
    """A deliberately unattributable unit (no page) used to demonstrate the write-time fence (E4.2)."""
    return Unit(unit_id="demo:broken-unit", type="text", text="Torque all bolts to 25 Nm.",
                search_text="Torque all bolts to 25 Nm.",
                provenance=Provenance(document_id="demo", page=None, unit_id="demo:broken-unit"))


def build_index(settings: Settings, *, reset: bool = True, ingest_only: bool = False,
                inject_broken_unit: bool = False, client: OpenRouterClient | None = None) -> dict:
    t_start = time.perf_counter()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    cache = KVCache(settings.cache_dir / "cache.sqlite")
    if client is None and not ingest_only:
        client = OpenRouterClient(settings, cache)
    report = IngestReport()
    ctx = IngestContext(settings=settings, report=report, ocr=make_ocr(settings, client, cache, report), cache=cache)

    # 1-2. parse, route, extract
    units = ingest(ctx)
    t_ingest = time.perf_counter()
    raw_counts = dict(Counter(u.type for u in units))
    (settings.data_dir / "review_queue.json").write_text(json.dumps(report.review_queue, indent=1), encoding="utf-8")

    if ingest_only:
        out = {
            "mode": "ingest_only", "corpora": settings.corpora, "pages_by_class": report.pages_by_class(),
            "units_extracted": raw_counts, "review_queue": len(report.review_queue),
            "ocr": {"backend": report.ocr_backend, "pages_ocrd": report.ocr_pages},
            "rejected": report.rejected, "notes": report.notes, "pages": report.pages,
            "ingest_s": round(t_ingest - t_start, 2),
        }
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        (settings.reports_dir / "ingest_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return out

    assert client is not None
    # 3. enrich
    kept = enrich_units_wrapper(units, settings, report, client, cache)
    t_enrich = time.perf_counter()

    # 4. embed (same model as queries; figures by caption or by image)
    if settings.embed_images:
        fig = [u for u in kept if u.type == "figure"]
        other = [u for u in kept if u.type != "figure"]
        vecs_other = client.embed_texts([u.search_text for u in other])
        vecs_fig = client.embed_images([settings.data_dir / u.image_path for u in fig])  # type: ignore[operator]
        kept = other + fig
        vectors = vecs_other + vecs_fig
    else:
        vectors = client.embed_texts([u.search_text for u in kept])
    t_embed = time.perf_counter()
    dim = len(vectors[0]) if vectors else (settings.embed_dimensions or 0)

    # 5. upsert with provenance checked at write time (fail closed)
    store = get_store(settings)
    if reset:
        store.reset(dim)
    to_write, to_vecs = list(kept), list(vectors)
    if inject_broken_unit and vectors:
        to_write.append(broken_unit())
        to_vecs.append(vectors[0])
    rejected_at_write = store.write(to_write, to_vecs)
    for r in rejected_at_write:
        report.rejected.append(r)
    rejected_ids = {r["unit_id"] for r in rejected_at_write}
    stored_units = [u for u in to_write if u.unit_id not in rejected_ids]
    save_catalog(settings, stored_units)
    t_upsert = time.perf_counter()

    # captions export (human-readable evidence for the caption cache)
    caps = {k: v for k, v in cache.items("caption")}
    (settings.cache_dir / "captions.json").write_text(json.dumps(caps, indent=1), encoding="utf-8")

    counts = store.count()
    idx = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpora": settings.corpora,
        "embedding": {"model": settings.embed_model, "dimensions": settings.embed_dimensions,
                      "vector_size": dim, "figure_mode": figure_mode(settings)},
        "vector_backend": settings.vector_backend,
        "collection": settings.collection if settings.vector_backend == "qdrant" else str(settings.index_dir),
        "units_by_type": counts,
        "units_total": sum(counts.values()),
        "units_extracted": raw_counts,
        "rejected": report.rejected,
        "rejected_by_stage": dict(Counter(r["stage"] for r in report.rejected)),
        "pages_by_class": report.pages_by_class(),
        "review_queue": {"count": len(report.review_queue), "file": "data/review_queue.json"},
        "ocr": {"backend": report.ocr_backend, "pages_ocrd": report.ocr_pages},
        "timings_s": {"ingest": round(t_ingest - t_start, 2), "enrich": round(t_enrich - t_ingest, 2),
                      "embed": round(t_embed - t_enrich, 2), "upsert": round(t_upsert - t_embed, 2),
                      "total": round(t_upsert - t_start, 2)},
        "model_calls": dict(client.stats),
        "cache_hits": dict(client.cache_hits),
        "index_size_bytes": store.size_bytes(),
        "settings": {"link_tables_to_figures": settings.link_tables_to_figures, "max_captions": settings.max_captions},
        "notes": report.notes,
    }
    settings.index_report_path.write_text(json.dumps(idx, indent=1), encoding="utf-8")
    client.close()
    return idx


def enrich_units_wrapper(units, settings, report, client, cache):
    from .enrich import enrich_units

    return enrich_units(units, settings, report, client, cache)


class RagService:
    """Lazily-built query pipeline. The API server and the evaluator both call `query`."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._lock = threading.Lock()
        self._client: OpenRouterClient | None = None
        self._retriever: Retriever | None = None

    @property
    def client(self) -> OpenRouterClient:
        if self._client is None:
            self._client = OpenRouterClient(self.s)
        return self._client

    @property
    def retriever(self) -> Retriever:
        with self._lock:
            if self._retriever is None:
                self._retriever = Retriever(self.s, self.client)
            return self._retriever

    def reload(self) -> None:
        with self._lock:
            self._retriever = None

    def query(self, question: str, k: int | None = None, types: list[str] | None = None,
              generate: bool = True) -> QueryResponse:
        retriever = self.retriever
        calls_before = sum(self.client.stats.values())  # live calls only (cache hits cost nothing)
        r = retriever.retrieve(question, k, types)
        timings = Timings(embed_ms=r.embed_ms, search_ms=r.search_ms, retrieval_ms=r.embed_ms + r.search_ms)
        resp = answer_question(question, r.hits, self.s, self.client, timings, generate=generate)
        resp.model_calls = sum(self.client.stats.values()) - calls_before
        return resp
