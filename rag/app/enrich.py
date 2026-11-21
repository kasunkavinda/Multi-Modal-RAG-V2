"""Enrichment (E3): make figures and tables retrievable without discarding the artefact.

- figures: search-oriented vision caption (reasoning off, cached by image hash, one call per figure)
- tables: linearised to column=value rows for search; structured cells stay the artefact
- text: heading + prose
- units that cannot be enriched are rejected visibly (listed with a reason), never indexed
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from .config import Settings
from .ingest import IngestReport
from .openrouter import KVCache, OpenRouterClient, UpstreamError
from .schemas import TableData, Unit

log = logging.getLogger(__name__)

CAPTION_PROMPT_VERSION = "cap-v2"
CAPTION_PROMPT = """You are indexing a figure from a technical maintenance manual so engineers can find it by searching.
Describe it in about 100 words, plain text, no preamble. Include:
1. The system, assembly or component shown (be specific: e.g. "diesel engine cylinder block", "fuel control main flow body").
2. Every visible label, callout number, part number, reference designator and title text you can read.
3. The kind of drawing (exploded view, wiring diagram, schematic, cutaway, photo, chart) and what it is used for.
Do not guess numbers you cannot read.{hints}"""

TRANSCRIBE_PROMPT = """Transcribe this table image exactly as tab-separated values.
First line: the column headers. One line per table row. No commentary, no markdown fences."""

GENERIC_RE = re.compile(
    r"^(this is |the image shows |an? )?(technical )?(diagram|drawing|image|figure|illustration)\.?$", re.I)


def is_usable_caption(text: str | None) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty caption"
    t = text.strip()
    if len(t) < 60:
        return False, f"caption too short to retrieve on ({len(t)} chars)"
    if GENERIC_RE.match(t) or t.lower().startswith(("i cannot", "i can't", "sorry")):
        return False, "generic or refused caption"
    return True, "ok"


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ tables
def _norm_cell(c: str) -> str:
    return re.sub(r"\s+", " ", c or "").strip()


def linearise_table(t: TableData, caption: str | None = None, heading: str | None = None,
                    context: str | None = None, max_chars: int = 8000) -> str:
    """column=value rows for search. The structured TableData is untouched."""
    head = [h for h in [caption, heading, context] if h]
    cols = [_norm_cell(c) or f"col{i+1}" for i, c in enumerate(t.columns)]
    lines = []
    if head:
        lines.append(" | ".join(_norm_cell(h) for h in head))
    lines.append("Columns: " + ", ".join(cols))
    for r in t.rows:
        cells = [f"{c}={_norm_cell(v)}" for c, v in zip(cols, r) if _norm_cell(v)]
        if cells:
            lines.append("; ".join(cells))
    out = "\n".join(lines)
    return out[:max_chars]


def parse_tsv_text(text: str) -> TableData | None:
    text = re.sub(r"^```\w*|```$", "", text.strip(), flags=re.M).strip()
    rows = [[c.strip() for c in ln.split("\t")] for ln in text.splitlines() if ln.strip()]
    if len(rows) < 2 or len(rows[0]) < 2:
        return None
    w = len(rows[0])
    return TableData(columns=rows[0], rows=[(r + [""] * w)[:w] for r in rows[1:]])


# ------------------------------------------------------------------ figures
def caption_figure(unit: Unit, client: OpenRouterClient, cache: KVCache, settings: Settings) -> tuple[str, str, bool]:
    """Returns (caption, model, from_cache). Cached by image bytes + prompt version (not by model:
    a re-run with a fallback model must not pay twice)."""
    path = settings.data_dir / (unit.image_path or "")
    hints = []
    if unit.caption:
        hints.append(f'Title printed under the figure: "{unit.caption}".')
    if unit.meta.get("labels"):
        hints.append(f"Text found inside the figure area: {unit.meta['labels'][:300]}")
    hint_txt = ("\nHints from the page:\n" + "\n".join(hints)) if hints else ""
    key = KVCache.key(CAPTION_PROMPT_VERSION, image_hash(path), hint_txt)
    hit = cache.get("caption", key)
    if hit:
        return hit["caption"], hit["model"], True
    text, model = client.vision_with_fallback(CAPTION_PROMPT.format(hints=hint_txt), [path],
                                              role="caption", max_tokens=700)
    ok, _ = is_usable_caption(text)
    if ok:  # never cache an unusable caption: a re-run should try again
        cache.put("caption", key, {"caption": text, "model": model, "image": unit.image_path,
                                   "unit_id": unit.unit_id})
    return text, model, False


def transcribe_table(unit: Unit, client: OpenRouterClient, cache: KVCache, settings: Settings) -> TableData | None:
    path = settings.data_dir / (unit.image_path or "")
    key = KVCache.key("transcribe-v1", image_hash(path))
    hit = cache.get("table", key)
    if hit:
        return TableData.model_validate(hit)
    text, _ = client.vision_with_fallback(TRANSCRIBE_PROMPT, [path], role="caption", max_tokens=3000)
    data = parse_tsv_text(text)
    if data:
        cache.put("table", key, data.model_dump())
    return data


# ------------------------------------------------------------------ driver
def enrich_units(units: list[Unit], settings: Settings, report: IngestReport,
                 client: OpenRouterClient | None, cache: KVCache) -> list[Unit]:
    """Fill search_text for every unit; reject (visibly) any unit that cannot be made retrievable."""
    kept: list[Unit] = []
    captions_done = 0
    stats = {"captions_live": 0, "captions_cached": 0, "tables_transcribed": 0}
    figure_titles = {u.provenance.document_id: u for u in units if u.type == "figure"}

    for u in units:
        # provenance first: an unattributable unit is never worth paying to enrich
        missing = u.provenance.missing_fields()
        if missing:
            report.reject(u.unit_id, f"missing provenance: {', '.join(missing)}", "enrich")
            continue

        if u.type == "text":
            body = (u.text or "").strip()
            if len(body) < 40:
                report.reject(u.unit_id, "text too short to be useful", "enrich")
                continue
            u.search_text = (f"{u.heading}\n" if u.heading else "") + body

        elif u.type == "table":
            if u.table is None and u.meta.get("needs_transcription") and u.image_path:
                if client is None:
                    report.reject(u.unit_id, "table has no structured data and no model to transcribe it", "enrich")
                    continue
                try:
                    u.table = transcribe_table(u, client, cache, settings)
                    stats["tables_transcribed"] += 1
                except UpstreamError as e:
                    report.reject(u.unit_id, f"table transcription failed: {e}", "enrich")
                    continue
            if u.table is None or not u.table.rows:
                report.reject(u.unit_id, "table has no rows", "enrich")
                continue
            context = None
            if settings.link_tables_to_figures:
                fig = figure_titles.get(u.provenance.document_id)
                if fig is not None and (fig.caption or fig.meta.get("vision_caption")):
                    context = "Parts list for: " + (fig.caption or fig.meta["vision_caption"][:300])
            u.search_text = linearise_table(u.table, u.caption, u.heading, context)

        elif u.type == "figure":
            img = settings.data_dir / (u.image_path or "")
            if not u.image_path or not img.exists():
                report.reject(u.unit_id, "figure image missing on disk", "enrich")
                continue
            if client is None:
                report.reject(u.unit_id, "no vision model available to caption (no API key)", "enrich")
                continue
            if captions_done >= settings.max_captions:
                report.reject(u.unit_id, f"caption budget exhausted (MAX_CAPTIONS={settings.max_captions})", "enrich")
                continue
            try:
                cap, model, cached = caption_figure(u, client, cache, settings)
            except UpstreamError as e:
                report.reject(u.unit_id, f"captioning failed: {e}", "enrich")
                continue
            captions_done += 1
            stats["captions_cached" if cached else "captions_live"] += 1
            ok, why = is_usable_caption(cap)
            if not ok:
                report.reject(u.unit_id, f"unusable caption: {why}", "enrich", caption=cap[:200])
                continue
            u.meta["vision_caption"] = cap
            u.meta["caption_model"] = model
            u.search_text = (f"{u.caption}\n" if u.caption else "") + cap

        if not u.search_text.strip():
            report.reject(u.unit_id, "empty search text", "enrich")
            continue
        kept.append(u)

    report.notes.append(f"enrichment: {stats}")
    log.info("enrichment %s", stats)
    return kept

