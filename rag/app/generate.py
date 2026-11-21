"""Answer with the artefact (E6).

1. Drop hits whose provenance cannot name a page (or whose artefact is missing).
2. The top SOURCED hit decides the modality: figure / table / text.
3. Return the real artefact (image / exact rows / span) + its citation, and a short grounded
   answer (vision model for figures, text model for tables and text) that cites unit ids.
4. Nothing sourced left -> abstain and say why. Never invent a citation.
"""
from __future__ import annotations

import re
import time

from .config import Settings
from .openrouter import OpenRouterClient, UpstreamError
from .retrieve import tokenize
from .schemas import Artefact, Citation, Hit, QueryResponse, Source, Timings, Unit

NOT_IN_SOURCES = "NOT_IN_SOURCES"

SYSTEM_PROMPT = f"""You answer questions from field maintenance engineers using ONLY the numbered sources provided.
Rules:
- Answer in at most 80 words, plain text.
- Cite every claim with the source id in square brackets, e.g. [doc-1:p3:fig1]. Use only ids that appear in the sources.
- If the sources are relevant but do not fully settle the question, answer what they show and say what is missing.
- If the sources are unrelated to the question, reply exactly {NOT_IN_SOURCES}.
- Never invent part numbers, values or page numbers."""


def drop_reason(hit: Hit, settings: Settings) -> str | None:
    u = hit.unit
    missing = u.provenance.missing_fields()
    if missing:
        return f"no resolvable provenance ({', '.join(missing)})"
    if u.type == "figure":
        if not u.image_path or not (settings.data_dir / u.image_path).exists():
            return "figure artefact missing"
    elif u.type == "table":
        if u.table is None or not u.table.rows:
            return "table artefact missing"
    elif not (u.text or "").strip():
        return "text artefact missing"
    return None


def _preview(u: Unit) -> str:
    if u.type == "figure":
        return (u.caption + " - " if u.caption else "") + (u.meta.get("vision_caption") or u.search_text)[:220]
    return (u.search_text or "")[:220]


def highlight_rows(u: Unit, question: str, limit: int = 8) -> list[int]:
    """Rows of a table that share distinctive tokens with the question (item numbers, part names)."""
    if u.table is None:
        return []
    q = {t for t in tokenize(question) if len(t) > 2 or t.isdigit()}
    q -= {"the", "and", "for", "are", "this", "that", "with", "what", "which", "does", "item", "items", "part"}
    scored = []
    for i, row in enumerate(u.table.rows):
        toks = set(tokenize(" ".join(row)))
        s = len(q & toks)
        if s:
            scored.append((s, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return sorted(i for _, i in scored[:limit])


def context_block(u: Unit) -> str:
    if u.type == "table":
        body = u.search_text
    elif u.type == "figure":
        body = (f"Figure title: {u.caption}\n" if u.caption else "") + \
               f"Figure description: {u.meta.get('vision_caption') or u.search_text}"
    else:
        body = u.text or u.search_text
    return f"[{u.unit_id}] ({u.type}, {u.provenance.citation()})\n{body[:6000]}"


def build_artefact(u: Unit, question: str) -> Artefact:
    url = f"/artefact/{u.unit_id}"
    a = Artefact(type=u.type, unit_id=u.unit_id, url=url, caption=u.caption or u.heading)
    if u.type == "figure":
        a.image_url = url
    elif u.type == "table":
        a.table = u.table
        a.highlight_rows = highlight_rows(u, question)
        if u.image_path:
            a.image_url = url + "?view=image"
    else:
        a.text = u.text
    return a


def parse_citations(answer: str, allowed: set[str]) -> tuple[list[str], list[str]]:
    found = re.findall(r"\[([^\[\]]+?)\]", answer)
    ids: list[str] = []
    for f in found:
        for part in re.split(r"[,;]\s*", f):
            part = part.strip()
            if part and part not in ids:
                ids.append(part)
    good = [i for i in ids if i in allowed]
    bad = [i for i in ids if i not in allowed]
    return good, bad


def answer_question(question: str, hits: list[Hit], settings: Settings, client: OpenRouterClient | None,
                    timings: Timings, generate: bool = True) -> QueryResponse:
    t0 = time.perf_counter()
    calls_before = sum(client.stats.values()) if client else 0
    sources: list[Source] = []
    sourced: list[Hit] = []
    for h in hits:
        why = drop_reason(h, settings)
        p = h.unit.provenance
        sources.append(Source(rank=h.rank, unit_id=h.unit.unit_id, type=h.unit.type, score=round(h.score, 4),
                              vector_score=h.vector_score, bm25_score=h.bm25_score,
                              document_id=p.document_id, page=p.page,
                              citation=p.citation() if not why else f"{p.document_id or '?'} > p.{p.page or '?'} > {p.unit_id or '?'}",
                              sourced=why is None, drop_reason=why, preview=_preview(h.unit)))
        if why is None:
            sourced.append(h)

    resp = QueryResponse(question=question, status="abstained", sources=sources, timings=timings,
                         models={"text": settings.text_model, "figure": settings.figure_model,
                                 "embeddings": settings.embed_model})

    def finish() -> QueryResponse:
        timings.generation_ms = (time.perf_counter() - t0) * 1000
        timings.total_ms = timings.retrieval_ms + timings.generation_ms
        resp.model_calls = (sum(client.stats.values()) - calls_before) if client else 0
        return resp

    if not hits:
        resp.abstain_reason = "Nothing was retrieved for this question."
        return finish()
    if not sourced:
        resp.abstain_reason = (f"All {len(hits)} retrieved units were dropped because their provenance could not "
                               "name a page or their artefact is missing; refusing to answer without a source.")
        return finish()
    top = sourced[0]
    if top.score < settings.min_score:
        resp.abstain_reason = (f"Best sourced match scored {top.score:.3f}, below the MIN_SCORE threshold "
                               f"{settings.min_score:.3f}: nothing relevant enough was retrieved.")
        return finish()

    u = top.unit
    resp.answer_type = u.type
    resp.artefact = build_artefact(u, question)
    resp.citation = Citation(document_id=u.provenance.document_id, page=u.provenance.page or 0,
                             unit_id=u.unit_id, text=u.provenance.citation())
    ctx_units = [h.unit for h in sourced[: max(1, settings.context_units)]]
    used = {x.unit_id for x in ctx_units}
    for s in sources:
        s.used_in_context = s.unit_id in used

    if not generate or client is None:
        resp.status = "partial"
        resp.generation_error = "generation skipped" if not generate else "no model client (missing API key)"
        resp.cited_unit_ids = [u.unit_id]
        _mark_cited(resp)
        return finish()

    context = "\n\n".join(context_block(x) for x in ctx_units)
    prompt = f"Question: {question}\n\nSources:\n{context}\n\nAnswer with citations."
    try:
        if u.type == "figure":
            text = client.vision(prompt, [settings.data_dir / u.image_path], model=settings.figure_model,  # type: ignore[operator]
                                 role="figure_answer", max_tokens=500, system=SYSTEM_PROMPT +
                                 f"\nThe attached image IS source [{u.unit_id}]; look at it.")
        else:
            text = client.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                               model=settings.text_model, role="text_answer", max_tokens=500)
    except UpstreamError as e:
        # Retrieval is sourced, so the artefact + citation still stand; say generation failed.
        resp.status = "partial"
        resp.generation_error = f"Answer generation failed ({e}). The retrieved source is shown; verify it directly."
        resp.cited_unit_ids = [u.unit_id]
        _mark_cited(resp)
        return finish()

    if text.strip().upper().startswith(NOT_IN_SOURCES):
        resp.status = "abstained"
        resp.abstain_reason = ("The model judged the retrieved sources unrelated to the question. "
                               "Closest sources are listed for checking.")
        resp.artefact = None
        resp.citation = None
        resp.answer_type = None
        return finish()

    good, bad = parse_citations(text, used)
    if bad:
        for b in bad:  # never show an invented citation
            text = text.replace(f"[{b}]", "")
        resp.generation_error = f"Removed {len(bad)} citation(s) to ids that were not in the sources: {bad}"
    resp.answer = text.strip()
    resp.cited_unit_ids = good or [u.unit_id]
    resp.status = "answered"
    _mark_cited(resp)
    return finish()


def _mark_cited(resp: QueryResponse) -> None:
    cited = set(resp.cited_unit_ids)
    for s in resp.sources:
        s.cited = s.unit_id in cited
