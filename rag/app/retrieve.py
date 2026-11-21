"""Retrieval (E5.1, E5.2): embed the query, search with an optional type filter, check the
index was built with the same embedding model, optionally fuse BM25 for exact identifiers.
No generation here: every hit carries its citation and can be measured on its own."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from .config import ConfigError, Settings
from .openrouter import OpenRouterClient
from .schemas import Hit, Provenance, Unit
from .store import StoredHit, VectorStore, get_store, load_catalog


def figure_mode(settings: Settings) -> str:
    return "image" if settings.embed_images else "caption"


def load_index_report(settings: Settings) -> dict | None:
    p = settings.index_report_path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def check_index(settings: Settings, report: dict | None = None) -> dict:
    """Refuse to query an index built with a different embedder, dimension, mode or backend."""
    report = report if report is not None else load_index_report(settings)
    rebuild = "Run `python scripts/build_index.py` to rebuild the index with the current configuration."
    if not report:
        raise ConfigError("No index found (data/index_report.json missing).", fix=rebuild)
    built = report.get("embedding", {})
    problems = []
    if built.get("model") != settings.embed_model:
        problems.append(f"index built with EMBED_MODEL={built.get('model')!r}, configured {settings.embed_model!r}")
    if built.get("dimensions") != settings.embed_dimensions:
        problems.append(f"index built with EMBED_DIMENSIONS={built.get('dimensions')}, configured {settings.embed_dimensions}")
    if built.get("figure_mode") != figure_mode(settings):
        problems.append(f"index built in figure mode {built.get('figure_mode')!r}, configured {figure_mode(settings)!r}")
    if report.get("vector_backend") != settings.vector_backend:
        problems.append(f"index built on backend {report.get('vector_backend')!r}, configured {settings.vector_backend!r}")
    if problems:
        raise ConfigError("Embedding/index mismatch: " + "; ".join(problems) + ".", fix=rebuild)
    return report


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; codes like 5340-00-530-5543 are kept whole AND split into parts."""
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        out.append(tok)
        if any(c in tok for c in "-/."):
            compact = re.sub(r"[-/.]", "", tok)
            out.append(compact)
            out.extend(p for p in re.split(r"[-/.]", tok) if len(p) > 1)
    return out


@dataclass
class RetrievalResult:
    hits: list[Hit]
    embed_ms: float
    search_ms: float


class Retriever:
    def __init__(self, settings: Settings, client: OpenRouterClient, store: VectorStore | None = None):
        self.s = settings
        self.report = check_index(settings)
        self.client = client
        self.store = store or get_store(settings)
        self.catalog = load_catalog(settings)
        self._bm25 = None
        self._bm25_ids: list[str] = []

    # --------------------------------------------------------------- bm25
    def _bm25_index(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._bm25_ids = list(self.catalog)
            corpus = [tokenize(self.catalog[i].search_text) for i in self._bm25_ids]
            self._bm25 = BM25Okapi(corpus or [[""]])
        return self._bm25

    # ------------------------------------------------------------- search
    def _hydrate(self, h: StoredHit) -> Unit:
        u = self.catalog.get(h.unit_id)
        if u is not None:
            return u
        # Unit missing from the catalog: rebuild from the payload. Provenance comes from the DB,
        # and there is no artefact, so the answer step cannot present it as a figure/table.
        p = h.payload
        return Unit(unit_id=p.get("unit_id", ""), type=p.get("type", "text"), search_text=p.get("search_text", ""),
                    provenance=Provenance(document_id=p.get("document_id") or "", page=p.get("page"),
                                          unit_id=p.get("unit_id") or "", source_path=p.get("source_path")),
                    meta={"hydrated_from_payload": True})

    def search_vector(self, qvec: list[float], question: str, k: int, types: list[str] | None) -> list[Hit]:
        if not self.s.fusion:
            stored = self.store.search(qvec, k, types)
            return [Hit(unit=self._hydrate(h), rank=i + 1, score=h.score, vector_score=h.score)
                    for i, h in enumerate(stored)]

        # Fusion: candidate pool from both rankers, min-max normalised, weighted sum.
        pool_k = max(k * 4, 20)
        stored = {h.unit_id: h for h in self.store.search(qvec, pool_k, types)}
        bm = self._bm25_index()
        raw = bm.get_scores(tokenize(question))
        order = sorted(range(len(raw)), key=lambda i: -raw[i])
        bm_scores: dict[str, float] = {}
        for i in order:
            uid = self._bm25_ids[i]
            if types and self.catalog[uid].type not in types:
                continue
            if raw[i] <= 0 or len(bm_scores) >= pool_k:
                break
            bm_scores[uid] = float(raw[i])
        ids = set(stored) | set(bm_scores)
        if not ids:
            return []

        def norm(d: dict[str, float]) -> dict[str, float]:
            if not d:
                return {}
            lo, hi = min(d.values()), max(d.values())
            return {i: (v - lo) / (hi - lo) if hi > lo else 1.0 for i, v in d.items()}

        vs = norm({i: h.score for i, h in stored.items()})
        bs = norm(bm_scores)
        w = self.s.bm25_weight
        fused = {i: (1 - w) * vs.get(i, 0.0) + w * bs.get(i, 0.0) for i in ids}
        ranked = sorted(fused, key=lambda i: -fused[i])[:k]
        hits = []
        for r, uid in enumerate(ranked, 1):
            if uid in stored:
                unit = self._hydrate(stored[uid])
            elif uid in self.catalog:
                unit = self.catalog[uid]
            else:
                continue
            hits.append(Hit(unit=unit, rank=r, score=fused[uid],
                            vector_score=stored[uid].score if uid in stored else None,
                            bm25_score=bm_scores.get(uid)))
        return hits

    def retrieve(self, question: str, k: int | None = None, types: list[str] | None = None) -> RetrievalResult:
        k = k or self.s.top_k
        t0 = time.perf_counter()
        qvec = self.client.embed_texts([question], is_query=True, role="query_embed")[0]
        t1 = time.perf_counter()
        hits = self.search_vector(qvec, question, k, types)
        t2 = time.perf_counter()
        return RetrievalResult(hits, (t1 - t0) * 1000, (t2 - t1) * 1000)

    def retrieve_batch(self, questions: list[str], k: int | None = None,
                       types: list[str] | None = None) -> list[RetrievalResult]:
        """Evaluation path: embed all questions in batches (not one by one)."""
        k = k or self.s.top_k
        t0 = time.perf_counter()
        vecs = self.client.embed_texts(questions, is_query=True, role="query_embed")
        per_q_embed = (time.perf_counter() - t0) * 1000 / max(1, len(questions))
        out = []
        for q, v in zip(questions, vecs):
            t1 = time.perf_counter()
            hits = self.search_vector(v, q, k, types)
            out.append(RetrievalResult(hits, per_q_embed, (time.perf_counter() - t1) * 1000))
        return out
