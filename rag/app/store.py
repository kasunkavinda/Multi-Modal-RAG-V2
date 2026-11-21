"""Vector-store interface with two backends (Qdrant, in-memory) (E4).

Provenance is enforced at WRITE time: a unit missing document, page or unit id is refused
and logged, never stored. Payload carries document, page, unit id and type (filterable).
"""
from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ConfigError, Settings
from .schemas import Unit

log = logging.getLogger(__name__)
NS = uuid.UUID("6f1c3e2a-6b1d-4f7e-9a51-3d2c1b0a9f88")


@dataclass
class StoredHit:
    unit_id: str
    score: float
    payload: dict


class ProvenanceRejected(ValueError):
    pass


def payload_for(u: Unit) -> dict:
    p = u.provenance
    return {
        "unit_id": u.unit_id,
        "type": u.type,
        "document_id": p.document_id,
        "page": p.page,
        "citation": p.citation(),
        "source_path": p.source_path,
        "bbox": p.bbox,
        "search_text": u.search_text[:2000],
        "corpus": u.meta.get("corpus"),
    }


def validate_for_write(u: Unit) -> str | None:
    """Fail closed: return a rejection reason, or None if the unit may be stored."""
    missing = u.provenance.missing_fields()
    if missing:
        return f"missing provenance: {', '.join(missing)}"
    if u.provenance.unit_id != u.unit_id:
        return f"provenance unit id {u.provenance.unit_id!r} != unit id {u.unit_id!r}"
    if not u.search_text.strip():
        return "empty search text"
    return None


class VectorStore(ABC):
    backend = "abstract"

    def __init__(self, settings: Settings):
        self.s = settings
        self.rejections: list[dict] = []

    def write(self, units: list[Unit], vectors: list[list[float]]) -> list[dict]:
        """Validate every unit, upsert the valid ones, return (and log) the rejected ones."""
        if len(units) != len(vectors):
            raise ValueError("units and vectors differ in length")
        ok_u, ok_v, rejected = [], [], []
        for u, v in zip(units, vectors):
            reason = validate_for_write(u)
            if reason:
                rec = {"unit_id": u.unit_id or "<none>", "stage": "upsert", "reason": reason}
                log.warning("REJECTED at write: %s (%s)", rec["unit_id"], reason)
                rejected.append(rec)
            else:
                ok_u.append(u)
                ok_v.append(v)
        if ok_u:
            self._upsert(ok_u, ok_v)
        self.rejections.extend(rejected)
        return rejected

    @abstractmethod
    def reset(self, dim: int) -> None: ...

    @abstractmethod
    def _upsert(self, units: list[Unit], vectors: list[list[float]]) -> None: ...

    @abstractmethod
    def search(self, vector: list[float], k: int, types: list[str] | None = None) -> list[StoredHit]: ...

    @abstractmethod
    def count(self) -> dict[str, int]: ...

    @abstractmethod
    def ping(self) -> None: ...

    def size_bytes(self) -> int | None:
        return None


# ------------------------------------------------------------------ memory
class MemoryStore(VectorStore):
    """In-process cosine index persisted to data/index/ (numpy + json). No Docker needed."""

    backend = "memory"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.dir = settings.index_dir
        self.vec_path = self.dir / "memory_vectors.npy"
        self.pay_path = self.dir / "memory_payloads.json"
        self.ids: list[str] = []
        self.payloads: list[dict] = []
        self.mat = np.zeros((0, 0), dtype=np.float32)
        if self.vec_path.exists() and self.pay_path.exists():
            self.mat = np.load(self.vec_path)
            self.payloads = json.loads(self.pay_path.read_text(encoding="utf-8"))
            self.ids = [p["unit_id"] for p in self.payloads]

    def reset(self, dim: int) -> None:
        self.ids, self.payloads = [], []
        self.mat = np.zeros((0, dim), dtype=np.float32)
        self._save()

    def _upsert(self, units, vectors) -> None:
        v = np.asarray(vectors, dtype=np.float32)
        v /= np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
        index = {uid: i for i, uid in enumerate(self.ids)}
        if self.mat.size == 0:
            self.mat = np.zeros((0, v.shape[1]), dtype=np.float32)
        for u, row in zip(units, v):
            if u.unit_id in index:
                i = index[u.unit_id]
                self.mat[i] = row
                self.payloads[i] = payload_for(u)
            else:
                self.mat = np.vstack([self.mat, row[None, :]])
                self.ids.append(u.unit_id)
                self.payloads.append(payload_for(u))
                index[u.unit_id] = len(self.ids) - 1
        self._save()

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.vec_path, self.mat)
        self.pay_path.write_text(json.dumps(self.payloads), encoding="utf-8")

    def search(self, vector, k, types=None) -> list[StoredHit]:
        if not self.ids:
            return []
        q = np.asarray(vector, dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        if q.shape[0] != self.mat.shape[1]:
            raise ConfigError(f"query vector has {q.shape[0]} dims but the index has {self.mat.shape[1]}",
                              fix="Rebuild the index with the configured embedding model and dimensions.")
        scores = self.mat @ q
        if types:
            mask = np.array([p["type"] in types for p in self.payloads])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [StoredHit(self.ids[i], float(scores[i]), self.payloads[i]) for i in order if np.isfinite(scores[i])]

    def count(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.payloads:
            out[p["type"]] = out.get(p["type"], 0) + 1
        return out

    def ping(self) -> None:
        return None

    def size_bytes(self) -> int | None:
        return sum(p.stat().st_size for p in (self.vec_path, self.pay_path) if p.exists())


# ------------------------------------------------------------------ qdrant
class QdrantStore(VectorStore):
    backend = "qdrant"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=settings.qdrant_url, timeout=30, check_compatibility=False)
        self.collection = settings.collection

    def ping(self) -> None:
        try:
            self.client.get_collections()
        except Exception as e:
            raise ConfigError(f"Qdrant not reachable at {self.s.qdrant_url}: {e.__class__.__name__}",
                              fix="Run `docker compose up -d qdrant` or set VECTOR_BACKEND=memory.") from e

    def reset(self, dim: int) -> None:
        from qdrant_client import models as qm

        self.ping()
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE)
        )
        for field in ("type", "document_id", "unit_id"):
            self.client.create_payload_index(self.collection, field, qm.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection, "page", qm.PayloadSchemaType.INTEGER)

    def _upsert(self, units, vectors) -> None:
        from qdrant_client import models as qm

        pts = [qm.PointStruct(id=str(uuid.uuid5(NS, u.unit_id)), vector=list(map(float, v)), payload=payload_for(u))
               for u, v in zip(units, vectors)]
        for i in range(0, len(pts), 128):
            self.client.upsert(self.collection, points=pts[i : i + 128], wait=True)

    def search(self, vector, k, types=None) -> list[StoredHit]:
        from qdrant_client import models as qm

        flt = None
        if types:
            flt = qm.Filter(must=[qm.FieldCondition(key="type", match=qm.MatchAny(any=list(types)))])
        try:
            res = self.client.query_points(self.collection, query=list(map(float, vector)), limit=k,
                                           query_filter=flt, with_payload=True)
        except Exception as e:
            msg = str(e)
            if "dimension" in msg.lower():
                raise ConfigError(f"vector size mismatch with Qdrant collection: {msg[:200]}",
                                  fix="Rebuild the index with the configured embedding model and dimensions.") from e
            raise ConfigError(f"Qdrant search failed: {msg[:200]}",
                              fix="Check Qdrant is running and the index was built (scripts/build_index.py).") from e
        return [StoredHit(p.payload["unit_id"], float(p.score), p.payload) for p in res.points]

    def count(self) -> dict[str, int]:
        from qdrant_client import models as qm

        out = {}
        if not self.client.collection_exists(self.collection):
            return out
        for t in ("text", "figure", "table"):
            out[t] = self.client.count(
                self.collection, count_filter=qm.Filter(must=[qm.FieldCondition(key="type", match=qm.MatchValue(value=t))]),
                exact=True).count
        return out

    def size_bytes(self) -> int | None:
        try:
            info = self.client.get_collection(self.collection)
            dim = info.config.params.vectors.size  # type: ignore[union-attr]
            return int((info.points_count or 0) * dim * 4)
        except Exception:
            return None


def get_store(settings: Settings) -> VectorStore:
    if settings.vector_backend == "qdrant":
        return QdrantStore(settings)
    return MemoryStore(settings)


# ------------------------------------------------------------ unit catalog
def catalog_path(settings: Settings) -> Path:
    return settings.index_dir / "units.jsonl"


def save_catalog(settings: Settings, units: list[Unit]) -> None:
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    with catalog_path(settings).open("w", encoding="utf-8") as fh:
        for u in units:
            fh.write(u.model_dump_json() + "\n")


def load_catalog(settings: Settings) -> dict[str, Unit]:
    p = catalog_path(settings)
    if not p.exists():
        return {}
    out = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                u = Unit.model_validate_json(line)
                out[u.unit_id] = u
    return out
