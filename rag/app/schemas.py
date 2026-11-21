"""Data shapes: the Unit (search_text vs artefact, tied by provenance) and the API types."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

UnitType = Literal["text", "figure", "table"]
UNIT_TYPES: tuple[UnitType, ...] = ("text", "figure", "table")


class Provenance(BaseModel):
    """document > page > unit id (+ optional bounding box and source path)."""

    document_id: str = ""
    page: int | None = None          # 1-based
    unit_id: str = ""
    bbox: list[float] | None = None  # PDF points, x0 y0 x1 y1
    source_path: str | None = None   # relative to the data folder

    def missing_fields(self) -> list[str]:
        missing = []
        if not (self.document_id or "").strip():
            missing.append("document_id")
        if not isinstance(self.page, int) or isinstance(self.page, bool) or self.page < 1:
            missing.append("page")
        if not (self.unit_id or "").strip():
            missing.append("unit_id")
        return missing

    def is_resolvable(self) -> bool:
        return not self.missing_fields()

    def citation(self) -> str:
        return f"{self.document_id} > p.{self.page} > {self.unit_id}"


class TableData(BaseModel):
    """The structured artefact for a table: never flattened."""

    columns: list[str]
    rows: list[list[str]]


class Unit(BaseModel):
    unit_id: str
    type: UnitType
    # What we EMBED (filled by enrich): prose chunk / vision caption / linearised table.
    search_text: str = ""
    # What we RETURN (one of these depending on type). Paths are relative to the data folder.
    text: str | None = None
    image_path: str | None = None
    table: TableData | None = None
    # Context found next to the artefact in the document (figure title, table caption, heading).
    caption: str | None = None
    heading: str | None = None
    provenance: Provenance
    meta: dict[str, Any] = Field(default_factory=dict)


class Hit(BaseModel):
    unit: Unit
    rank: int
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None

    @property
    def citation(self) -> str:
        return self.unit.provenance.citation()


# ---------------------------------------------------------------- API types
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)
    types: list[UnitType] | None = None


class Citation(BaseModel):
    document_id: str
    page: int
    unit_id: str
    text: str  # "document > p.N > unit"


class Artefact(BaseModel):
    type: UnitType
    unit_id: str
    url: str                          # GET /artefact/{unit_id}
    image_url: str | None = None      # figures
    table: TableData | None = None    # tables: exact rows
    highlight_rows: list[int] = []    # row indexes that match the question
    text: str | None = None           # text spans
    caption: str | None = None


class Source(BaseModel):
    rank: int
    unit_id: str
    type: UnitType
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    document_id: str
    page: int | None
    citation: str
    sourced: bool                     # False = dropped (no resolvable provenance)
    cited: bool = False
    used_in_context: bool = False
    preview: str = ""
    drop_reason: str | None = None


class Timings(BaseModel):
    embed_ms: float = 0
    search_ms: float = 0
    retrieval_ms: float = 0
    generation_ms: float = 0
    total_ms: float = 0


class QueryResponse(BaseModel):
    question: str
    status: Literal["answered", "partial", "abstained"]
    answer_type: UnitType | None = None
    answer: str | None = None
    abstain_reason: str | None = None
    generation_error: str | None = None
    artefact: Artefact | None = None
    citation: Citation | None = None
    cited_unit_ids: list[str] = []
    sources: list[Source] = []
    timings: Timings = Timings()
    model_calls: int = 0
    models: dict[str, str] = {}


class Problem(BaseModel):
    kind: Literal["configuration", "upstream", "index", "vector_store"]
    message: str
    fix: str = ""


class Health(BaseModel):
    status: Literal["healthy", "degraded"]
    problems: list[Problem] = []
    vector_backend: str
    index: dict[str, Any] | None = None
    models: dict[str, str] = {}
    model_check: dict[str, Any] | None = None
