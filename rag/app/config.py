"""Single source of truth for configuration. Nothing else in the service reads the environment.

Values come from the process environment, falling back to rag/.env (never committed).
A missing OpenRouter key does not break imports (ingestion dry-runs and tests work offline),
but every remote call goes through `Settings.require_key()`, which stops the run first.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

RAG_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(RuntimeError):
    """A problem the operator fixes in configuration (key, slug, backend, stale index)."""

    def __init__(self, message: str, fix: str = ""):
        super().__init__(message)
        self.fix = fix


def _clean(value: str | None) -> str:
    # Strip whitespace and accidental quotes: a classic "my key does not work" cause.
    return (value or "").strip().strip('"').strip("'").strip()


def _list(value: str | None) -> list[str]:
    return [v.strip() for v in _clean(value).split(",") if v.strip()]


def _bool(value: str | None, default: bool = False) -> bool:
    v = _clean(value).lower()
    return default if not v else v in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int | None) -> int | None:
    v = _clean(value)
    return int(v) if v else default


def _float(value: str | None, default: float) -> float:
    v = _clean(value)
    return float(v) if v else default


class Settings(BaseModel):
    # --- OpenRouter -------------------------------------------------------
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    requests_per_minute: int = 20          # free-tier cap per model
    max_retries: int = 4
    request_timeout_s: float = 120.0

    # --- Model slugs (all from config; swapping is a config change only) --
    embed_model: str = "nvidia/nemotron-3.5-embed-1b:free"
    embed_dimensions: int | None = 1024    # client-side Matryoshka truncation; None = full size
    embed_images: bool = False             # E3.4: embed figure images instead of captions
    embed_batch_size: int = 32             # one batch = one call; brief suggests 16, 32 halves the calls
    embed_query_prefix: str = ""
    embed_doc_prefix: str = ""
    vision_model: str = "nvidia/nemotron-3.5-super-vl:free"
    vision_fallbacks: list[str] = ["google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free"]
    text_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    figure_model: str = "nvidia/nemotron-3.5-super-vl:free"
    judge_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    reasoning_off: bool = True

    # --- Vector store -----------------------------------------------------
    vector_backend: Literal["qdrant", "memory"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    collection: str = "manual_units"

    # --- Corpus selection ---------------------------------------------------
    corpora: list[str] = ["enginuity_t2"]
    enginuity_pdfs: list[str] = ["data-7.pdf"]
    vidore_pdfs: list[str] = ["TO_00-25-172CL-1.pdf", "TO_00-25-172CL-3.pdf", "TO_00-25-172CL-4.pdf",
                              "TO_00-25-172CL-5.pdf"]
    max_pages_per_pdf: int | None = None
    max_captions: int = 80                 # budget guard: figures beyond this are rejected visibly
    link_tables_to_figures: bool = False   # prepend the same document's figure title to table search text

    # --- Routing / OCR gate -----------------------------------------------
    ocr_backend: Literal["auto", "tesseract", "vision", "none"] = "auto"
    ocr_min_confidence: float = 60.0       # tesseract mean word confidence (0-100)
    ocr_max_garble: float = 0.35           # share of garbled tokens allowed
    max_vision_ocr_pages: int = 10         # budget guard when OCR runs through the vision model
    min_figure_px: int = 120               # ignore rules/icons smaller than this on either side
    min_figure_area: float = 0.02          # ... or smaller than this share of the page

    # --- Retrieval / answering --------------------------------------------
    top_k: int = 5
    min_score: float = 0.0                 # abstain if the best sourced hit scores below this
    fusion: bool = False                   # E5.2: BM25 + vector fusion
    bm25_weight: float = 0.3
    context_units: int = 4                 # sourced units passed to the generator
    figure_max_side: int = 1600            # downscale before sending to a vision model

    # --- Service ----------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:3000"]
    data_dir: Path = RAG_ROOT / "data"

    # ---------------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def artefact_dir(self) -> Path:
        return self.data_dir / "artefacts"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def index_report_path(self) -> Path:
        return self.data_dir / "index_report.json"

    @property
    def qa_path(self) -> Path:
        return self.data_dir / "qa.json"

    @property
    def model_check_path(self) -> Path:
        return self.data_dir / "model_check.json"

    def require_key(self) -> str:
        if not self.openrouter_api_key:
            raise ConfigError(
                "OPENROUTER_API_KEY is not set.",
                fix="Create a key at https://openrouter.ai/keys and put it in rag/.env (see rag/.env.example).",
            )
        return self.openrouter_api_key

    def model_roles(self) -> dict[str, str]:
        return {
            "embeddings": self.embed_model,
            "vision": self.vision_model,
            "text": self.text_model,
            "figure": self.figure_model,
        }


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or RAG_ROOT / ".env", override=False)
    e = os.environ.get
    d = Settings()
    s = Settings(
        openrouter_api_key=_clean(e("OPENROUTER_API_KEY")),
        openrouter_base_url=_clean(e("OPENROUTER_BASE_URL")) or d.openrouter_base_url,
        requests_per_minute=_int(e("REQUESTS_PER_MINUTE"), d.requests_per_minute),
        max_retries=_int(e("MAX_RETRIES"), d.max_retries),
        request_timeout_s=_float(e("REQUEST_TIMEOUT_S"), d.request_timeout_s),
        embed_model=_clean(e("EMBED_MODEL")) or d.embed_model,
        embed_dimensions=_int(e("EMBED_DIMENSIONS"), d.embed_dimensions) or None,
        embed_images=_bool(e("EMBED_IMAGES"), d.embed_images),
        embed_batch_size=_int(e("EMBED_BATCH_SIZE"), d.embed_batch_size),
        embed_query_prefix=e("EMBED_QUERY_PREFIX", d.embed_query_prefix),
        embed_doc_prefix=e("EMBED_DOC_PREFIX", d.embed_doc_prefix),
        vision_model=_clean(e("VISION_MODEL")) or d.vision_model,
        vision_fallbacks=_list(e("VISION_FALLBACKS")) if e("VISION_FALLBACKS") is not None else d.vision_fallbacks,
        text_model=_clean(e("TEXT_MODEL")) or d.text_model,
        figure_model=_clean(e("FIGURE_MODEL")) or d.figure_model,
        judge_model=_clean(e("JUDGE_MODEL")) or _clean(e("TEXT_MODEL")) or d.judge_model,
        reasoning_off=_bool(e("REASONING_OFF"), d.reasoning_off),
        vector_backend=(_clean(e("VECTOR_BACKEND")) or d.vector_backend).lower(),  # type: ignore[arg-type]
        qdrant_url=_clean(e("QDRANT_URL")) or d.qdrant_url,
        collection=_clean(e("QDRANT_COLLECTION")) or d.collection,
        corpora=_list(e("CORPORA")) or d.corpora,
        enginuity_pdfs=_list(e("ENGINUITY_PDFS")) or d.enginuity_pdfs,
        vidore_pdfs=_list(e("VIDORE_PDFS")) or d.vidore_pdfs,
        max_pages_per_pdf=_int(e("MAX_PAGES_PER_PDF"), d.max_pages_per_pdf),
        max_captions=_int(e("MAX_CAPTIONS"), d.max_captions),
        link_tables_to_figures=_bool(e("LINK_TABLE_TO_FIGURE"), d.link_tables_to_figures),
        ocr_backend=(_clean(e("OCR_BACKEND")) or d.ocr_backend).lower(),  # type: ignore[arg-type]
        ocr_min_confidence=_float(e("OCR_MIN_CONFIDENCE"), d.ocr_min_confidence),
        ocr_max_garble=_float(e("OCR_MAX_GARBLE"), d.ocr_max_garble),
        max_vision_ocr_pages=_int(e("MAX_VISION_OCR_PAGES"), d.max_vision_ocr_pages),
        top_k=_int(e("TOP_K"), d.top_k),
        min_score=_float(e("MIN_SCORE"), d.min_score),
        fusion=_bool(e("FUSION"), d.fusion),
        bm25_weight=_float(e("BM25_WEIGHT"), d.bm25_weight),
        context_units=_int(e("CONTEXT_UNITS"), d.context_units),
        cors_origins=_list(e("CORS_ORIGINS")) or d.cors_origins,
        data_dir=Path(_clean(e("DATA_DIR"))) if _clean(e("DATA_DIR")) else d.data_dir,
    )
    if s.vector_backend not in ("qdrant", "memory"):
        raise ConfigError(f"VECTOR_BACKEND must be 'qdrant' or 'memory', got {s.vector_backend!r}")
    return s


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
