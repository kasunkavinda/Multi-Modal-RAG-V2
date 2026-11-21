"""HTTP API (E5.3): POST /query, GET /health, GET /examples, GET /artefact/{unit_id}.

Configuration problems -> 503 {"error": {"kind": "configuration", ...}}
Upstream model problems -> 502 {"error": {"kind": "upstream", "model": ...}}
Artefact downloads are confined to the data folder.

Run:  uvicorn app.server:app --port 8000   (from rag/)
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import ConfigError, Settings, get_settings
from .diagnostics import check_models, last_model_check
from .openrouter import UpstreamError
from .pipeline import RagService
from .retrieve import check_index, load_index_report
from .schemas import Health, Problem, QueryRequest, QueryResponse
from .store import get_store, load_catalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def safe_data_path(settings: Settings, relative: str) -> Path:
    """Resolve a stored artefact path and refuse anything outside the data folder."""
    base = settings.data_dir.resolve()
    p = (base / relative).resolve()
    if not p.is_relative_to(base):
        raise HTTPException(status_code=403, detail="artefact path outside the data folder")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="artefact file not found")
    return p


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(title="Multimodal manual RAG", version="2.0")
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])
    service = RagService(s)
    app.state.settings = s
    app.state.service = service

    @app.exception_handler(ConfigError)
    async def _config_error(_: Request, e: ConfigError):
        return JSONResponse(status_code=503, content={"error": {"kind": "configuration", "message": str(e),
                                                                "fix": e.fix}})

    @app.exception_handler(UpstreamError)
    async def _upstream_error(_: Request, e: UpstreamError):
        return JSONResponse(status_code=502, content={"error": {
            "kind": "upstream", "model": e.model, "message": e.reason,
            "fix": "The model provider failed. Retry shortly, or switch the slug in rag/.env (see /health)."}})

    @app.get("/health", response_model=Health)
    def health(live: bool = False) -> Health:
        problems: list[Problem] = []
        if not s.openrouter_api_key:
            problems.append(Problem(kind="configuration", message="OPENROUTER_API_KEY is not set",
                                    fix="Add it to rag/.env and restart the API."))
        report = load_index_report(s)
        try:
            check_index(s, report)
        except ConfigError as e:
            problems.append(Problem(kind="index", message=str(e), fix=e.fix))
        try:
            get_store(s).ping()
        except ConfigError as e:
            problems.append(Problem(kind="vector_store", message=str(e), fix=e.fix))
        mc = None
        if live and s.openrouter_api_key:
            try:
                mc = check_models(s)
            except ConfigError as e:
                problems.append(Problem(kind="configuration", message=str(e), fix=e.fix))
        else:
            mc = last_model_check(s)
        if mc:
            configured = s.model_roles()
            for r in mc.get("results", []):
                if r.get("model") != configured.get(r.get("role")):
                    continue  # stale result for a slug no longer configured
                if not r.get("ok"):
                    problems.append(Problem(kind="upstream", message=f"{r['role']} model {r['model']} failed: {r['notes']}",
                                            fix="Pick another slug from section 6 and set it in rag/.env."))
            stale = [role for role, m in configured.items()
                     if not any(r.get("role") == role and r.get("model") == m for r in mc.get("results", []))]
            if stale and s.openrouter_api_key:
                problems.append(Problem(kind="upstream", message=f"models not yet checked: {', '.join(stale)}",
                                        fix="Run `python scripts/check_models.py` or GET /health?live=true."))
        idx = None
        if report:
            idx = {k: report.get(k) for k in ("built_at", "embedding", "units_by_type", "units_total", "corpora",
                                              "vector_backend")}
            idx["rejected"] = len(report.get("rejected", []))
            idx["review_queue"] = report.get("review_queue", {}).get("count")
        return Health(status="healthy" if not problems else "degraded", problems=problems,
                      vector_backend=s.vector_backend, index=idx, models=s.model_roles(), model_check=mc)

    @app.get("/examples")
    def examples() -> dict:
        p = s.data_dir / "examples.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        out = []
        if s.qa_path.exists():
            qa = json.loads(s.qa_path.read_text(encoding="utf-8"))
            rng = random.Random(7)
            for t in ("figure", "table", "text"):
                pool = [q for q in qa if q.get("qtype") == t]
                if pool:
                    q = rng.choice(pool)
                    out.append({"question": q["question"], "type": t, "qid": q["qid"]})
        if not out:
            out = [{"question": "Show the cylinder block assembly diagram", "type": "figure"},
                   {"question": "What is the part number for the engine block plug?", "type": "table"},
                   {"question": "What does the manual say about servicing aircraft?", "type": "text"}]
        return {"examples": out}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        return service.query(req.question.strip(), req.k, list(req.types) if req.types else None)

    @app.get("/artefact/{unit_id:path}")
    def artefact(unit_id: str, view: str | None = None):
        cat = service._retriever.catalog if service._retriever else load_catalog(s)
        u = cat.get(unit_id)
        if u is None:
            raise HTTPException(status_code=404, detail=f"unknown unit {unit_id}")
        if u.provenance.missing_fields():
            raise HTTPException(status_code=409, detail="unit has no resolvable provenance; not served")
        prov = u.provenance.model_dump()
        if u.type == "figure" or (u.type == "table" and view == "image" and u.image_path):
            return FileResponse(safe_data_path(s, u.image_path or ""),
                                headers={"X-Citation": u.provenance.citation()})
        if u.type == "table":
            return {"unit_id": u.unit_id, "type": "table", "table": u.table.model_dump() if u.table else None,
                    "caption": u.caption, "provenance": prov}
        return {"unit_id": u.unit_id, "type": "text", "text": u.text, "heading": u.heading, "provenance": prov}

    @app.get("/ui", include_in_schema=False)
    def ui():
        """Dependency-free fallback UI (same features as web/), for machines without npm access."""
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.post("/admin/reload")
    def reload() -> dict:
        service.reload()
        return {"reloaded": True}

    return app


app = create_app()
