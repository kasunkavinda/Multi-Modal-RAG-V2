"""Run the indexing pipeline: ingest > enrich > embed > upsert (fail closed).

    python scripts/build_index.py                       # full rebuild with config from rag/.env
    python scripts/build_index.py --ingest-only         # Stage 1: routing/OCR/extraction report, no model calls
    python scripts/build_index.py --inject-broken-unit  # E4.2: show a unit with no page being refused
    python scripts/build_index.py --corpora enginuity_t2,vidore
"""
import _bootstrap  # noqa: F401
import argparse
import json
import sys

from app.config import ConfigError, get_settings
from app.openrouter import UpstreamError
from app.pipeline import build_index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest-only", action="store_true")
    ap.add_argument("--no-reset", action="store_true", help="upsert into the existing collection")
    ap.add_argument("--inject-broken-unit", action="store_true")
    ap.add_argument("--corpora", help="comma list overriding CORPORA")
    a = ap.parse_args()
    s = get_settings()
    if a.corpora:
        s.corpora = [c.strip() for c in a.corpora.split(",") if c.strip()]
    try:
        out = build_index(s, reset=not a.no_reset, ingest_only=a.ingest_only,
                          inject_broken_unit=a.inject_broken_unit)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}\n  fix: {e.fix}")
        return 2
    except UpstreamError as e:
        print(f"UPSTREAM ERROR: {e}\n  The model provider failed; captions/embeddings done so far are cached, re-run to resume.")
        return 3

    if a.ingest_only:
        print("\n=== Routing report (Stage 1) ===")
        print("pages per class:", out["pages_by_class"])
        print("units extracted:", out["units_extracted"])
        print(f"OCR backend: {out['ocr']['backend']}  pages OCR'd: {out['ocr']['pages_ocrd']}  "
              f"review queue (not indexed): {out['review_queue']}")
        print(f"ingest time: {out['ingest_s']} s   details: data/reports/ingest_report.json, data/review_queue.json")
        return 0

    print("\n=== Index report ===")
    show = {k: out[k] for k in ("embedding", "vector_backend", "units_by_type", "units_extracted", "pages_by_class",
                                "review_queue", "ocr", "timings_s", "model_calls", "cache_hits", "index_size_bytes")}
    print(json.dumps(show, indent=1))
    print(f"rejected units: {len(out['rejected'])} by stage {out['rejected_by_stage']}")
    for r in out["rejected"][:15]:
        print(f"  - {r['unit_id']} [{r['stage']}]: {r['reason']}")
    if len(out["rejected"]) > 15:
        print(f"  ... see data/index_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
