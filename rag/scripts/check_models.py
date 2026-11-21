"""Stage 0 / E0.1: confirm each configured model slug is live (one tiny call per role).

    python scripts/check_models.py            # embeddings, vision, text, figure
Prints OK/FAIL with model name and reason; writes data/model_check.json (template T1 rows).
"""
import _bootstrap  # noqa: F401
import sys

from app.config import ConfigError, get_settings
from app.diagnostics import check_models


def main() -> int:
    s = get_settings()
    try:
        out = check_models(s)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}\n  fix: {e.fix}")
        return 2
    print(f"{'Role':<12} {'Slug':<48} {'OK?':<5} {'ms':>6}  Notes / error")
    for r in out["results"]:
        print(f"{r['role']:<12} {r['model']:<48} {'OK' if r['ok'] else 'FAIL':<5} {r['ms']:>6}  {r['notes']}")
    print(f"\nall ok: {out['all_ok']}  (saved to {s.model_check_path})")
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
