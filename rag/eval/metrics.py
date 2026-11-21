"""Retrieval and grounding metrics. Pure functions, no network (unit-tested).

Relevance: a hit is relevant to a gold target when
  - the target names a unit id: the hit IS that unit (Enginuity), or
  - the target names (document, page, type): the hit is on that page AND of that type (ViDoRe).
Matching only the page while the unit type is wrong does NOT count (counted separately as page_only).
"""
from __future__ import annotations

from statistics import mean


def is_relevant(hit: dict, target: dict) -> bool:
    if target.get("unit_id"):
        return hit.get("unit_id") == target["unit_id"]
    return (hit.get("document_id") == target.get("document_id") and hit.get("page") == target.get("page")
            and hit.get("type") == target.get("type"))


def page_match(hit: dict, target: dict) -> bool:
    return hit.get("document_id") == target.get("document_id") and hit.get("page") == target.get("page")


def first_relevant_rank(hits: list[dict], targets: list[dict]) -> int | None:
    for i, h in enumerate(hits, 1):
        if any(is_relevant(h, t) for t in targets):
            return i
    return None


def recall_at(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def precision_at(hits: list[dict], targets: list[dict], k: int) -> float:
    top = hits[:k]
    if not top:
        return 0.0
    return sum(any(is_relevant(h, t) for t in targets) for h in top) / len(top)


def per_type_ranks(hits: list[dict], gold: list[dict]) -> dict[str, int | None]:
    """For each target type in the gold set: rank of the first relevant hit for a target of that type."""
    out: dict[str, int | None] = {}
    for t in sorted({g["type"] for g in gold}):
        out[t] = first_relevant_rank(hits, [g for g in gold if g["type"] == t])
    return out


def aggregate(rows: list[dict], ks=(1, 3, 5)) -> dict[str, dict]:
    """rows: [{"ranks": {type: rank|None}, "precision@5": float}] -> per-type + overall table."""
    by_type: dict[str, list[int | None]] = {}
    for r in rows:
        for t, rank in r["ranks"].items():
            by_type.setdefault(t, []).append(rank)
    out = {}
    for t in ("text", "figure", "table"):
        ranks = by_type.get(t)
        if not ranks:
            continue
        out[t] = {"n": len(ranks), **{f"recall@{k}": round(mean(recall_at(r, k) for r in ranks), 3) for k in ks},
                  "mrr": round(mean(reciprocal_rank(r) for r in ranks), 3)}
    all_ranks = [r for v in by_type.values() for r in v]
    if all_ranks:
        out["all"] = {"n": len(all_ranks), **{f"recall@{k}": round(mean(recall_at(r, k) for r in all_ranks), 3) for k in ks},
                      "mrr": round(mean(reciprocal_rank(r) for r in all_ranks), 3)}
    if rows and "precision@5" in rows[0]:
        out.setdefault("all", {})["precision@5"] = round(mean(r["precision@5"] for r in rows), 3)
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    i = (len(v) - 1) * p / 100
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (i - lo), 1)
