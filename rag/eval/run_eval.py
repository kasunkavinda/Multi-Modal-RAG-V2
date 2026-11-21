"""One command for the three evaluation layers (E8.1-E8.4).

    python -m eval.run_eval --label baseline                 # Layer A only (retrieval, generator out of the loop)
    python -m eval.run_eval --label baseline --layers A,B,C --sample 20
    python -m eval.run_eval --label fix1 --compare baseline  # lists question ids that moved

Layer A: recall@1/3/5 and MRR@10 per type (text / figure / table), precision@5, page-only hits.
Layer B: answer correctness + faithfulness (LLM judge, temperature 0, fixed prompt), citation accuracy,
         wrong-source rate, artefact correctness, abstention rate. Uses the SAME query pipeline as the API.
Layer C: latency p50/p95 (retrieval vs generation), calls per query, ingestion time, index size,
         queries/minute under the free-tier cap.
Reports: data/reports/<label>.json (per-question results) and <label>.md (templates T2/T4).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from app.config import ConfigError, Settings, get_settings  # noqa: E402
from app.openrouter import UpstreamError  # noqa: E402
from app.pipeline import RagService  # noqa: E402
from app.retrieve import load_index_report  # noqa: E402
from eval.metrics import (aggregate, is_relevant, page_match, per_type_ranks, percentile,  # noqa: E402
                          precision_at)

EVAL_K = 10
JUDGE_PROMPT = """You are grading an answer produced by a retrieval system for maintenance engineers.
Return ONLY a JSON object: {{"correct": true|false, "faithful": true|false, "reason": "<one sentence>"}}

correct  = the system answer agrees with the key facts of the reference answer (wording may differ; an answer
           that is incomplete but not contradictory and covers the main point counts as correct).
faithful = every factual claim in the system answer is supported by the SOURCES text shown below.

QUESTION: {question}
REFERENCE ANSWER: {reference}
SOURCES THE SYSTEM SAW:
{sources}
SYSTEM ANSWER: {answer}"""


def hit_dict(h) -> dict:
    p = h.unit.provenance
    return {"unit_id": h.unit.unit_id, "type": h.unit.type, "document_id": p.document_id, "page": p.page,
            "score": round(h.score, 4)}


# ------------------------------------------------------------------ Layer A
def layer_a(service: RagService, qa: list[dict]) -> dict:
    retr = service.retriever
    calls0 = sum(service.client.stats.values())
    t0 = time.perf_counter()
    results = retr.retrieve_batch([q["question"] for q in qa], k=EVAL_K)
    wall = time.perf_counter() - t0
    rows = []
    for q, r in zip(qa, results):
        hits = [hit_dict(h) for h in r.hits]
        ranks = per_type_ranks(hits, q["gold"])
        page_only = sum(1 for h in hits[:5] if any(page_match(h, g) for g in q["gold"])
                        and not any(is_relevant(h, g) for g in q["gold"]))
        rows.append({"qid": q["qid"], "corpus": q["corpus"], "qtype": q["qtype"], "ranks": ranks,
                     "precision@5": precision_at(hits, q["gold"], 5), "page_only_hits@5": page_only,
                     "top5": hits[:5], "question": q["question"]})
    by_corpus = {c: aggregate([r for r in rows if r["corpus"] == c]) for c in sorted({r["corpus"] for r in rows})}
    return {"k": EVAL_K, "n_questions": len(rows), "by_type": aggregate(rows), "by_corpus": by_corpus,
            "page_only_hits@5": sum(r["page_only_hits@5"] for r in rows),
            "embedding_calls": sum(service.client.stats.values()) - calls0, "wall_s": round(wall, 2),
            "questions": rows}


# ------------------------------------------------------------------ Layer B
def stratified_sample(qa: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0 or n >= len(qa):
        return list(qa)
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for q in qa:
        groups.setdefault(q["qtype"], []).append(q)
    for g in groups.values():
        rng.shuffle(g)
    out: list[dict] = []
    while len(out) < n and any(groups.values()):
        for k in sorted(groups):
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def parse_judge(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    def b(v):
        return v if isinstance(v, bool) else str(v).strip().lower() == "true"
    return {"correct": b(d.get("correct", False)), "faithful": b(d.get("faithful", False)),
            "reason": str(d.get("reason", text[:200]))}


def layer_b(service: RagService, qa: list[dict], sample: int, seed: int, judge: bool) -> dict:
    s: Settings = service.s
    picked = stratified_sample(qa, sample, seed)
    catalog = service.retriever.catalog
    rows = []
    for i, q in enumerate(picked, 1):
        print(f"  [B {i}/{len(picked)}] {q['qid']} ...", flush=True)
        t0 = time.perf_counter()
        try:
            resp = service.query(q["question"])
        except (UpstreamError, ConfigError) as e:
            rows.append({"qid": q["qid"], "qtype": q["qtype"], "error": str(e)})
            continue
        cited = resp.cited_unit_ids
        cited_h = [{"unit_id": c, "type": catalog[c].type, "document_id": catalog[c].provenance.document_id,
                    "page": catalog[c].provenance.page} for c in cited if c in catalog]
        cit_ok = any(is_relevant(h, g) for h in cited_h for g in q["gold"]) if cited_h else False
        art = resp.artefact
        art_ok = None
        if art is not None and q["qtype"] in ("figure", "table"):
            u = catalog[art.unit_id]
            art_ok = any(is_relevant({"unit_id": u.unit_id, "type": u.type, "document_id": u.provenance.document_id,
                                      "page": u.provenance.page}, g) for g in q["gold"])
        row = {"qid": q["qid"], "qtype": q["qtype"], "corpus": q["corpus"], "status": resp.status,
               "answer_type": resp.answer_type, "answer": resp.answer, "abstain_reason": resp.abstain_reason,
               "generation_error": resp.generation_error, "cited": cited, "citation_correct": cit_ok,
               "artefact_unit": art.unit_id if art else None, "artefact_correct": art_ok,
               "timings": resp.timings.model_dump(), "model_calls": resp.model_calls,
               "wall_ms": round((time.perf_counter() - t0) * 1000, 1), "reference": q["answer"]}
        if judge and resp.status == "answered" and resp.answer and q.get("answer"):
            srcs = "\n\n".join(_context_for(catalog, [x.unit_id for x in resp.sources if x.used_in_context]))
            try:
                out = service.client.chat(
                    [{"role": "user", "content": JUDGE_PROMPT.format(question=q["question"], reference=q["answer"],
                                                                     sources=srcs[:12000], answer=resp.answer)}],
                    model=s.judge_model, role="judge", max_tokens=300, temperature=0.0)
                row["judge"] = parse_judge(out)
            except UpstreamError as e:
                row["judge_error"] = str(e)
        rows.append(row)

    ok = [r for r in rows if "error" not in r]
    answered = [r for r in ok if r["status"] in ("answered", "partial")]
    judged = [r for r in ok if "judge" in r]
    art = [r for r in ok if r["artefact_correct"] is not None]

    def rate(xs, key):
        return round(mean(1.0 if key(x) else 0.0 for x in xs), 3) if xs else None

    return {
        "n_sampled": len(picked), "n_ok": len(ok), "errors": len(rows) - len(ok),
        "answer_correctness": rate(judged, lambda r: r["judge"]["correct"]),
        "faithfulness": rate(judged, lambda r: r["judge"]["faithful"]),
        "n_judged": len(judged),
        "citation_accuracy": rate(answered, lambda r: r["citation_correct"]),
        "wrong_source_rate": rate(answered, lambda r: not r["citation_correct"]),
        "artefact_correctness": rate(art, lambda r: r["artefact_correct"]),
        "n_artefact_questions": len(art),
        "abstention_rate": rate(ok, lambda r: r["status"] == "abstained"),
        "status_counts": dict(Counter(r["status"] for r in ok)),
        "by_qtype": {t: {"n": len(xs), "citation_accuracy": rate([x for x in xs if x["status"] != "abstained"],
                                                                  lambda r: r["citation_correct"]),
                         "artefact_correctness": rate([x for x in xs if x["artefact_correct"] is not None],
                                                      lambda r: r["artefact_correct"])}
                     for t in ("text", "figure", "table") if (xs := [r for r in ok if r["qtype"] == t])},
        "questions": rows,
    }


def _context_for(catalog, ids):
    from app.generate import context_block

    return [context_block(catalog[i]) for i in ids if i in catalog]


# ------------------------------------------------------------------ Layer C
def layer_c(s: Settings, b: dict | None, a: dict | None) -> dict:
    idx = load_index_report(s) or {}
    out: dict = {"ingestion_s": idx.get("timings_s"), "index_size_bytes": idx.get("index_size_bytes"),
                 "units": idx.get("units_by_type"), "ingestion_model_calls": idx.get("model_calls")}
    rows = [r for r in (b or {}).get("questions", []) if "timings" in r]
    if rows:
        ret = [r["timings"]["retrieval_ms"] for r in rows]
        gen = [r["timings"]["generation_ms"] for r in rows]
        tot = [r["timings"]["total_ms"] for r in rows]
        calls = [r["model_calls"] for r in rows]
        cpq = mean(calls)
        out.update({
            "n_queries": len(rows),
            "latency_ms": {"retrieval_p50": percentile(ret, 50), "retrieval_p95": percentile(ret, 95),
                           "generation_p50": percentile(gen, 50), "generation_p95": percentile(gen, 95),
                           "total_p50": percentile(tot, 50), "total_p95": percentile(tot, 95)},
            "calls_per_query": round(cpq, 2),
            "calls_note": "live API attempts incl. retries; cached query embeddings cost 0",
            # Free tier: 20 req/min per model. A query hits the embedder once and one generator once,
            # so each model sees ~1 call/query; a single shared limit would give rpm / calls_per_query.
            "queries_per_minute_per_model_limit": s.requests_per_minute,
            "queries_per_minute_if_shared_limit": round(s.requests_per_minute / cpq, 1) if cpq else None,
        })
    if a:
        out["layer_a_embedding_calls"] = a["embedding_calls"]
        out["layer_a_wall_s"] = a["wall_s"]
    return out


# ------------------------------------------------------------------ report
def compare(cur: dict, prev: dict) -> dict:
    p = {r["qid"]: r for r in prev.get("layer_a", {}).get("questions", [])}
    moved = {"improved": [], "regressed": []}
    for r in cur.get("layer_a", {}).get("questions", []):
        o = p.get(r["qid"])
        if not o:
            continue
        for t, rank in r["ranks"].items():
            before = o["ranks"].get(t)
            hit_now, hit_before = rank is not None and rank <= 5, before is not None and before <= 5
            if hit_now and not hit_before:
                moved["improved"].append(f"{r['qid']}:{t} ({before}->{rank})")
            elif hit_before and not hit_now:
                moved["regressed"].append(f"{r['qid']}:{t} ({before}->{rank})")
    return moved


def to_markdown(rep: dict, prev: dict | None) -> str:
    L = [f"# Evaluation report: {rep['label']}", "", f"- run at: {rep['run_at']}",
         f"- index: {json.dumps(rep.get('index', {}))}", ""]
    if "layer_a" in rep:
        a = rep["layer_a"]
        L += ["## Layer A - retrieval by type (generator out of the loop)", "",
              f"{a['n_questions']} questions, top-{a['k']} retrieved, {a['embedding_calls']} embedding calls (batched).", "",
              "| Type | n | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Stage |", "|---|---|---|---|---|---|---|"]
        rows = [(rep["label"], a["by_type"])] + ([(prev["label"], prev["layer_a"]["by_type"])] if prev and "layer_a" in prev else [])
        for t in ("text", "figure", "table", "all"):
            for label, bt in rows:
                if t in bt and "n" in bt[t]:
                    m = bt[t]
                    L.append(f"| {t} | {m['n']} | {m['recall@1']} | {m['recall@3']} | {m['recall@5']} | {m['mrr']} | {label} |")
        L += ["", f"precision@5 (all): {a['by_type'].get('all', {}).get('precision@5')}; "
              f"page-only hits in top 5 (right page, wrong unit type - not counted): {a['page_only_hits@5']}", ""]
        for c, bt in a["by_corpus"].items():
            L.append(f"- {c}: " + "; ".join(f"{t} R@5={m['recall@5']} MRR={m['mrr']} (n={m['n']})"
                                            for t, m in bt.items() if "n" in m))
        miss = [f"{r['qid']}:{t}" for r in a["questions"] for t, rank in r["ranks"].items() if not rank or rank > 5]
        L += ["", f"Gold missing from top 5 ({len(miss)}): " + ", ".join(miss[:80]) + (" ..." if len(miss) > 80 else ""), ""]
    if rep.get("moved"):
        L += ["## Question ids that moved vs " + (prev or {}).get("label", "previous"), "",
              f"- improved ({len(rep['moved']['improved'])}): " + ", ".join(rep["moved"]["improved"]),
              f"- regressed ({len(rep['moved']['regressed'])}): " + ", ".join(rep["moved"]["regressed"]), ""]
    if "layer_b" in rep:
        b = rep["layer_b"]
        L += ["## Layer B - answer and grounding", "", "| Metric | Value | How measured / sample size |", "|---|---|---|",
              f"| Answer correctness | {b['answer_correctness']} | LLM judge (temp 0, fixed prompt), n={b['n_judged']} answered |",
              f"| Faithfulness | {b['faithfulness']} | same judge vs the sources shown, n={b['n_judged']} |",
              f"| Citation accuracy | {b['citation_accuracy']} | cited unit matches gold, n={b['n_ok']} sampled (stratified) |",
              f"| Wrong-source rate | {b['wrong_source_rate']} | answered but cited a non-gold unit |",
              f"| Artefact correctness | {b['artefact_correctness']} | figure/table questions, n={b['n_artefact_questions']} |",
              f"| Abstention rate | {b['abstention_rate']} | status counts {b['status_counts']} |", ""]
    if "layer_c" in rep:
        c = rep["layer_c"]
        L += ["## Layer C - operational", "", "```json", json.dumps(c, indent=1), "```", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--layers", default="A")
    ap.add_argument("--sample", type=int, default=20, help="Layer B sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--corpus", help="only questions from this corpus")
    ap.add_argument("--compare", help="label (or path) of a previous report")
    a = ap.parse_args()

    s = get_settings()
    qa = json.loads(s.qa_path.read_text(encoding="utf-8"))
    if a.corpus:
        qa = [q for q in qa if q["corpus"] == a.corpus]
    layers = {x.strip().upper() for x in a.layers.split(",")}
    try:
        service = RagService(s)
        rep: dict = {"label": a.label, "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "settings": {"fusion": s.fusion, "bm25_weight": s.bm25_weight, "top_k": s.top_k,
                                  "min_score": s.min_score, "text_model": s.text_model, "figure_model": s.figure_model,
                                  "judge_model": s.judge_model}}
        idx = service.retriever.report
        rep["index"] = {"embedding": idx.get("embedding"), "built_at": idx.get("built_at"),
                        "units_by_type": idx.get("units_by_type"), "settings": idx.get("settings")}
        if "A" in layers:
            print(f"Layer A: {len(qa)} questions ...")
            rep["layer_a"] = layer_a(service, qa)
        if "B" in layers:
            print(f"Layer B: sample {a.sample or len(qa)} ...")
            rep["layer_b"] = layer_b(service, qa, a.sample, a.seed, judge=not a.no_judge)
        if "C" in layers:
            rep["layer_c"] = layer_c(s, rep.get("layer_b"), rep.get("layer_a"))
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}\n  fix: {e.fix}")
        return 2
    except UpstreamError as e:
        print(f"UPSTREAM ERROR: {e}")
        return 3

    prev = None
    if a.compare:
        pp = Path(a.compare) if a.compare.endswith(".json") else s.reports_dir / f"{a.compare}.json"
        prev = json.loads(pp.read_text(encoding="utf-8"))
        rep["moved"] = compare(rep, prev)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for name in (f"{a.label}.json", f"{a.label}_{stamp}.json"):
        (s.reports_dir / name).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(rep, prev)
    (s.reports_dir / f"{a.label}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nsaved data/reports/{a.label}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
