"""Build the held-out gold file data/qa.json (E0.3) and data/examples.json.

Each record: qid, corpus, question, answer, qtype (text|figure|table), gold targets.
  Enginuity Task 2: targets are unit ids (the figure and its parts-list tables).
  ViDoRe v3 Industrial: targets are (document, page, type) from the qrels; only English
  queries whose relevant pages ALL fall inside the PDFs configured in VIDORE_PDFS.

    python scripts/make_gold.py
"""
import _bootstrap  # noqa: F401
import glob
import json
import re
from collections import Counter

from app.config import get_settings

TABLE_Q = re.compile(r"part number|stock number|\bnsn\b|national stock|quantity|how many|\bqty\b|"
                     r"order (each|them|it)|sold separately|item \d+.*(number|ordered)", re.I)
VIDORE_TYPE = {"text": "text", "table": "table", "image": "figure", "infographic": "figure", "chart": "figure"}


def clean_context(ctx: str) -> str:
    """Several Enginuity contexts end with stray caption text from other figures: cut it."""
    ctx = re.split(r"\s(?:Figure|FIGURE|Fig\.)\s*\d+", ctx)[0]
    ctx = re.sub(r"(\s[A-Z][A-Z0-9,\-/]+){4,}\s*$", "", ctx)   # trailing ALL-CAPS caption run
    return ctx.strip()


def enginuity(s) -> list[dict]:
    base = s.raw_dir / "enginuity" / "task-2-question-answering"
    recs = json.loads((base / "qa_dataset.json").read_text(encoding="utf-8"))
    out = []
    for i, r in enumerate(recs, 1):
        nn = re.search(r"figure_(\d+)", r["figure"]).group(1)
        doc = f"enginuity-t2-fig{nn}"
        gold = [{"unit_id": f"{doc}:figure", "type": "figure", "document_id": doc, "page": 1}]
        for t in r["tables"]:
            m = int(re.search(r"table_\d+_(\d+)", t).group(1))
            gold.append({"unit_id": f"{doc}:table{m}", "type": "table", "document_id": doc, "page": 1 + m})
        out.append({"qid": f"en-{i:03d}", "corpus": "enginuity_t2", "question": r["question"].strip(),
                    "answer": r["answer"].strip(), "context": clean_context(r.get("context", "")),
                    "qtype": "table" if TABLE_Q.search(r["question"]) else "figure", "gold": gold})
    return out


def vidore(s) -> list[dict]:
    import pyarrow.parquet as pq

    base = s.raw_dir / "vidore"
    pm_path = base / "corpus_pagemap.json"
    if not pm_path.exists():
        print("ViDoRe page map missing: run scripts/fetch_vidore_pagemap.py")
        return []
    pm = json.loads(pm_path.read_text(encoding="utf-8"))
    docs = {p.rsplit(".", 1)[0] for p in s.vidore_pdfs}
    max_page = s.max_pages_per_pdf or 10**9
    qs = [q for q in pq.read_table(glob.glob(str(base / "queries" / "*.parquet"))).to_pylist()
          if q["language"] == "english"]
    qrels = pq.read_table(glob.glob(str(base / "qrels" / "*.parquet"))).to_pylist()
    by_q: dict[str, list[dict]] = {}
    for r in qrels:
        by_q.setdefault(str(r["query_id"]), []).append(r)
    out = []
    for q in qs:
        rels = [r for r in by_q.get(str(q["query_id"]), []) if int(r["score"]) >= 1]
        if not rels:
            continue
        pages = [pm[str(r["corpus_id"])] for r in rels]
        if not all(p["doc_id"] in docs and p["page"] <= max_page for p in pages):
            continue
        gold = []
        for r, p in zip(rels, pages):
            types = {VIDORE_TYPE.get(t.strip().lower()) for t in eval_list(r["content_type"])} - {None}
            for t in sorted(types or {"text"}):
                gold.append({"document_id": p["doc_id"], "page": p["page"], "type": t, "score": int(r["score"])})
        qtypes = {VIDORE_TYPE.get(t.strip().lower()) for t in eval_list(q["content_type"])} - {None}
        qtype = "table" if "table" in qtypes else "figure" if "figure" in qtypes else "text"
        out.append({"qid": f"vd-{q['query_id']}", "corpus": "vidore", "question": q["query"].strip(),
                    "answer": (q.get("answer") or "").strip(), "qtype": qtype, "gold": gold,
                    "query_format": q.get("query_format"), "origin": q.get("query_generator")})
    return out


def eval_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    return re.findall(r"'([^']*)'", str(v)) or [str(v)]


def main() -> None:
    s = get_settings()
    qa = []
    if "enginuity_t2" in s.corpora:
        qa += enginuity(s)
    if "vidore" in s.corpora:
        qa += vidore(s)
    s.qa_path.write_text(json.dumps(qa, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(qa)} questions -> {s.qa_path}")
    print("by corpus:", dict(Counter(q["corpus"] for q in qa)))
    print("by qtype:", dict(Counter(q["qtype"] for q in qa)))
    print("gold targets by type:", dict(Counter(g["type"] for q in qa for g in q["gold"])))
    ex = []
    for t in ("figure", "table", "text"):
        q = next((q for q in qa if q["qtype"] == t), None)
        if q:
            ex.append({"question": q["question"], "type": t, "qid": q["qid"]})
    ex.append({"question": "What is the recommended wine pairing for grilled salmon?", "type": "abstain",
               "qid": "off-topic"})
    (s.data_dir / "examples.json").write_text(json.dumps({"examples": ex}, indent=1, ensure_ascii=False),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
