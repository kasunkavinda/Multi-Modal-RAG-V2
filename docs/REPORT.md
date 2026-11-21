# Lab report: multimodal retrieval over technical manuals

Fill in as you go. Numbers come from `rag/data/model_check.json`, `rag/data/reports/*.md` and
`rag/data/index_report.json`. Never type a number the tools did not produce.

## Stage 0: prediction (write BEFORE any metric is run)
- Question type we expect to retrieve worst: ____ because ____
- Recorded on: ____

## T1 Model readiness log (`python scripts/check_models.py`)
| Role | Slug | Call OK? | Time (ms) | Notes / error |
|---|---|---|---|---|
| Embeddings | | | | |
| Vision / captioning | | | | |
| Text generation | | | | |
| Figure answers | | | | |

## Stage 1: routing (`python scripts/build_index.py --ingest-only`)
Measured on this machine without OCR (tesseract not installed), all three corpora:

| Class | Pages |
|---|---|
| native | 210 |
| mixed (figures / tables / drawings) | 147 |
| scanned (no text layer, page-filling image) | 25 |
| scanned with a text layer | 6 |
| blank | 64 |

Extracted: 104 figures, 97 tables, 497 text chunks (plus Enginuity T2: 12 figures, 15 tables).
Review queue: 2 pages (data-7 p15 and p17: scanned change-notice text pages, no OCR backend). The other
scanned pages carry a "Figure N. Title" text layer and are routed to the figure path, not OCR.

Two-sentence note: a plain `get_text()` loader would have indexed data-7's parts lists as one string per
column (every row of the NSN column joined together), losing which part number belongs to which item,
and would have dropped every exploded diagram because they are raster images with no text.
____ (edit in your own words)

## Stage 2: enrichment evidence
- Caption cache: `rag/data/cache/captions.json`
- Rejected units with reasons: `rag/data/index_report.json` -> `rejected`
- Figure baseline decision: caption text (not image embedding), because ____

Five-unit spot-check:
| Unit id | Page resolves? | Caption names the system? | Notes |
|---|---|---|---|
| | | | |

## T2 Layer A: retrieval by type
| Type | n | Recall@1 | Recall@3 | Recall@5 | MRR | Stage |
|---|---|---|---|---|---|---|
| Text | | | | | | Baseline |
| Text | | | | | | After fix 1 |
| Text | | | | | | After fix 2 |
| Figure | | | | | | Baseline |
| Figure | | | | | | After fix 1 |
| Figure | | | | | | After fix 2 |
| Table | | | | | | Baseline |
| Table | | | | | | After fix 1 |
| Table | | | | | | After fix 2 |

Baseline tag: `git tag baseline` at commit ____.

## T3 Failure-mode log (Stage 4)
Candidate levers already wired as single config switches (change ONE, rebuild what changed, re-run Layer A with `--compare`):
- `LINK_TABLE_TO_FIGURE=true`: parts-list tables carry no assembly name ("right table never retrieved").
- `FUSION=true` / `BM25_WEIGHT`: part numbers and NSNs smeared by dense vectors.
- `EMBED_IMAGES=true`: caption versus image embedding for figures (stretch).
- Deliberate damage for a strong baseline: `MAX_CAPTIONS=0` (figures without captions) and `--inject-broken-unit`.

| # | Symptom and question IDs | Root cause | Change made | Before | After |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |

## T4 Layers B and C
| Layer | Metric | Value | How measured / sample size |
|---|---|---|---|
| B | Answer correctness | | |
| B | Faithfulness (five-answer check) | | |
| B | Citation accuracy | | |
| B | Wrong-source rate | | |
| B | Artefact correctness | | |
| C | Latency p50 / p95 (retrieval, generation) | | |
| C | Calls per query | | |
| C | Ingestion time and index size | | |
| C | Queries per minute under the cap | | |

Five-answer faithfulness check through the UI (E8.6):
| Question | Artefact on screen is gold? | Reads well but wrong figure? |
|---|---|---|
| | | |

## T5 Demo checklist
- [ ] Text question: answer, citation and sources panel shown.
- [ ] Figure question: the actual figure image shown with page citation.
- [ ] Table question: the exact rows shown with page citation.
- [ ] A question the system should refuse: it abstains and says why (example: the wine-pairing question).
- [ ] One known weakness shown honestly, with the number that proves it.
- [ ] Status badge shown healthy; one degraded state demonstrated (set `TEXT_MODEL=retired/slug:free`, restart, ask).

## Provenance and data-handling note (E9.2)
**How provenance travels.**
- *Ingest:* every unit is created with `Provenance(document_id, page, unit_id, bbox, source_path)`. PDF
  pages are 1-based. Enginuity T2 crops have no source page, so the figure group is the document and
  its pages are numbered (figure = p1, tables = p2+).
- *Enrich:* units missing any field are rejected before a paid call.
- *Store:* `VectorStore.write` refuses any unit without document, page and unit id, and logs it. The
  payload carries document, page, unit id, type and the citation string.
- *Answer:* hits are re-checked; unsourced ones are dropped and shown struck through. The answer may
  only cite ids that were in its context; invented ids are stripped.
- *UI:* the citation line and the sources panel are always visible. `/artefact/{id}` serves files
  only from inside the data folder and sends an `X-Citation` header.

**Real manual sets.** ____ (what you would redact or restrict: personal names and signatures on change
notices, export-controlled drawings, customer asset ids; and who may query which collection.)

**OCR-failed pages** go to `data/review_queue.json` with the reason and a page render; they are never
indexed. ____

**Residency.** Free OpenRouter endpoints are for public datasets only. Confidential manuals would need
captioning and embeddings on a private or self-hosted endpoint, so the bytes never leave your tenancy. ____

## One-page reflection (E9.3)
- Enrichment and UI choices:
- Change that moved figure and table recall most:
- Latency and cost reality:
- Grounding-ladder position (prompt / retrieve / fine-tune / distil):
- Stage 0 prediction versus result:
- One limitation a polished demo would hide:
