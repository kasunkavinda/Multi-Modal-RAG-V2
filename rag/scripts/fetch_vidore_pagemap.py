"""Fetch the ViDoRe v3 Industrial corpus_id -> (doc_id, page) map without downloading page images.

The qrels reference `corpus_id`; the mapping lives in the 2 GB corpus parquet. Parquet is columnar,
so we read only the small columns over HTTP range requests (about 30 s per shard).
Output: data/raw/vidore/corpus_pagemap.json  {corpus_id: {"doc_id":..., "page": 1-based}}
"""
from __future__ import annotations

import json
import sys

import _bootstrap  # noqa: F401

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from app.config import get_settings

REVISION = "233d20721f4deb392a09a86cca01761adbc91157"


def main() -> None:
    out = get_settings().raw_dir / "vidore" / "corpus_pagemap.json"
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/vidore/vidore_v3_industrial@{REVISION}/corpus/*.parquet"))
    mapping: dict[str, dict] = {}
    for f in files:
        print(f"reading {f.rsplit('/', 1)[-1]} ...", flush=True)
        with fs.open(f) as fh:
            tb = pq.ParquetFile(fh).read(columns=["corpus_id", "doc_id", "page_number_in_doc"])
        for row in tb.to_pylist():
            # page_number_in_doc is 0-based in the dataset; provenance uses 1-based pages.
            mapping[str(row["corpus_id"])] = {"doc_id": row["doc_id"], "page": int(row["page_number_in_doc"]) + 1}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping), encoding="utf-8")
    print(f"wrote {len(mapping)} pages -> {out}")


if __name__ == "__main__":
    main()
