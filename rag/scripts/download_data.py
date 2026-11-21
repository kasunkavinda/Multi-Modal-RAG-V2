"""Download the lab corpora (section 7). Tier A by default.

    python scripts/download_data.py               # tier A: Enginuity Task 2 (~6.6 MB)
    python scripts/download_data.py --tier B      # + data-7.pdf (smallest manual) or --all-pdfs
    python scripts/download_data.py --tier C      # + ViDoRe v3 Industrial labels + small PDFs + page map
"""
import _bootstrap  # noqa: F401
import argparse
import subprocess
import sys

from huggingface_hub import snapshot_download

from app.config import get_settings

VIDORE_REV = "233d20721f4deb392a09a86cca01761adbc91157"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="A", choices=["A", "B", "C", "all"])
    ap.add_argument("--all-pdfs", action="store_true", help="tier B: all 10 manuals (122 MB) instead of data-7.pdf")
    a = ap.parse_args()
    s = get_settings()
    eng = s.raw_dir / "enginuity"
    snapshot_download("enginuity2025/enginuity-bench", repo_type="dataset", local_dir=eng,
                      allow_patterns=["task-2-question-answering/*"])
    print("tier A ok")
    if a.tier in ("B", "all"):
        pdfs = (["task-1-component-identification/pdfs/*"] if a.all_pdfs
                else [f"task-1-component-identification/pdfs/{p}" for p in s.enginuity_pdfs])
        snapshot_download("enginuity2025/enginuity-bench", repo_type="dataset", local_dir=eng,
                          allow_patterns=pdfs + ["task-1-component-identification/human-validation-set/*"])
        print("tier B ok")
    if a.tier in ("C", "all"):
        snapshot_download("vidore/vidore_v3_industrial", repo_type="dataset", revision=VIDORE_REV,
                          local_dir=s.raw_dir / "vidore",
                          allow_patterns=["queries/*", "qrels/*", "documents_metadata/*", "pdfs/metadata.csv",
                                          "pdfs/TO_1-1-3.pdf", "pdfs/TO_1-1-8.pdf", "pdfs/TO_00-25-234.pdf",
                                          "pdfs/TO_00-25-172CL-*.pdf"])
        if not (s.raw_dir / "vidore" / "corpus_pagemap.json").exists():
            subprocess.run([sys.executable, str(_bootstrap.ROOT / "scripts" / "fetch_vidore_pagemap.py")], check=True)
        print("tier C ok")


if __name__ == "__main__":
    main()
