"use client";

import { artefactUrl } from "@/lib/api";
import type { Source } from "@/lib/types";

// Visible by default: this is the trust mechanism. Every retrieved unit is listed, including
// the ones dropped for missing provenance (struck through, with the reason).
export default function SourcesPanel({ sources }: { sources: Source[] }) {
  return (
    <section className="card sources">
      <h2>Sources ({sources.length})</h2>
      {sources.length === 0 && <p className="muted">Nothing retrieved.</p>}
      <ol>
        {sources.map((s) => (
          <li key={`${s.rank}-${s.unit_id}`} className={`${s.cited ? "cited" : ""} ${s.sourced ? "" : "dropped"}`}>
            <div className="src-head">
              <span className="rank">#{s.rank}</span>
              <span className={`type type-${s.type}`}>{s.type}</span>
              <span className="score" title={`vector ${s.vector_score ?? "-"} · bm25 ${s.bm25_score ?? "-"}`}>
                {s.score.toFixed(3)}
              </span>
              {s.cited && <span className="tag">cited</span>}
              {s.used_in_context && !s.cited && <span className="tag tag-soft">in context</span>}
            </div>
            <div className="src-cite">
              {s.sourced ? (
                <a href={artefactUrl(`/artefact/${s.unit_id}`)} target="_blank" rel="noreferrer">{s.citation}</a>
              ) : (
                <s>{s.citation}</s>
              )}
            </div>
            {!s.sourced && <div className="drop">dropped: {s.drop_reason}</div>}
            <div className="preview">{s.preview}</div>
          </li>
        ))}
      </ol>
    </section>
  );
}
