"use client";

import { useState } from "react";
import { artefactUrl } from "@/lib/api";
import type { QueryResponse, TableData } from "@/lib/types";

function AnswerText({ text }: { text: string }) {
  // Render [unit-id] citations as chips so they are visibly sourced.
  const parts = text.split(/(\[[^\]]+\])/g);
  return (
    <p className="answer-text">
      {parts.map((p, i) =>
        /^\[[^\]]+\]$/.test(p) ? <span key={i} className="cite">{p.slice(1, -1)}</span> : <span key={i}>{p}</span>
      )}
    </p>
  );
}

function Table({ table, highlight }: { table: TableData; highlight: number[] }) {
  const hl = new Set(highlight);
  const [onlyMatches, setOnlyMatches] = useState(highlight.length > 0 && table.rows.length > 12);
  const rows = table.rows.map((r, i) => ({ r, i })).filter(({ i }) => !onlyMatches || hl.has(i));
  return (
    <div>
      {highlight.length > 0 && (
        <label className="toggle">
          <input type="checkbox" checked={onlyMatches} onChange={(e) => setOnlyMatches(e.target.checked)} />
          Show only the {highlight.length} rows matching the question ({table.rows.length} rows in table)
        </label>
      )}
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              {table.columns.map((c, i) => <th key={i}>{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ r, i }) => (
              <tr key={i} className={hl.has(i) ? "hl" : ""}>
                <td className="rownum">{i + 1}</td>
                {r.map((c, j) => <td key={j}>{c}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function AnswerPanel({ resp }: { resp: QueryResponse }) {
  const a = resp.artefact;
  const [showTableImage, setShowTableImage] = useState(false);

  if (resp.status === "abstained") {
    return (
      <section className="card answer abstain">
        <h2>No sourced answer</h2>
        <p>{resp.abstain_reason}</p>
        <p className="muted">The system refuses to answer rather than cite nothing. Check the sources panel for the closest units.</p>
      </section>
    );
  }

  return (
    <section className="card answer">
      <div className="answer-head">
        <span className={`type type-${resp.answer_type}`}>{resp.answer_type}</span>
        {resp.status === "partial" && <span className="warn">artefact only (no generated answer)</span>}
      </div>

      {resp.answer && <AnswerText text={resp.answer} />}
      {resp.generation_error && <p className="warn-box">{resp.generation_error}</p>}

      {a && (
        <div className="artefact">
          {a.caption && <div className="artefact-caption">{a.caption}</div>}
          {a.type === "figure" && a.image_url && (
            <a href={artefactUrl(a.image_url)} target="_blank" rel="noreferrer" title="Open full size">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={artefactUrl(a.image_url)} alt={`Figure ${a.unit_id}`} className="figure" />
            </a>
          )}
          {a.type === "table" && a.table && (
            <>
              <Table table={a.table} highlight={a.highlight_rows} />
              {a.image_url && (
                <button className="link" onClick={() => setShowTableImage(!showTableImage)}>
                  {showTableImage ? "Hide" : "Show"} the original table image
                </button>
              )}
              {showTableImage && a.image_url && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={artefactUrl(a.image_url)} alt={`Table ${a.unit_id}`} className="figure" />
              )}
            </>
          )}
          {a.type === "text" && a.text && <blockquote className="span">{a.text}</blockquote>}
        </div>
      )}

      {resp.citation && (
        <div className="citation">
          <b>Source:</b> {resp.citation.document_id} &gt; page {resp.citation.page} &gt; <code>{resp.citation.unit_id}</code>
        </div>
      )}
      <div className="timings">
        retrieval {resp.timings.retrieval_ms.toFixed(0)} ms · generation {resp.timings.generation_ms.toFixed(0)} ms · total{" "}
        {resp.timings.total_ms.toFixed(0)} ms · {resp.model_calls} model call{resp.model_calls === 1 ? "" : "s"}
      </div>
    </section>
  );
}
