"use client";

import { useState } from "react";
import type { Example, UnitType } from "@/lib/types";

interface Props {
  examples: Example[];
  loading: boolean;
  onAsk: (question: string, types: UnitType[]) => void;
}

const FILTERS: { label: string; value: UnitType[] }[] = [
  { label: "Any type", value: [] },
  { label: "Figures", value: ["figure"] },
  { label: "Tables", value: ["table"] },
  { label: "Text", value: ["text"] },
];

export default function QueryBox({ examples, loading, onAsk }: Props) {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState(0);

  const submit = (question: string) => {
    const text = question.trim();
    if (!text || loading) return;
    setQ(text);
    onAsk(text, FILTERS[filter].value);
  };

  return (
    <section className="card">
      <form
        className="querybox"
        onSubmit={(e) => {
          e.preventDefault();
          submit(q);
        }}
      >
        <input
          aria-label="Question"
          placeholder="Ask about a component, a diagram or a spec… e.g. show the cylinder block assembly"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          maxLength={2000}
        />
        <select aria-label="Content type filter" value={filter} onChange={(e) => setFilter(Number(e.target.value))}>
          {FILTERS.map((f, i) => (
            <option key={f.label} value={i}>{f.label}</option>
          ))}
        </select>
        <button type="submit" disabled={loading || !q.trim()}>{loading ? "Searching…" : "Ask"}</button>
      </form>
      {examples.length > 0 && (
        <div className="examples">
          <span>Try:</span>
          {examples.map((ex) => (
            <button key={ex.question} className="chip" onClick={() => submit(ex.question)} disabled={loading} title={ex.qid}>
              <span className={`type type-${ex.type}`}>{ex.type}</span> {ex.question.length > 90 ? ex.question.slice(0, 88) + "…" : ex.question}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
