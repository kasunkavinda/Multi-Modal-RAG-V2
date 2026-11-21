"use client";

import { useState } from "react";
import type { Health } from "@/lib/types";

interface Props {
  health: Health | null;
  error: string | null;
  onRefresh: (live: boolean) => void;
  checking: boolean;
}

export default function StatusBadge({ health, error, onRefresh, checking }: Props) {
  const [open, setOpen] = useState(false);
  const state = error ? "down" : health?.status ?? "unknown";
  const label =
    state === "healthy" ? "Healthy" : state === "degraded" ? `Degraded (${health?.problems.length})` : state === "down" ? "API unreachable" : "Checking…";
  const units = health?.index?.units_by_type as Record<string, number> | undefined;

  return (
    <div className="status">
      <button className={`badge badge-${state}`} onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="dot" /> {label}
      </button>
      {open && (
        <div className="status-panel" role="dialog">
          {error && <p className="problem"><b>API:</b> {error}</p>}
          {health?.problems.map((p, i) => (
            <div key={i} className="problem">
              <b>{p.kind}:</b> {p.message}
              {p.fix && <div className="fix">Fix: {p.fix}</div>}
            </div>
          ))}
          {health && health.problems.length === 0 && <p>All checks passed.</p>}
          {health && (
            <dl className="meta">
              <dt>Vector store</dt><dd>{health.vector_backend}</dd>
              {Object.entries(health.models).map(([role, slug]) => (
                <span key={role} style={{ display: "contents" }}><dt>{role}</dt><dd><code>{slug}</code></dd></span>
              ))}
              {units && (<><dt>Index</dt><dd>{Object.entries(units).map(([t, n]) => `${n} ${t}`).join(" · ")}</dd></>)}
              {health.model_check && (<><dt>Models checked</dt><dd>{health.model_check.checked_at}</dd></>)}
            </dl>
          )}
          <div className="row">
            <button onClick={() => onRefresh(false)} disabled={checking}>Refresh</button>
            <button onClick={() => onRefresh(true)} disabled={checking} title="Makes one tiny call per model (counts against the free-tier cap)">
              {checking ? "Checking…" : "Check models live"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
