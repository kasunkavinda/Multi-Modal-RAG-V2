"use client";

import { useCallback, useEffect, useState } from "react";
import AnswerPanel from "@/components/AnswerPanel";
import QueryBox from "@/components/QueryBox";
import SourcesPanel from "@/components/SourcesPanel";
import StatusBadge from "@/components/StatusBadge";
import { ApiFailure, getExamples, getHealth, postQuery } from "@/lib/api";
import type { ApiError, Example, Health, QueryResponse, UnitType } from "@/lib/types";

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [examples, setExamples] = useState<Example[]>([]);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [lastLatency, setLastLatency] = useState<number | null>(null);

  const refreshHealth = useCallback(async (live = false) => {
    setChecking(true);
    try {
      setHealth(await getHealth(live));
      setHealthError(null);
    } catch (e) {
      setHealthError(e instanceof ApiFailure ? e.detail.message : String(e));
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    getExamples().then((r) => setExamples(r.examples)).catch(() => setExamples([]));
  }, [refreshHealth]);

  const ask = async (question: string, types: UnitType[]) => {
    setLoading(true);
    setError(null);
    const t0 = performance.now();
    try {
      setResp(await postQuery(question, types));
    } catch (e) {
      setResp(null);
      setError(e instanceof ApiFailure ? e.detail : { kind: "unknown", message: String(e) });
      refreshHealth();
    } finally {
      setLastLatency(performance.now() - t0);
      setLoading(false);
    }
  };

  return (
    <main>
      <header className="top">
        <div>
          <h1>Manual Answers</h1>
          <p className="muted">Ask a question; get the real figure or table rows back, with the page they came from.</p>
        </div>
        <div className="top-right">
          {lastLatency !== null && <span className="muted">last answer {(lastLatency / 1000).toFixed(1)} s</span>}
          <StatusBadge health={health} error={healthError} onRefresh={refreshHealth} checking={checking} />
        </div>
      </header>

      <QueryBox examples={examples} loading={loading} onAsk={ask} />

      {loading && (
        <div className="card busy" role="status">
          <span className="spinner" /> Retrieving sources and preparing a grounded answer… (free-tier models can take 10-30 s)
        </div>
      )}

      {error && !loading && (
        <div className="card error" role="alert">
          <h2>{error.kind === "configuration" ? "Configuration problem" : error.kind === "upstream" ? "Model provider problem" : "Request failed"}</h2>
          <p>{error.message}{error.model ? ` (model: ${error.model})` : ""}</p>
          {error.fix && <p className="fix">Fix: {error.fix}</p>}
        </div>
      )}

      {resp && !loading && (
        <div className="results">
          <AnswerPanel resp={resp} />
          <SourcesPanel sources={resp.sources} />
        </div>
      )}
    </main>
  );
}
