// Thin API client. No business logic: the service decides artefacts, citations and abstention.
import type { ApiError, Example, Health, QueryResponse, UnitType } from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").replace(/\/$/, "");

export class ApiFailure extends Error {
  constructor(public detail: ApiError) {
    super(detail.message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, cache: "no-store" });
  } catch {
    throw new ApiFailure({
      kind: "network",
      message: `Cannot reach the API at ${API_BASE}.`,
      fix: "Start it from rag/: uvicorn app.server:app --port 8000 (and check CORS_ORIGINS).",
    });
  }
  if (res.ok) return (await res.json()) as T;
  let body: any = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error */
  }
  if (body?.error) throw new ApiFailure(body.error as ApiError);
  if (res.status === 422) {
    throw new ApiFailure({ kind: "validation", message: "The question was rejected by the API (empty or too long)." });
  }
  throw new ApiFailure({ kind: "unknown", message: `API error ${res.status}: ${body?.detail ?? res.statusText}` });
}

export const getHealth = (live = false) => request<Health>(`/health${live ? "?live=true" : ""}`);
export const getExamples = () => request<{ examples: Example[] }>("/examples");
export const postQuery = (question: string, types?: UnitType[], k?: number) =>
  request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, types: types && types.length ? types : null, k: k ?? null }),
  });

export const artefactUrl = (path: string) => `${API_BASE}${path}`;
