// Shared types mirroring rag/app/schemas.py (the API contract).
export type UnitType = "text" | "figure" | "table";

export interface TableData {
  columns: string[];
  rows: string[][];
}

export interface Artefact {
  type: UnitType;
  unit_id: string;
  url: string;
  image_url: string | null;
  table: TableData | null;
  highlight_rows: number[];
  text: string | null;
  caption: string | null;
}

export interface Citation {
  document_id: string;
  page: number;
  unit_id: string;
  text: string;
}

export interface Source {
  rank: number;
  unit_id: string;
  type: UnitType;
  score: number;
  vector_score: number | null;
  bm25_score: number | null;
  document_id: string;
  page: number | null;
  citation: string;
  sourced: boolean;
  cited: boolean;
  used_in_context: boolean;
  preview: string;
  drop_reason: string | null;
}

export interface Timings {
  embed_ms: number;
  search_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface QueryResponse {
  question: string;
  status: "answered" | "partial" | "abstained";
  answer_type: UnitType | null;
  answer: string | null;
  abstain_reason: string | null;
  generation_error: string | null;
  artefact: Artefact | null;
  citation: Citation | null;
  cited_unit_ids: string[];
  sources: Source[];
  timings: Timings;
  model_calls: number;
  models: Record<string, string>;
}

export interface Problem {
  kind: "configuration" | "upstream" | "index" | "vector_store";
  message: string;
  fix: string;
}

export interface Health {
  status: "healthy" | "degraded";
  problems: Problem[];
  vector_backend: string;
  index: Record<string, unknown> | null;
  models: Record<string, string>;
  model_check: { checked_at: string; all_ok: boolean } | null;
}

export interface Example {
  question: string;
  type: string;
  qid?: string;
}

export interface ApiError {
  kind: "configuration" | "upstream" | "network" | "validation" | "unknown";
  message: string;
  fix?: string;
  model?: string;
}
