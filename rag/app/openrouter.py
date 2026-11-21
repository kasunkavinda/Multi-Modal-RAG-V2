"""The only module that calls the model API (OpenRouter, OpenAI-compatible).

- throttles to REQUESTS_PER_MINUTE (free tier: 20/min),
- retries transient failures (429, 5xx, timeouts) with exponential backoff,
- raises readable errors that name the model (404 "No endpoints found" = retired slug),
- caches embeddings and vision outputs on disk keyed by content, so re-runs never pay twice,
- counts calls per role (Layer C cost proxy).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np
from PIL import Image

from .config import ConfigError, Settings

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ errors
class UpstreamError(RuntimeError):
    """The model provider failed (as opposed to our configuration)."""

    def __init__(self, model: str, message: str, status: int | None = None):
        super().__init__(f"[{model}] {message}")
        self.model = model
        self.status = status
        self.reason = message


class ModelUnavailable(UpstreamError):
    """Slug retired or no provider currently serving it."""


class EmptyResponse(UpstreamError):
    """Model returned no visible text (often: reasoning used the whole token budget)."""


# ------------------------------------------------------------------- cache
class KVCache:
    """Tiny content-addressed disk cache (sqlite, stdlib). Namespaces: embed, caption, ocr, table."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (ns TEXT, k TEXT, v TEXT, PRIMARY KEY (ns, k))")
        self._db.commit()

    @staticmethod
    def key(*parts: str | bytes) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p if isinstance(p, bytes) else p.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, ns: str, k: str) -> Any | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE ns=? AND k=?", (ns, k)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, ns: str, k: str, v: Any) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?)", (ns, k, json.dumps(v)))
            self._db.commit()

    def items(self, ns: str) -> list[tuple[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT k, v FROM kv WHERE ns=?", (ns,)).fetchall()
        return [(k, json.loads(v)) for k, v in rows]


# ----------------------------------------------------------------- helpers
def image_to_data_url(path: Path, max_side: int = 1600) -> str:
    """Downscale (long side <= max_side) and encode as a data URL."""
    with Image.open(path) as im:
        im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
        if max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def truncate_normalise(vec: list[float], dims: int | None) -> list[float]:
    """Matryoshka-style truncation, then L2-normalise (cosine-ready)."""
    v = np.asarray(vec, dtype=np.float32)
    if dims and dims < v.shape[0]:
        v = v[:dims]
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 0 else v.tolist()


def _error_text(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error", body)
        msg = err.get("message") if isinstance(err, dict) else str(err)
        meta = err.get("metadata") if isinstance(err, dict) else None
        if meta and isinstance(meta, dict) and meta.get("raw"):
            msg = f"{msg} ({str(meta['raw'])[:200]})"
        return str(msg)[:500]
    except Exception:
        return resp.text[:500]


# ------------------------------------------------------------------ client
class OpenRouterClient:
    _throttle_lock = threading.Lock()
    _last_call: dict[str, float] = {}

    def __init__(self, settings: Settings, cache: KVCache | None = None):
        settings.require_key()  # a missing key stops the run before any call
        self.s = settings
        self.cache = cache or KVCache(settings.cache_dir / "cache.sqlite")
        self.stats: Counter[str] = Counter()       # live calls per role
        self.cache_hits: Counter[str] = Counter()
        self._http = httpx.Client(
            base_url=settings.openrouter_base_url,
            timeout=settings.request_timeout_s,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                # Generic client identification only (no personal details).
                "X-Title": "multimodal-rag-lab",
            },
        )

    # ---------------------------------------------------------- transport
    def _throttle(self, model: str) -> None:
        """Free-tier limit is per model: keep >= 60/RPM seconds between calls to the same model."""
        min_gap = 60.0 / max(1, self.s.requests_per_minute)
        with self._throttle_lock:
            wait = self._last_call.get(model, 0.0) + min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call[model] = time.monotonic()

    def _post(self, path: str, payload: dict, *, model: str, role: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.s.max_retries + 1):
            self._throttle(model)
            self.stats[role] += 1
            try:
                resp = self._http.post(path, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = UpstreamError(model, f"network error: {e.__class__.__name__}: {e}")
                self._backoff(attempt, None)
                continue

            if resp.status_code == 200:
                body = resp.json()
                if isinstance(body, dict) and body.get("error"):
                    # OpenRouter can return provider errors inside a 200.
                    code = body["error"].get("code") if isinstance(body["error"], dict) else None
                    msg = _error_text(resp)
                    if code in (429, 502, 503, 524, 529) or code is None:
                        last_err = UpstreamError(model, f"provider error: {msg}", code)
                        self._backoff(attempt, None)
                        continue
                    raise UpstreamError(model, f"provider error: {msg}", code)
                return body

            msg = _error_text(resp)
            if resp.status_code == 401:
                raise ConfigError(
                    f"OpenRouter rejected the API key (401): {msg}",
                    fix="Check OPENROUTER_API_KEY in rag/.env (no spaces or quotes).",
                )
            if resp.status_code == 402:
                raise UpstreamError(model, f"payment required / out of credits (402): {msg}", 402)
            if resp.status_code == 404:
                raise ModelUnavailable(
                    model, f"no endpoints for this slug (404) - it is retired or offline: {msg}", 404
                )
            if resp.status_code in (408, 429, 500, 502, 503, 504, 524, 529):
                last_err = UpstreamError(model, f"HTTP {resp.status_code}: {msg}", resp.status_code)
                if resp.status_code == 429 and "per-day" in msg.lower():
                    raise UpstreamError(model, f"daily free-tier cap reached (429): {msg}", 429)
                self._backoff(attempt, resp.headers.get("retry-after"))
                continue
            raise UpstreamError(model, f"HTTP {resp.status_code}: {msg}", resp.status_code)
        assert last_err is not None
        raise last_err

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> None:
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        time.sleep(min(60.0, max(delay, 2.0 * (2**attempt))))

    # --------------------------------------------------------------- chat
    def chat(
        self,
        messages: list[dict],
        *,
        model: str,
        role: str = "text",
        max_tokens: int = 600,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.s.reasoning_off:
            payload["reasoning"] = {"enabled": False}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            body = self._post("/chat/completions", payload, model=model, role=role)
        except UpstreamError as e:
            # Some models have mandatory reasoning or reject json mode: retry once without them.
            if e.status == 400 and ("reasoning" in e.reason.lower() or "response_format" in e.reason.lower()):
                payload.pop("reasoning", None)
                payload.pop("response_format", None)
                payload["max_tokens"] = max_tokens * 4
                body = self._post("/chat/completions", payload, model=model, role=role)
            else:
                raise
        try:
            content = body["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise UpstreamError(model, f"unexpected response shape: {str(body)[:300]}")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = content.strip()
        if not content:
            finish = body["choices"][0].get("finish_reason")
            raise EmptyResponse(
                model,
                f"empty answer (finish_reason={finish}); reasoning may have used the token budget - "
                "switch reasoning off or raise max_tokens",
            )
        return content

    def vision(
        self,
        prompt: str,
        image_paths: Iterable[Path],
        *,
        model: str,
        role: str = "vision",
        max_tokens: int = 600,
        system: str | None = None,
    ) -> str:
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for p in image_paths:
            parts.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(p, self.s.figure_max_side)}}
            )
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": parts}
        ]
        return self.chat(messages, model=model, role=role, max_tokens=max_tokens)

    def vision_with_fallback(self, prompt: str, image_paths: list[Path], *, role: str = "vision",
                             max_tokens: int = 600) -> tuple[str, str]:
        """Try VISION_MODEL then VISION_FALLBACKS. Returns (text, model_used)."""
        errors = []
        for model in [self.s.vision_model, *self.s.vision_fallbacks]:
            try:
                return self.vision(prompt, image_paths, model=model, role=role, max_tokens=max_tokens), model
            except (ModelUnavailable, EmptyResponse, UpstreamError) as e:
                errors.append(str(e))
                log.warning("vision model failed, trying next: %s", e)
        raise UpstreamError(self.s.vision_model, "all vision models failed: " + " | ".join(errors))

    # --------------------------------------------------------- embeddings
    def _embed_key(self, kind: str, content: str | bytes) -> str:
        return KVCache.key(kind, self.s.embed_model, content)

    def embed_texts(self, texts: list[str], *, is_query: bool = False, role: str = "embeddings") -> list[list[float]]:
        """Batched, cached text embeddings (full vectors cached; truncation applied on read)."""
        prefix = self.s.embed_query_prefix if is_query else self.s.embed_doc_prefix
        inputs = [prefix + t for t in texts]
        out: list[list[float] | None] = [None] * len(inputs)
        todo: list[int] = []
        for i, t in enumerate(inputs):
            cached = self.cache.get("embed", self._embed_key("text", t))
            if cached is not None:
                out[i] = cached
                self.cache_hits["embeddings"] += 1
            else:
                todo.append(i)
        bs = max(1, self.s.embed_batch_size)
        for start in range(0, len(todo), bs):
            idxs = todo[start : start + bs]
            vecs = self._embed_call([inputs[i] for i in idxs], role=role)
            for i, v in zip(idxs, vecs):
                self.cache.put("embed", self._embed_key("text", inputs[i]), v)
                out[i] = v
        return [truncate_normalise(v, self.s.embed_dimensions) for v in out]  # type: ignore[arg-type]

    def embed_images(self, paths: list[Path], role: str = "embeddings") -> list[list[float]]:
        """Embed figure images directly with the multimodal embedder (E3.4 stretch)."""
        out = []
        for p in paths:
            raw = p.read_bytes()
            k = self._embed_key("image", raw)
            v = self.cache.get("embed", k)
            if v is None:
                url = image_to_data_url(p, 1024)
                v = self._embed_call([{"content": [{"type": "image_url", "image_url": {"url": url}}]}], role=role)[0]
                self.cache.put("embed", k, v)
            else:
                self.cache_hits["embeddings"] += 1
            out.append(truncate_normalise(v, self.s.embed_dimensions))
        return out

    def _embed_call(self, inputs: list, role: str) -> list[list[float]]:
        model = self.s.embed_model
        body = self._post("/embeddings", {"model": model, "input": inputs, "encoding_format": "float"},
                          model=model, role=role)
        data = body.get("data")
        if not data or len(data) != len(inputs):
            raise UpstreamError(model, f"embeddings response had {len(data or [])} vectors for {len(inputs)} inputs")
        data = sorted(data, key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    def close(self) -> None:
        self._http.close()
