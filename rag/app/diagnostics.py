"""Model readiness: one tiny call per configured role, OK/FAIL with model name and reason (E0.1)."""
from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from .config import ConfigError, Settings
from .openrouter import KVCache, OpenRouterClient


def _tiny_image() -> Path:
    im = Image.new("RGB", (64, 64), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, 56, 56], outline="black", width=3)
    d.line([8, 8, 56, 56], fill="red", width=3)
    p = Path(tempfile.gettempdir()) / "rag_model_check.png"
    im.save(p)
    return p


def check_models(settings: Settings, roles: list[str] | None = None) -> dict:
    """Never cached: this is the live readiness check. Costs one call per role."""
    settings.require_key()
    # A throwaway cache so the check really calls the endpoint.
    cache = KVCache(Path(tempfile.gettempdir()) / "rag_model_check_cache.sqlite")
    client = OpenRouterClient(settings, cache)
    roles = roles or ["embeddings", "vision", "text", "figure"]
    results = []
    img = _tiny_image()
    for role in roles:
        model = settings.model_roles()[role]
        if role == "figure" and model == settings.vision_model and "vision" in roles:
            prev = next(r for r in results if r["role"] == "vision")
            results.append({**prev, "role": "figure", "notes": "same slug as vision; not called twice"})
            continue
        t0 = time.perf_counter()
        try:
            if role == "embeddings":
                v = client._embed_call([f"model readiness check {time.time()}"], role="check")[0]
                note = f"vector size {len(v)}"
            elif role in ("vision", "figure"):
                out = client.vision("What shapes and colours are in this image? Answer in 10 words.", [img],
                                    model=model, role="check", max_tokens=60)
                note = out[:80]
            else:
                out = client.chat([{"role": "user", "content": "Reply with the single word OK."}],
                                  model=model, role="check", max_tokens=20)
                note = out[:80]
            results.append({"role": role, "model": model, "ok": True,
                            "ms": round((time.perf_counter() - t0) * 1000), "notes": note})
        except ConfigError:
            raise
        except Exception as e:
            results.append({"role": role, "model": model, "ok": False,
                            "ms": round((time.perf_counter() - t0) * 1000), "notes": str(e)[:300]})
    client.close()
    out = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "all_ok": all(r["ok"] for r in results), "results": results}
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_check_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def last_model_check(settings: Settings) -> dict | None:
    p = settings.model_check_path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
