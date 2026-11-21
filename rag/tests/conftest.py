"""Shared fixtures. Tests never touch the network: a FakeClient stands in for OpenRouter."""
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.schemas import Provenance, TableData, Unit  # noqa: E402


class FakeClient:
    """Deterministic bag-of-words embedder + canned chat. Mirrors the OpenRouterClient surface."""

    def __init__(self, answer="The figure shows it.", fail=None, dim=64):
        self.stats = Counter()
        self.cache_hits = Counter()
        self.answer = answer
        self.fail = fail
        self.dim = dim

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_texts(self, texts, is_query=False, role="embeddings"):
        self.stats[role] += 1
        return [self._vec(t) for t in texts]

    def chat(self, messages, model, role="text", max_tokens=600, temperature=0.0, json_mode=False):
        self.stats[role] += 1
        if self.fail:
            raise self.fail
        return self.answer

    def vision(self, prompt, image_paths, model, role="vision", max_tokens=600, system=None):
        self.stats[role] += 1
        if self.fail:
            raise self.fail
        return self.answer

    def vision_with_fallback(self, prompt, image_paths, role="vision", max_tokens=600):
        return self.vision(prompt, image_paths, model="fake", role=role), "fake"

    def close(self):
        pass


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(openrouter_api_key="", data_dir=tmp_path / "data", vector_backend="memory",
                 embed_model="fake-embed", embed_dimensions=None)
    s.data_dir.mkdir(parents=True)
    return s


@pytest.fixture
def figure_png(settings) -> Path:
    from PIL import Image

    p = settings.data_dir / "artefacts" / "figures" / "f.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 150), "white").save(p)
    return p


def make_units(settings, figure_png):
    fig = Unit(unit_id="docA:p3:fig1", type="figure", image_path="artefacts/figures/f.png",
               caption="Figure 3. Relay control wiring",
               search_text="relay control wiring diagram contactor K1 coil terminals",
               provenance=Provenance(document_id="docA", page=3, unit_id="docA:p3:fig1"),
               meta={"vision_caption": "Wiring diagram of the relay control circuit"})
    tab = Unit(unit_id="docA:p4:tab1", type="table",
               table=TableData(columns=["item", "part", "torque"],
                               rows=[["1", "bolt M8", "25 Nm"], ["2", "nut", "10 Nm"]]),
               search_text="torque specification bolt M8 25 Nm nut 10 Nm",
               provenance=Provenance(document_id="docA", page=4, unit_id="docA:p4:tab1"))
    txt = Unit(unit_id="docA:p1:t1", type="text", text="Disconnect battery before servicing the pump.",
               search_text="Disconnect battery before servicing the pump.",
               provenance=Provenance(document_id="docA", page=1, unit_id="docA:p1:t1"))
    return [fig, tab, txt]
