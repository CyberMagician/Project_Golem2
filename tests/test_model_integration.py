"""Optional real model smoke test; excluded from default offline CI."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from golem2.config import DEFAULT_AUDIO_CACHE_DIR, DEFAULT_MODEL_DIR, MODEL_SPEC
from golem2.model import EmbeddingGemma2Encoder


pytestmark = pytest.mark.skipif(
    os.environ.get("GOLEM2_RUN_MODEL_INTEGRATION") != "1",
    reason="Set GOLEM2_RUN_MODEL_INTEGRATION=1 after downloading the pinned model and curated audio cache.",
)


def test_real_text_image_audio_embedding_is_finite_normalized_768d() -> None:
    audio_path = DEFAULT_AUDIO_CACHE_DIR / "ada-lovelace.ogg"
    if not DEFAULT_MODEL_DIR.is_dir() or not audio_path.is_file():
        pytest.skip("The pinned model and cached Ada Lovelace Commons audio are required.")
    encoder = EmbeddingGemma2Encoder(DEFAULT_MODEL_DIR)
    image = Image.new("RGB", (4, 4), color=(16, 32, 64))

    vector = encoder.embed_multimodal_documents(
        ["title: Ada Lovelace | text: English mathematician and computing pioneer."],
        [image],
        [str(audio_path)],
    )

    assert vector.shape == (1, MODEL_SPEC.embedding_dimension)
    assert np.isfinite(vector).all()
    assert np.allclose(np.linalg.norm(vector, axis=1), 1.0, rtol=1e-5, atol=1e-5)
