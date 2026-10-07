"""Pinned EmbeddingGemma 2 loading and normalization rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import DEFAULT_MODEL_DIR, MODEL_SPEC, ModelSpec


class ModelLoadError(RuntimeError):
    """The local model cache does not match the artifact contract."""


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


def select_torch_dtype(torch_module: Any) -> Any:
    """Use BF16 only on CUDA hardware that explicitly supports it."""
    cuda = getattr(torch_module, "cuda", None)
    supports_bf16 = bool(
        cuda
        and callable(getattr(cuda, "is_available", None))
        and cuda.is_available()
        and callable(getattr(cuda, "is_bf16_supported", None))
        and cuda.is_bf16_supported()
    )
    return torch_module.bfloat16 if supports_bf16 else torch_module.float32


def normalize_vectors(vectors: Any, *, expected_rows: int | None = None) -> np.ndarray:
    """Return finite float32 vectors normalized for cosine/dot-product search."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != MODEL_SPEC.embedding_dimension:
        raise ModelLoadError(
            f"Expected vectors shaped (n, {MODEL_SPEC.embedding_dimension}), got {matrix.shape}."
        )
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ModelLoadError(f"Expected {expected_rows} vectors, got {matrix.shape[0]}.")
    if not np.isfinite(matrix).all():
        raise ModelLoadError("Embedding matrix contains non-finite values.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ModelLoadError("Embedding matrix contains a zero or invalid vector.")
    return matrix / norms


def verify_local_model(model_dir: Path = DEFAULT_MODEL_DIR, spec: ModelSpec = MODEL_SPEC) -> None:
    """Require a completion marker written by the pinned downloader."""
    marker_path = model_dir / ".golem2-model.json"
    config_path = model_dir / "config.json"
    if not marker_path.is_file() or not config_path.is_file():
        raise ModelLoadError(
            f"Pinned model is absent or incomplete at {model_dir}. "
            "Run: python scripts/download_model.py"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelLoadError(f"Invalid model completion marker: {error}") from error
    if marker.get("model_id") != spec.model_id or marker.get("revision") != spec.revision:
        raise ModelLoadError(
            "Local model identity/revision does not match the pinned EmbeddingGemma 2 contract."
        )


class EmbeddingGemma2Encoder:
    """Text-only adapter that always applies the model card's retrieval prompts."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, spec: ModelSpec = MODEL_SPEC):
        verify_local_model(model_dir, spec)
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ModelLoadError("Install requirements.txt before loading the embedding model.") from error

        self.spec = spec
        self.model = SentenceTransformer(
            str(model_dir),
            model_kwargs={"torch_dtype": select_torch_dtype(torch)},
        )

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """Embed manually formatted titled documents without a second title prefix."""
        vectors = self.model.encode(
            documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return normalize_vectors(vectors, expected_rows=len(documents))

    def embed_multimodal_documents(
        self,
        documents: list[str],
        images: list[Any],
        audios: list[Any | None],
    ) -> np.ndarray:
        """Embed titled text plus a vetted image and optional curated audio clip."""
        if len(documents) != len(images) or len(documents) != len(audios) or not documents:
            raise ModelLoadError("Multimodal documents, images, and audio must be non-empty and one-to-one.")
        inputs: list[dict[str, Any]] = []
        for document, image, audio in zip(documents, images, audios, strict=True):
            item: dict[str, Any] = {"text": f"{document} <|image|>", "image": image}
            if audio is not None:
                item["text"] += " <|audio|>"
                item["audio"] = audio
            inputs.append(item)
        vectors = self.model.encode(
            inputs,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return normalize_vectors(vectors, expected_rows=len(documents))

    def embed_query(self, query: str) -> np.ndarray:
        vectors = self.model.encode(
            [query],
            prompt_name=self.spec.query_prompt_name,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return normalize_vectors(vectors, expected_rows=1)[0]
