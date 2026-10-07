"""Shared, deliberately fixed compatibility settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "embeddinggemma-2"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "targets.json"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    embedding_dimension: int
    corpus_format: str
    query_prompt_name: str
    embedding_modalities: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "embedding_dimension": self.embedding_dimension,
            "corpus_format": self.corpus_format,
            "query_prompt_name": self.query_prompt_name,
            "embedding_modalities": list(self.embedding_modalities),
        }


MODEL_SPEC = ModelSpec(
    model_id="google/embeddinggemma-2",
    revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    embedding_dimension=768,
    corpus_format="title: {title} | text: {content}",
    query_prompt_name="SearchQuery",
    embedding_modalities=("text", "image"),
)

ARTIFACT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
USER_AGENT = (
    "ProjectGolem2/1.0 "
    "(https://github.com/CyberMagician/Project_Golem2; local Wikipedia graph)"
)
