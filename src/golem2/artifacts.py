"""Versioned corpus artifact writing and strict compatibility validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import ARTIFACT_SCHEMA_VERSION, DEFAULT_ARTIFACT_DIR, MODEL_SPEC
from .manifest import Target, load_manifest
from .mediawiki import allows_missing_license_url, safe_https_url, trusted_wikimedia_url
from .model import ModelLoadError, normalize_vectors


class ArtifactValidationError(RuntimeError):
    """Generated files are missing, changed, or incompatible with this project."""


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    corpus: Path
    vectors: Path
    metadata: Path

    @classmethod
    def in_directory(cls, root: Path = DEFAULT_ARTIFACT_DIR) -> "ArtifactPaths":
        return cls(root=root, corpus=root / "corpus.json", vectors=root / "vectors.npy", metadata=root / "metadata.json")


@dataclass(frozen=True)
class LoadedArtifacts:
    nodes: tuple[dict[str, Any], ...]
    vectors: np.ndarray
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"Cannot read artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"Artifact {path} must be a JSON object.")
    return value


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"Node has no valid {key}.")
    return value


def _validate_nodes(nodes: Any, targets: tuple[Target, ...]) -> tuple[dict[str, Any], ...]:
    if not isinstance(nodes, list) or len(nodes) != len(targets):
        raise ArtifactValidationError(f"Corpus must contain exactly {len(targets)} graph nodes.")
    expected_ids = [target.id for target in targets]
    target_by_id = {target.id: target for target in targets}
    node_ids: list[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            raise ArtifactValidationError("Each corpus node must be an object.")
        node_id = _require_string(node, "id")
        node_ids.append(node_id)
        target = target_by_id.get(node_id)
        if target is None:
            raise ArtifactValidationError(f"Corpus has unknown node id: {node_id}")
        if node.get("manifest_title") != target.title:
            raise ArtifactValidationError(f"Node {node_id} does not match its manifest title.")
        if node.get("category") != target.category or node.get("color") != target.color:
            raise ArtifactValidationError(f"Node {node_id} has changed manifest display metadata.")
        _require_string(node, "title")
        _require_string(node, "summary")
        if not trusted_wikimedia_url(node.get("article_url")):
            raise ArtifactValidationError(f"Node {node_id} has an invalid article URL.")
        position = node.get("position")
        if (
            not isinstance(position, list)
            or len(position) != 3
            or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in position)
        ):
            raise ArtifactValidationError(f"Node {node_id} has invalid graph coordinates.")
        neighbors = node.get("neighbors")
        if (
            not isinstance(neighbors, list)
            or node_id in neighbors
            or len(neighbors) != len(set(neighbors))
            or not all(isinstance(neighbor, str) and neighbor in target_by_id for neighbor in neighbors)
        ):
            raise ArtifactValidationError(f"Node {node_id} has invalid neighbor topology.")
        provenance = node.get("image")
        if not isinstance(provenance, dict):
            raise ArtifactValidationError(f"Node {node_id} has no image provenance.")
        for key in ("thumbnail_url", "source_url", "commons_file_url"):
            if not trusted_wikimedia_url(provenance.get(key)):
                raise ArtifactValidationError(f"Node {node_id} has an invalid image {key}.")
        if not (
            safe_https_url(provenance.get("license_url"))
            or (
                provenance.get("license_url") is None
                and allows_missing_license_url(
                    _require_string(provenance, "license_name"),
                    _require_string(provenance, "license_terms"),
                )
            )
        ):
            raise ArtifactValidationError(f"Node {node_id} has an invalid image license_url.")
        for key in ("commons_file_title", "license_name", "license_terms", "mime"):
            _require_string(provenance, key)
        author = provenance.get("author")
        credit = provenance.get("credit")
        if not (
            isinstance(author, str)
            and author.strip()
            or isinstance(credit, str)
            and credit.strip()
        ):
            raise ArtifactValidationError(f"Node {node_id} has no image author or credit.")
        audio = node.get("audio")
        if audio is not None:
            if not isinstance(audio, dict):
                raise ArtifactValidationError(f"Node {node_id} audio metadata must be an object.")
            for key in ("commons_file_url", "source_url"):
                if not trusted_wikimedia_url(audio.get(key)):
                    raise ArtifactValidationError(f"Node {node_id} has an invalid audio {key}.")
            if not (
                safe_https_url(audio.get("license_url"))
                or (
                    audio.get("license_url") is None
                    and allows_missing_license_url(
                        _require_string(audio, "license_name"),
                        _require_string(audio, "license_terms"),
                    )
                )
            ):
                raise ArtifactValidationError(f"Node {node_id} has an invalid audio license_url.")
            for key in (
                "cache_file",
                "commons_file_title",
                "license_name",
                "license_terms",
                "mime",
                "sha256",
            ):
                _require_string(audio, key)
            cache_file = audio.get("cache_file")
            if Path(cache_file).name != cache_file:
                raise ArtifactValidationError(f"Node {node_id} has an unsafe audio cache file.")
            if not isinstance(audio.get("byte_count"), int) or audio["byte_count"] <= 0:
                raise ArtifactValidationError(f"Node {node_id} has an invalid audio byte count.")
            author = audio.get("author")
            credit = audio.get("credit")
            if not (
                isinstance(author, str)
                and author.strip()
                or isinstance(credit, str)
                and credit.strip()
            ):
                raise ArtifactValidationError(f"Node {node_id} audio has no author or credit.")
    if node_ids != expected_ids:
        raise ArtifactValidationError("Corpus node order must match the canonical manifest order.")
    return tuple(nodes)


def _validate_metadata(metadata: dict[str, Any], paths: ArtifactPaths, node_count: int) -> None:
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError("Artifact schema version is unsupported.")
    if metadata.get("model") != MODEL_SPEC.to_dict():
        raise ArtifactValidationError("Artifacts were built with a different model, revision, or prompt contract.")
    if metadata.get("node_count") != node_count:
        raise ArtifactValidationError("Artifact metadata node count is incorrect.")
    if metadata.get("embedding_dimension") != MODEL_SPEC.embedding_dimension:
        raise ArtifactValidationError("Artifact embedding dimension is incompatible.")
    if metadata.get("embedding_modalities") != list(MODEL_SPEC.embedding_modalities):
        raise ArtifactValidationError("Artifact embedding modalities are incompatible.")
    audio_count = metadata.get("audio_node_count")
    if not isinstance(audio_count, int) or not 0 <= audio_count <= node_count:
        raise ArtifactValidationError("Artifact audio node count is invalid.")
    if metadata.get("vectors_sha256") != _sha256(paths.vectors):
        raise ArtifactValidationError("Vector artifact checksum does not match metadata.")
    if metadata.get("corpus_sha256") != _sha256(paths.corpus):
        raise ArtifactValidationError("Corpus artifact checksum does not match metadata.")


def load_validated_artifacts(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    *,
    manifest_path: Path | None = None,
) -> LoadedArtifacts:
    """Load only exact, normalized 50x768 artifacts built by this model revision."""
    paths = ArtifactPaths.in_directory(artifact_dir)
    if not all(path.is_file() for path in (paths.corpus, paths.vectors, paths.metadata)):
        raise ArtifactValidationError(
            f"Missing corpus artifacts in {artifact_dir}. Run: python -m golem2.ingest"
        )
    targets = load_manifest(manifest_path) if manifest_path else load_manifest()
    corpus = _read_json(paths.corpus)
    if corpus.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError("Corpus schema version is unsupported.")
    nodes = _validate_nodes(corpus.get("nodes"), targets)
    metadata = _read_json(paths.metadata)
    _validate_metadata(metadata, paths, len(nodes))
    try:
        vectors = np.load(paths.vectors, allow_pickle=False)
        normalized = normalize_vectors(vectors, expected_rows=len(nodes))
    except (OSError, ValueError, ModelLoadError) as error:
        raise ArtifactValidationError(f"Invalid vector matrix: {error}") from error
    if not np.allclose(normalized, vectors, rtol=1e-4, atol=1e-5):
        raise ArtifactValidationError("Vector matrix must already be L2-normalized.")
    return LoadedArtifacts(nodes=nodes, vectors=normalized, metadata=metadata)


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_artifacts(
    nodes: Iterable[dict[str, Any]],
    vectors: Any,
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    manifest_path: Path | None = None,
) -> ArtifactPaths:
    """Write a corpus and its matrix atomically enough that metadata publishes last."""
    paths = ArtifactPaths.in_directory(artifact_dir)
    targets = load_manifest(manifest_path) if manifest_path else load_manifest()
    node_list = list(nodes)
    _validate_nodes(node_list, targets)
    try:
        normalized = normalize_vectors(vectors, expected_rows=len(node_list))
    except ModelLoadError as error:
        raise ArtifactValidationError(str(error)) from error

    corpus = {"schema_version": ARTIFACT_SCHEMA_VERSION, "nodes": node_list}
    _write_atomic(paths.corpus, json.dumps(corpus, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    paths.root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".vectors.", suffix=".npy", dir=paths.root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, paths.vectors)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model": MODEL_SPEC.to_dict(),
        "node_count": len(node_list),
        "embedding_dimension": MODEL_SPEC.embedding_dimension,
        "embedding_modalities": list(MODEL_SPEC.embedding_modalities),
        "audio_node_count": sum(node.get("audio") is not None for node in node_list),
        "vectors_sha256": _sha256(paths.vectors),
        "corpus_sha256": _sha256(paths.corpus),
    }
    _write_atomic(paths.metadata, json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    load_validated_artifacts(artifact_dir, manifest_path=manifest_path)
    return paths
