"""Build the exact 50-node local corpus from MediaWiki and EmbeddingGemma 2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from .artifacts import ArtifactPaths, write_artifacts
from .audio import CachedAudio, cache_curated_audio, load_audio_manifest
from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_AUDIO_CACHE_DIR, DEFAULT_MODEL_DIR
from .image_inputs import load_embedding_images
from .manifest import Target, load_manifest
from .mediawiki import MediaWikiResolver, ResolvedTarget
from .model import EmbeddingGemma2Encoder, normalize_vectors


class DocumentEncoder(Protocol):
    def embed_multimodal_documents(
        self,
        documents: list[str],
        images: list[Any],
        audios: list[Any | None],
    ) -> np.ndarray: ...


def project_coordinates(vectors: np.ndarray) -> np.ndarray:
    """Use deterministic PCA with a fixed sign convention for a stable 3D graph."""
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ right_vectors[:3].T
    for column in range(coordinates.shape[1]):
        pivot = int(np.argmax(np.abs(coordinates[:, column])))
        if coordinates[pivot, column] < 0:
            coordinates[:, column] *= -1
    scales = coordinates.std(axis=0)
    coordinates = coordinates / np.where(scales > 1e-8, scales, 1.0)
    return coordinates.astype(np.float32)


def neighbor_topology(vectors: np.ndarray, node_ids: list[str], count: int = 6) -> list[list[str]]:
    """Create a deterministic cosine-neighbor topology from normalized vectors."""
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    neighbor_count = min(count, len(node_ids) - 1)
    topology: list[list[str]] = []
    for index in range(len(node_ids)):
        selected = np.argsort(-similarity[index], kind="stable")[:neighbor_count]
        topology.append([node_ids[item] for item in selected])
    return topology


def _document_text(resolved: ResolvedTarget) -> str:
    return f"title: {resolved.article.title} | text: {resolved.article.summary}"


def build_corpus(
    targets: tuple[Target, ...],
    resolver: Any,
    encoder: DocumentEncoder,
    *,
    image_loader: Any = load_embedding_images,
    audio_by_target: Mapping[str, CachedAudio] | None = None,
    audio_cache_dir: Path = DEFAULT_AUDIO_CACHE_DIR,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    manifest_path: Path | None = None,
) -> ArtifactPaths:
    """Resolve every image first, then write one validated 50x768 corpus."""
    resolved = tuple(resolver.resolve_targets(targets))
    if len(resolved) != len(targets) or len(resolved) != 50:
        raise RuntimeError("Resolver did not return exactly the canonical 50 targets.")
    if [item.target.id for item in resolved] != [target.id for target in targets]:
        raise RuntimeError("Resolver changed the stable target order.")
    documents = [_document_text(item) for item in resolved]
    images = image_loader(resolved)
    if len(images) != len(resolved):
        raise RuntimeError("Image loader did not return one verified image per target.")
    audio_by_target = audio_by_target or {}
    unknown_audio_targets = set(audio_by_target) - {item.target.id for item in resolved}
    if unknown_audio_targets:
        raise RuntimeError(f"Audio metadata has unknown target ids: {sorted(unknown_audio_targets)}")
    audio_inputs = [
        str(audio_cache_dir / audio_by_target[item.target.id].cache_file)
        if item.target.id in audio_by_target
        else None
        for item in resolved
    ]
    vectors = normalize_vectors(
        encoder.embed_multimodal_documents(documents, images, audio_inputs),
        expected_rows=len(resolved),
    )
    coordinates = project_coordinates(vectors)
    neighbors = neighbor_topology(vectors, [target.id for target in targets])
    nodes: list[dict[str, Any]] = []
    for index, item in enumerate(resolved):
        node = {
            "id": item.target.id,
            "manifest_title": item.target.title,
            "title": item.article.title,
            "category": item.target.category,
            "color": item.target.color,
            "summary": item.article.summary,
            "article_url": item.article.article_url,
            "source_revision_id": item.article.source_revision_id,
            "source_revision_timestamp": item.article.source_revision_timestamp,
            "position": coordinates[index].tolist(),
            "neighbors": neighbors[index],
            "image": item.image.to_dict(),
            "audio": (
                audio_by_target[item.target.id].to_dict()
                if item.target.id in audio_by_target
                else None
            ),
        }
        nodes.append(node)
    return write_artifacts(nodes, vectors, artifact_dir=artifact_dir, manifest_path=manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--audio-cache-dir", type=Path, default=DEFAULT_AUDIO_CACHE_DIR)
    parser.add_argument("--manifest", type=Path)
    arguments = parser.parse_args()

    targets = load_manifest(arguments.manifest) if arguments.manifest else load_manifest()
    resolver = MediaWikiResolver()
    audio_targets = load_audio_manifest(valid_target_ids={target.id for target in targets})
    audio_by_target = cache_curated_audio(
        audio_targets,
        resolver,
        cache_dir=arguments.audio_cache_dir,
    )
    encoder = EmbeddingGemma2Encoder(arguments.model_dir)
    paths = build_corpus(
        targets,
        resolver,
        encoder,
        image_loader=load_embedding_images,
        audio_by_target=audio_by_target,
        audio_cache_dir=arguments.audio_cache_dir,
        artifact_dir=arguments.artifact_dir,
        manifest_path=arguments.manifest,
    )
    print(f"Built {len(targets)} nodes at {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
