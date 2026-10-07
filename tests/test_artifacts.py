import json

import numpy as np
import pytest

from golem2.artifacts import ArtifactValidationError, load_validated_artifacts
from golem2.ingest import build_corpus
from golem2.manifest import load_manifest
from golem2.mediawiki import ArticlePage, ImageProvenance, ResolvedTarget

from .support import build_valid_artifacts, image_provenance, normalized_vectors


def test_artifacts_require_normalized_50_by_768_vectors_and_matching_model(tmp_path) -> None:
    artifact_dir = build_valid_artifacts(tmp_path / "artifacts")
    loaded = load_validated_artifacts(artifact_dir)

    assert loaded.vectors.shape == (50, 768)
    assert np.all(np.isfinite(loaded.vectors))
    assert np.allclose(np.linalg.norm(loaded.vectors, axis=1), 1.0)

    metadata_path = artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model"]["revision"] = "not-the-pinned-revision"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ArtifactValidationError, match="different model"):
        load_validated_artifacts(artifact_dir)


class FakeResolver:
    def resolve_targets(self, targets):
        return tuple(
            ResolvedTarget(
                target=target,
                article=ArticlePage(
                    requested_title=target.title,
                    title=target.title,
                    summary=f"Summary for {target.title}",
                    article_url=f"https://en.wikipedia.org/wiki/{target.id}",
                    source_revision_id=1,
                    source_revision_timestamp="2026-01-01T00:00:00Z",
                    page_image_title="File:Example.jpg",
                ),
                image=ImageProvenance(**image_provenance()),
            )
            for target in targets
        )


class FakeEncoder:
    def embed_multimodal_documents(self, documents, images, audios):
        assert all(document.startswith("title: ") and " | text: " in document for document in documents)
        assert len(images) == len(documents)
        assert len(audios) == len(documents)
        return normalized_vectors(len(documents))


def test_builder_writes_exact_validated_corpus(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    paths = build_corpus(
        load_manifest(),
        FakeResolver(),
        FakeEncoder(),
        image_loader=lambda resolved: [object() for _ in resolved],
        artifact_dir=artifact_dir,
    )
    loaded = load_validated_artifacts(paths.root)

    assert len(loaded.nodes) == 50
    assert loaded.nodes[0]["neighbors"]
    assert all(len(node["position"]) == 3 for node in loaded.nodes)
