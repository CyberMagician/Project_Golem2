"""Local Flask server for a validated Project Golem 2 corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from .artifacts import ArtifactValidationError, LoadedArtifacts, load_validated_artifacts
from .audio import AudioError, audio_path_for
from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_AUDIO_CACHE_DIR, DEFAULT_MODEL_DIR, MODEL_SPEC
from .model import EmbeddingGemma2Encoder, ModelLoadError, normalize_vectors


class QueryError(ValueError):
    """A client query is malformed."""


class LocalQueryService:
    """Reload-and-validate artifacts before each use instead of serving stale vectors."""

    def __init__(
        self,
        artifact_dir: Path,
        model_dir: Path,
        audio_cache_dir: Path,
        query_encoder: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.model_dir = model_dir
        self.audio_cache_dir = audio_cache_dir
        self._query_encoder = query_encoder
        self._model: EmbeddingGemma2Encoder | None = None

    def artifacts(self) -> LoadedArtifacts:
        return load_validated_artifacts(self.artifact_dir)

    def _embed_query(self, text: str) -> np.ndarray:
        if self._query_encoder is not None:
            raw_vector = self._query_encoder(text)
        else:
            if self._model is None:
                self._model = EmbeddingGemma2Encoder(self.model_dir)
            raw_vector = self._model.embed_query(text)
        raw_vector = np.asarray(raw_vector, dtype=np.float32)
        if raw_vector.shape != (MODEL_SPEC.embedding_dimension,):
            raise ModelLoadError(
                f"Query encoder returned {raw_vector.shape}; expected ({MODEL_SPEC.embedding_dimension},)."
            )
        return normalize_vectors(raw_vector.reshape(1, -1), expected_rows=1)[0]

    def query(self, text: str, top_k: int) -> list[dict[str, object]]:
        artifacts = self.artifacts()
        vector = self._embed_query(text)
        scores = artifacts.vectors @ vector
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {"id": artifacts.nodes[index]["id"], "score": float(scores[index])}
            for index in indices
        ]


def _parse_query() -> tuple[str, int]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise QueryError("Request body must be a JSON object.")
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise QueryError("query must be a non-empty string.")
    query = query.strip()
    if len(query) > 2_000:
        raise QueryError("query must be at most 2000 characters.")
    top_k = data.get("top_k", 8)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise QueryError("top_k must be an integer from 1 through 50.")
    return query, top_k


def create_app(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    audio_cache_dir: Path = DEFAULT_AUDIO_CACHE_DIR,
    query_encoder: Callable[[str], np.ndarray] | None = None,
) -> Flask:
    """Create a server only after the stored corpus passes its complete contract."""
    service = LocalQueryService(artifact_dir, model_dir, audio_cache_dir, query_encoder)
    service.artifacts()
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")

    @app.get("/")
    def index() -> object:
        return send_from_directory(static_dir, "index.html")

    @app.get("/api/graph")
    def graph() -> object:
        artifacts = service.artifacts()
        return jsonify(
            {
                "schema_version": artifacts.metadata["schema_version"],
                "node_count": len(artifacts.nodes),
                "nodes": artifacts.nodes,
            }
        )

    @app.post("/api/query")
    def query() -> object:
        text, top_k = _parse_query()
        return jsonify({"matches": service.query(text, top_k)})

    @app.get("/api/audio/<node_id>")
    def audio(node_id: str) -> object:
        artifacts = service.artifacts()
        node = next((entry for entry in artifacts.nodes if entry["id"] == node_id), None)
        if node is None or node.get("audio") is None:
            return jsonify({"error": "No curated audio is available for this node."}), 404
        try:
            audio_path = audio_path_for(service.audio_cache_dir, node["audio"])
        except AudioError as error:
            return jsonify({"error": str(error)}), 503
        return send_from_directory(
            audio_path.parent,
            audio_path.name,
            mimetype=node["audio"]["mime"],
            conditional=True,
        )

    @app.errorhandler(QueryError)
    def query_error(error: QueryError) -> tuple[object, int]:
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(ArtifactValidationError)
    @app.errorhandler(ModelLoadError)
    def dependency_error(error: Exception) -> tuple[object, int]:
        return jsonify({"error": str(error)}), 503

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--audio-cache-dir", type=Path, default=DEFAULT_AUDIO_CACHE_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    application = create_app(
        artifact_dir=arguments.artifact_dir,
        model_dir=arguments.model_dir,
        audio_cache_dir=arguments.audio_cache_dir,
    )
    application.run(host=arguments.host, port=arguments.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
