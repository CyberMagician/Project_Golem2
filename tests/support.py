"""Reusable, local-only artifact and HTTP fixture helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from golem2.artifacts import write_artifacts
from golem2.manifest import Target, load_manifest


FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload: dict[str, Any], *, status_code: int = 200, content_type: str = "application/json"):
        self.payload = payload
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            from requests import HTTPError

            raise HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)

    def close(self) -> None:
        return None


class FixtureSession:
    def __init__(self, en_payload: dict[str, Any] | None = None, commons_payload: dict[str, Any] | None = None):
        self.en_payload = en_payload or json.loads((FIXTURES / "en_article_redirect.json").read_text())
        self.commons_payload = commons_payload or json.loads((FIXTURES / "commons_ada.json").read_text())
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params: dict[str, Any] | None = None, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, params or {}))
        if url.endswith("/w/api.php") and "en.wikipedia.org" in url:
            return FakeResponse(self.en_payload)
        if url.endswith("/w/api.php") and "commons.wikimedia.org" in url:
            return FakeResponse(self.commons_payload)
        if url.startswith("https://upload.wikimedia.org/"):
            return FakeResponse({}, content_type="image/jpeg")
        raise AssertionError(f"Unexpected fixture request: {url}")


def image_provenance() -> dict[str, object]:
    return {
        "thumbnail_url": "https://upload.wikimedia.org/example/thumb.jpg",
        "source_url": "https://upload.wikimedia.org/example/source.jpg",
        "commons_file_url": "https://commons.wikimedia.org/wiki/File:Example.jpg",
        "commons_file_title": "File:Example.jpg",
        "author": "Example author",
        "credit": "Example credit",
        "license_name": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "license_terms": "Creative Commons Attribution-ShareAlike 4.0",
        "attribution_required": True,
        "mime": "image/jpeg",
        "width": 1000,
        "height": 700,
        "thumbnail_width": 600,
        "thumbnail_height": 420,
        "requested_thumbnail_width": 600,
        "page_image_title": "File:Example.jpg",
    }


def valid_nodes(targets: tuple[Target, ...] | None = None) -> list[dict[str, object]]:
    targets = targets or load_manifest()
    ids = [target.id for target in targets]
    return [
        {
            "id": target.id,
            "manifest_title": target.title,
            "title": target.title,
            "category": target.category,
            "color": target.color,
            "summary": f"Summary for {target.title}.",
            "article_url": f"https://en.wikipedia.org/wiki/{target.id}",
            "source_revision_id": 1,
            "source_revision_timestamp": "2026-01-01T00:00:00Z",
            "position": [float(index), 0.0, 0.0],
            "neighbors": [candidate for candidate in ids if candidate != target.id][:6],
            "image": image_provenance(),
            "audio": None,
        }
        for index, target in enumerate(targets)
    ]


def normalized_vectors(rows: int = 50) -> np.ndarray:
    vectors = np.zeros((rows, 768), dtype=np.float32)
    for index in range(rows):
        vectors[index, index] = 1.0
    return vectors


def build_valid_artifacts(destination: Path) -> Path:
    write_artifacts(valid_nodes(), normalized_vectors(), artifact_dir=destination)
    return destination
