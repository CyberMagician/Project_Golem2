import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from golem2.artifacts import write_artifacts
from golem2.audio import AudioError, AudioTarget, audio_path_for, cache_curated_audio, load_audio_manifest
from golem2.mediawiki import MediaWikiResolver, ResolutionError
from golem2.server import create_app

from .support import FIXTURES, FakeResponse, FixtureSession, normalized_vectors, valid_nodes


class AudioResponse:
    def __init__(self, payload: bytes, content_type: str = "audio/ogg"):
        self.payload = payload
        self.status_code = 200
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(payload))}

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self.payload

    def close(self) -> None:
        return None


class AudioSession(FixtureSession):
    def __init__(self, commons_payload: dict[str, Any], audio: bytes = b"OggS-example-audio-data"):
        super().__init__(commons_payload=commons_payload)
        self.audio = audio

    def get(self, url: str, *, params=None, **kwargs):
        if url == "https://upload.wikimedia.org/example/Ada_Lovelace.ogg":
            return AudioResponse(self.audio)
        return super().get(url, params=params, **kwargs)


def test_curated_audio_manifest_is_scoped_to_three_canonical_nodes() -> None:
    clips = load_audio_manifest(valid_target_ids={node["id"] for node in valid_nodes()})

    assert {clip.target_id for clip in clips} == {"ada-lovelace", "astronomy", "music-theory"}
    assert all(clip.commons_file_title.startswith("File:") for clip in clips)


def test_audio_cache_resolves_attribution_and_revalidates_checksum(tmp_path: Path) -> None:
    commons = json.loads((FIXTURES / "commons_audio.json").read_text())
    session = AudioSession(commons)
    resolver = MediaWikiResolver(session=session, retry_delay_seconds=0)
    cached = cache_curated_audio(
        [AudioTarget("ada-lovelace", "File:Ada Lovelace.ogg")],
        resolver,
        cache_dir=tmp_path,
        session=session,
    )["ada-lovelace"]

    assert cached.mime == "audio/ogg"
    assert cached.author == "Example narrator"
    assert cached.license_name == "CC BY 4.0"
    assert cached.sha256 == hashlib.sha256(session.audio).hexdigest()
    assert audio_path_for(tmp_path, cached.to_dict()).read_bytes() == session.audio

    (tmp_path / cached.cache_file).write_bytes(b"tampered")
    with pytest.raises(AudioError, match="unexpected size|checksum"):
        audio_path_for(tmp_path, cached.to_dict())


def test_audio_resolver_rejects_non_audio_source() -> None:
    commons = json.loads((FIXTURES / "commons_audio.json").read_text())
    commons["query"]["pages"][0]["imageinfo"][0]["mime"] = "video/webm"
    resolver = MediaWikiResolver(session=FixtureSession(commons_payload=commons), retry_delay_seconds=0)

    with pytest.raises(ResolutionError, match="not an audio"):
        resolver.fetch_commons_audio(["File:Ada Lovelace.ogg"])


def test_server_serves_only_checksum_validated_curated_audio(tmp_path: Path) -> None:
    cache_dir = tmp_path / "audio"
    cache_dir.mkdir()
    contents = b"OggS-test"
    cache_file = cache_dir / "ada-lovelace.ogg"
    cache_file.write_bytes(contents)
    nodes = valid_nodes()
    nodes[0]["audio"] = {
        "cache_file": cache_file.name,
        "commons_file_title": "File:Ada Lovelace.ogg",
        "commons_file_url": "https://commons.wikimedia.org/wiki/File:Ada_Lovelace.ogg",
        "source_url": "https://upload.wikimedia.org/example/Ada_Lovelace.ogg",
        "mime": "audio/ogg",
        "size": len(contents),
        "author": "Example narrator",
        "credit": "Example Commons recording",
        "license_name": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_terms": "Creative Commons Attribution 4.0",
        "attribution_required": True,
        "sha256": hashlib.sha256(contents).hexdigest(),
        "byte_count": len(contents),
    }
    artifact_dir = tmp_path / "artifacts"
    write_artifacts(nodes, normalized_vectors(), artifact_dir=artifact_dir)

    app = create_app(
        artifact_dir=artifact_dir,
        model_dir=tmp_path / "model",
        audio_cache_dir=cache_dir,
        query_encoder=lambda _: np.eye(1, 768, 0, dtype=np.float32)[0],
    )
    response = app.test_client().get("/api/audio/ada-lovelace")
    assert response.status_code == 200
    assert response.data == contents
