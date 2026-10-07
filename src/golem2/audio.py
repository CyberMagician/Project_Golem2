"""Curated, attributable Commons audio for optional multimodal corpus enrichment."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from .config import DEFAULT_AUDIO_CACHE_DIR, DEFAULT_AUDIO_MANIFEST_PATH
from .mediawiki import CommonsAudio, MediaWikiResolver, REQUEST_TIMEOUT, trusted_wikimedia_url


AUDIO_MANIFEST_SCHEMA_VERSION = 1
MAX_AUDIO_BYTES = 32 * 1024 * 1024
_EXTENSION_BY_MIME = {
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
}


class AudioError(RuntimeError):
    """The curated optional audio layer is malformed, unsafe, or unavailable."""


@dataclass(frozen=True)
class AudioTarget:
    target_id: str
    commons_file_title: str


@dataclass(frozen=True)
class CachedAudio:
    target_id: str
    cache_file: str
    commons_file_title: str
    commons_file_url: str
    source_url: str
    mime: str
    size: int | None
    author: str
    credit: str
    license_name: str
    license_url: str
    license_terms: str
    attribution_required: bool
    sha256: str
    byte_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_file": self.cache_file,
            "commons_file_title": self.commons_file_title,
            "commons_file_url": self.commons_file_url,
            "source_url": self.source_url,
            "mime": self.mime,
            "size": self.size,
            "author": self.author,
            "credit": self.credit,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "license_terms": self.license_terms,
            "attribution_required": self.attribution_required,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }


def load_audio_manifest(
    path: Path = DEFAULT_AUDIO_MANIFEST_PATH,
    *,
    valid_target_ids: set[str] | None = None,
) -> tuple[AudioTarget, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AudioError(f"Cannot read curated audio manifest at {path}: {error}") from error
    if document.get("schema_version") != AUDIO_MANIFEST_SCHEMA_VERSION:
        raise AudioError("Curated audio manifest has an unsupported schema version.")
    clips = document.get("clips")
    if not isinstance(clips, list) or not clips:
        raise AudioError("Curated audio manifest must contain at least one clip.")
    targets: list[AudioTarget] = []
    for clip in clips:
        if not isinstance(clip, dict):
            raise AudioError("Each curated audio entry must be an object.")
        target_id = clip.get("target_id")
        title = clip.get("commons_file_title")
        if not isinstance(target_id, str) or not target_id:
            raise AudioError("Curated audio entry has an invalid target id.")
        if not isinstance(title, str) or not title.startswith("File:"):
            raise AudioError("Curated audio entry must reference a Commons File: title.")
        if valid_target_ids is not None and target_id not in valid_target_ids:
            raise AudioError(f"Curated audio references unknown graph target: {target_id}")
        targets.append(AudioTarget(target_id, title))
    if len({item.target_id for item in targets}) != len(targets):
        raise AudioError("Curated audio target ids must be unique.")
    if len({item.commons_file_title for item in targets}) != len(targets):
        raise AudioError("Curated Commons audio files must be unique.")
    return tuple(targets)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_cache_file(target_id: str, mime: str) -> str:
    extension = _EXTENSION_BY_MIME.get(mime.casefold())
    if extension is None:
        raise AudioError(f"Unsupported curated audio MIME type: {mime}")
    return f"{target_id}{extension}"


def _download_audio(source: CommonsAudio, destination: Path, session: requests.Session) -> tuple[str, int]:
    response: Any | None = None
    temporary: Path | None = None
    try:
        response = session.get(
            source.source_url,
            headers={"Accept": "audio/*"},
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].casefold()
        if not content_type.startswith("audio/"):
            raise AudioError("Curated audio response did not have an audio MIME type.")
        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > MAX_AUDIO_BYTES:
            raise AudioError("Curated audio exceeds the maximum allowed download size.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".audio-", dir=destination.parent)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        byte_count = 0
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                byte_count += len(chunk)
                if byte_count > MAX_AUDIO_BYTES:
                    raise AudioError("Curated audio exceeds the maximum allowed download size.")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if byte_count == 0:
            raise AudioError("Curated audio download was empty.")
        os.replace(temporary, destination)
        return digest.hexdigest(), byte_count
    except requests.RequestException as error:
        raise AudioError(f"Could not download curated Commons audio: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if response is not None:
            response.close()


def cache_curated_audio(
    items: Iterable[AudioTarget],
    resolver: MediaWikiResolver,
    *,
    cache_dir: Path = DEFAULT_AUDIO_CACHE_DIR,
    session: requests.Session | None = None,
) -> dict[str, CachedAudio]:
    """Fetch licensed Commons files once and revalidate cache checksums on reuse."""
    entries = tuple(items)
    metadata = resolver.fetch_commons_audio(entry.commons_file_title for entry in entries)
    client = session or requests.Session()
    cached: dict[str, CachedAudio] = {}
    for entry in entries:
        source = metadata[entry.commons_file_title]
        cache_file = _safe_cache_file(entry.target_id, source.mime)
        destination = cache_dir / cache_file
        marker_path = cache_dir / f"{cache_file}.json"
        marker: dict[str, Any] | None = None
        if destination.is_file() and marker_path.is_file():
            try:
                candidate = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                candidate = None
            if (
                isinstance(candidate, dict)
                and candidate.get("source_url") == source.source_url
                and candidate.get("mime") == source.mime
                and candidate.get("sha256") == _sha256(destination)
                and candidate.get("byte_count") == destination.stat().st_size
            ):
                marker = candidate
        if marker is None:
            digest, byte_count = _download_audio(source, destination, client)
            marker = {
                "source_url": source.source_url,
                "mime": source.mime,
                "sha256": digest,
                "byte_count": byte_count,
            }
            marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        cached[entry.target_id] = CachedAudio(
            target_id=entry.target_id,
            cache_file=cache_file,
            commons_file_title=source.title,
            commons_file_url=source.file_url,
            source_url=source.source_url,
            mime=source.mime,
            size=source.size,
            author=source.author,
            credit=source.credit,
            license_name=source.license_name,
            license_url=source.license_url,
            license_terms=source.license_terms,
            attribution_required=source.attribution_required,
            sha256=marker["sha256"],
            byte_count=marker["byte_count"],
        )
    return cached


def audio_path_for(cache_dir: Path, audio: dict[str, object]) -> Path:
    """Return a validated cache path without allowing artifact path traversal."""
    cache_file = audio.get("cache_file")
    if not isinstance(cache_file, str) or Path(cache_file).name != cache_file:
        raise AudioError("Audio artifact has an unsafe cache file name.")
    path = cache_dir / cache_file
    if not path.is_file():
        raise AudioError(f"Required audio cache file is missing: {path}")
    byte_count = audio.get("byte_count")
    checksum = audio.get("sha256")
    if not isinstance(byte_count, int) or path.stat().st_size != byte_count:
        raise AudioError(f"Audio cache file has an unexpected size: {path}")
    if not isinstance(checksum, str) or _sha256(path) != checksum:
        raise AudioError(f"Audio cache file checksum does not match the artifact: {path}")
    return path
