"""Bounded, verified Wikimedia thumbnail loading for multimodal corpus embeddings."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Iterable

import requests

from .mediawiki import REQUEST_TIMEOUT, ResolvedTarget, trusted_wikimedia_url


MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000


class ImageInputError(RuntimeError):
    """A persisted Commons thumbnail cannot safely be used as a model image input."""


def _read_bounded(response: Any) -> bytes:
    declared_length = response.headers.get("Content-Length")
    if declared_length and declared_length.isdigit() and int(declared_length) > MAX_IMAGE_BYTES:
        raise ImageInputError("Thumbnail exceeds the maximum allowed download size.")
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        if size > MAX_IMAGE_BYTES:
            raise ImageInputError("Thumbnail exceeds the maximum allowed download size.")
        chunks.append(chunk)
    return b"".join(chunks)


def load_wikimedia_image(url: str, *, session: requests.Session | None = None) -> Any:
    """Decode a persisted Wikimedia thumbnail as RGB after size and MIME checks."""
    trusted_url = trusted_wikimedia_url(url)
    if not trusted_url:
        raise ImageInputError("Thumbnail URL is not a trusted Wikimedia HTTPS URL.")
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise ImageInputError("Pillow is required for multimodal image embeddings.") from error

    client = session or requests.Session()
    try:
        response = client.get(
            trusted_url,
            headers={"Accept": "image/*"},
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].casefold()
        if not content_type.startswith("image/"):
            raise ImageInputError("Thumbnail response is not an image.")
        data = _read_bounded(response)
    except requests.RequestException as error:
        raise ImageInputError(f"Could not download Commons thumbnail: {error}") from error
    finally:
        close = locals().get("response")
        if close is not None and callable(getattr(close, "close", None)):
            close.close()

    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ImageInputError("Thumbnail could not be decoded as an image.") from error
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ImageInputError("Thumbnail exceeds the maximum allowed pixel count.")
    return image.convert("RGB")


def load_embedding_images(
    resolved: Iterable[ResolvedTarget],
    *,
    session: requests.Session | None = None,
) -> list[Any]:
    """Load only the resolver-persisted thumbnail URL for each accepted target."""
    images: list[Any] = []
    for item in resolved:
        try:
            images.append(load_wikimedia_image(item.image.thumbnail_url, session=session))
        except ImageInputError as error:
            raise ImageInputError(f"{item.target.title}: {error}") from error
    return images
