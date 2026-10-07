"""Bounded MediaWiki API resolution with attribution-safe image provenance."""

from __future__ import annotations

import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Iterable
from urllib.parse import urlparse

import requests

from .config import USER_AGENT
from .manifest import Target


EN_WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
REQUEST_TIMEOUT = (5, 30)
MAX_BATCH_SIZE = 50
MAX_FALLBACK_IMAGES = 20
THUMBNAIL_WIDTH = 600
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class MediaWikiError(RuntimeError):
    """A MediaWiki request failed or returned an unexpected response."""


class ResolutionError(MediaWikiError):
    """A target cannot satisfy the article/image attribution contract."""


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return " ".join("".join(self.parts).split())


def plain_text(value: Any) -> str:
    """Discard tags from untrusted API metadata before it reaches a browser."""
    if not isinstance(value, str):
        return ""
    parser = _PlainTextParser()
    parser.feed(value)
    parser.close()
    return parser.text()


def safe_https_url(value: Any) -> str | None:
    """Validate an HTTPS URL before persisting it as a browser navigation target."""
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        return None
    return value


def trusted_wikimedia_url(value: Any) -> str | None:
    """Only persist HTTPS article, Commons, and upload URLs returned by Wikimedia."""
    url = safe_https_url(value)
    if not url:
        return None
    host = (urlparse(url).hostname or "").lower()
    if not (
        host == "wikimedia.org"
        or host.endswith(".wikimedia.org")
        or host == "wikipedia.org"
        or host.endswith(".wikipedia.org")
    ):
        return None
    return url


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    entry = metadata.get(key)
    if isinstance(entry, dict):
        return plain_text(entry.get("value"))
    return plain_text(entry)


def _is_meaningful(value: str) -> bool:
    return bool(value and value.casefold() not in {"unknown", "unknown author", "n/a", "none"})


@dataclass(frozen=True)
class ArticlePage:
    requested_title: str
    title: str
    summary: str
    article_url: str
    source_revision_id: int | None
    source_revision_timestamp: str | None
    page_image_title: str


@dataclass(frozen=True)
class CommonsImage:
    title: str
    file_url: str
    source_url: str
    thumbnail_url: str
    mime: str
    width: int | None
    height: int | None
    thumbnail_width: int | None
    thumbnail_height: int | None
    author: str
    credit: str
    license_name: str
    license_url: str
    license_terms: str
    attribution_required: bool

    @property
    def is_static_image(self) -> bool:
        return self.mime.startswith("image/")


@dataclass(frozen=True)
class CommonsAudio:
    title: str
    file_url: str
    source_url: str
    mime: str
    size: int | None
    author: str
    credit: str
    license_name: str
    license_url: str
    license_terms: str
    attribution_required: bool


@dataclass(frozen=True)
class ImageProvenance:
    thumbnail_url: str
    source_url: str
    commons_file_url: str
    commons_file_title: str
    author: str
    credit: str
    license_name: str
    license_url: str
    license_terms: str
    attribution_required: bool
    mime: str
    width: int | None
    height: int | None
    thumbnail_width: int | None
    thumbnail_height: int | None
    requested_thumbnail_width: int
    page_image_title: str

    def to_dict(self) -> dict[str, object]:
        return {
            "thumbnail_url": self.thumbnail_url,
            "source_url": self.source_url,
            "commons_file_url": self.commons_file_url,
            "commons_file_title": self.commons_file_title,
            "author": self.author,
            "credit": self.credit,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "license_terms": self.license_terms,
            "attribution_required": self.attribution_required,
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "thumbnail_width": self.thumbnail_width,
            "thumbnail_height": self.thumbnail_height,
            "requested_thumbnail_width": self.requested_thumbnail_width,
            "page_image_title": self.page_image_title,
        }


@dataclass(frozen=True)
class ResolvedTarget:
    target: Target
    article: ArticlePage
    image: ImageProvenance


class MediaWikiResolver:
    """Resolve a fixed target set without HTML scraping or unbounded requests."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        user_agent: str = USER_AGENT,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        self.session = session or requests.Session()
        self.user_agent = user_agent
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._article_cache: dict[str, ArticlePage] = {}
        self._commons_cache: dict[str, CommonsImage] = {}
        self._thumbnail_cache: dict[str, bool] = {}

    def _request_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        request_params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    endpoint,
                    params=request_params,
                    headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.RequestException as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds * (attempt + 1))
                    continue
                break

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                time.sleep(self.retry_delay_seconds * (attempt + 1))
                continue
            try:
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as error:
                last_error = error
                break
            if not isinstance(payload, dict):
                raise MediaWikiError("MediaWiki returned a non-object JSON response.")
            api_error = payload.get("error")
            if isinstance(api_error, dict):
                code = api_error.get("code")
                if code == "maxlag" and attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds * (attempt + 1))
                    continue
                raise MediaWikiError(f"MediaWiki API error: {code or 'unknown'}")
            return payload

        raise MediaWikiError(f"MediaWiki request failed after retries: {last_error}") from last_error

    def fetch_articles(self, titles: Iterable[str]) -> dict[str, ArticlePage]:
        requested = list(dict.fromkeys(titles))
        missing = [title for title in requested if title not in self._article_cache]
        for batch in _chunks(missing, MAX_BATCH_SIZE):
            payload = self._request_json(
                EN_WIKIPEDIA_API,
                {
                    "action": "query",
                    "redirects": "1",
                    "prop": "pageimages|info|revisions|extracts",
                    "piprop": "thumbnail|original",
                    "pithumbsize": str(THUMBNAIL_WIDTH),
                    "inprop": "url",
                    "rvprop": "ids|timestamp",
                    "rvlimit": "1",
                    "exintro": "1",
                    "explaintext": "1",
                    "titles": "|".join(batch),
                },
            )
            query = payload.get("query")
            if not isinstance(query, dict):
                raise MediaWikiError("Wikipedia response has no query object.")
            pages = query.get("pages")
            if not isinstance(pages, list):
                raise MediaWikiError("Wikipedia response has no pages list.")
            pages_by_title = {
                page.get("title"): page
                for page in pages
                if isinstance(page, dict) and isinstance(page.get("title"), str)
            }
            redirects = {
                item.get("from"): item.get("to")
                for item in query.get("redirects", [])
                if isinstance(item, dict)
                and isinstance(item.get("from"), str)
                and isinstance(item.get("to"), str)
            }
            normalized = {
                item.get("from"): item.get("to")
                for item in query.get("normalized", [])
                if isinstance(item, dict)
                and isinstance(item.get("from"), str)
                and isinstance(item.get("to"), str)
            }
            for requested_title in batch:
                lookup_title = redirects.get(normalized.get(requested_title, requested_title), requested_title)
                page = pages_by_title.get(lookup_title)
                if not isinstance(page, dict) or page.get("missing") is not None:
                    raise ResolutionError(f"Wikipedia article is missing: {requested_title}")
                image_name = page.get("pageimage")
                summary = plain_text(page.get("extract"))
                article_url = trusted_wikimedia_url(page.get("fullurl"))
                if not isinstance(image_name, str) or not image_name:
                    raise ResolutionError(f"Wikipedia article has no page image: {requested_title}")
                if not summary:
                    raise ResolutionError(f"Wikipedia article has no usable summary: {requested_title}")
                if not article_url:
                    raise ResolutionError(f"Wikipedia article URL is invalid: {requested_title}")
                revision = page.get("revisions")
                first_revision = revision[0] if isinstance(revision, list) and revision else {}
                if not isinstance(first_revision, dict):
                    first_revision = {}
                self._article_cache[requested_title] = ArticlePage(
                    requested_title=requested_title,
                    title=page["title"],
                    summary=summary,
                    article_url=article_url,
                    source_revision_id=(
                        first_revision.get("revid") if isinstance(first_revision.get("revid"), int) else None
                    ),
                    source_revision_timestamp=(
                        first_revision.get("timestamp")
                        if isinstance(first_revision.get("timestamp"), str)
                        else None
                    ),
                    page_image_title=f"File:{image_name.removeprefix('File:')}",
                )
        return {title: self._article_cache[title] for title in requested}

    def fetch_commons_images(self, file_titles: Iterable[str]) -> dict[str, CommonsImage]:
        requested = list(dict.fromkeys(file_titles))
        missing = [title for title in requested if title not in self._commons_cache]
        for batch in _chunks(missing, MAX_BATCH_SIZE):
            payload = self._request_json(
                COMMONS_API,
                {
                    "action": "query",
                    "prop": "imageinfo|info",
                    "iiprop": "url|mime|size|extmetadata",
                    "iiurlwidth": str(THUMBNAIL_WIDTH),
                    "inprop": "url",
                    "titles": "|".join(batch),
                },
            )
            query = payload.get("query")
            if not isinstance(query, dict) or not isinstance(query.get("pages"), list):
                raise MediaWikiError("Commons response has no pages list.")
            pages_by_title = {
                page.get("title"): page
                for page in query["pages"]
                if isinstance(page, dict) and isinstance(page.get("title"), str)
            }
            normalized = {
                item.get("from"): item.get("to")
                for item in query.get("normalized", [])
                if isinstance(item, dict)
                and isinstance(item.get("from"), str)
                and isinstance(item.get("to"), str)
            }
            for requested_title in batch:
                page = pages_by_title.get(normalized.get(requested_title, requested_title))
                imageinfo = page.get("imageinfo") if isinstance(page, dict) else None
                info = imageinfo[0] if isinstance(imageinfo, list) and imageinfo else None
                if not isinstance(info, dict) or not isinstance(page, dict) or page.get("missing") is not None:
                    raise ResolutionError(f"Commons metadata is missing: {requested_title}")
                metadata = info.get("extmetadata")
                if not isinstance(metadata, dict):
                    raise ResolutionError(f"Commons attribution metadata is missing: {requested_title}")
                file_url = trusted_wikimedia_url(page.get("fullurl"))
                source_url = trusted_wikimedia_url(info.get("url"))
                thumbnail_url = trusted_wikimedia_url(info.get("thumburl"))
                mime = info.get("mime")
                license_url = safe_https_url(_metadata_value(metadata, "LicenseUrl"))
                if not all((file_url, source_url, thumbnail_url, license_url, isinstance(mime, str))):
                    raise ResolutionError(f"Commons image URLs or MIME are missing: {requested_title}")
                author = _metadata_value(metadata, "Artist") or _metadata_value(metadata, "Author")
                credit = _metadata_value(metadata, "Credit")
                license_name = _metadata_value(metadata, "LicenseShortName")
                license_terms = _metadata_value(metadata, "UsageTerms")
                if not (_is_meaningful(author) or _is_meaningful(credit)):
                    raise ResolutionError(f"Commons image lacks meaningful attribution: {requested_title}")
                if not _is_meaningful(license_name) or not _is_meaningful(license_terms):
                    raise ResolutionError(f"Commons image lacks license terms: {requested_title}")
                self._commons_cache[requested_title] = CommonsImage(
                    title=page["title"],
                    file_url=file_url,
                    source_url=source_url,
                    thumbnail_url=thumbnail_url,
                    mime=mime.casefold(),
                    width=info.get("width") if isinstance(info.get("width"), int) else None,
                    height=info.get("height") if isinstance(info.get("height"), int) else None,
                    thumbnail_width=(
                        info.get("thumbwidth") if isinstance(info.get("thumbwidth"), int) else None
                    ),
                    thumbnail_height=(
                        info.get("thumbheight") if isinstance(info.get("thumbheight"), int) else None
                    ),
                    author=author,
                    credit=credit,
                    license_name=license_name,
                    license_url=license_url,
                    license_terms=license_terms,
                    attribution_required=(
                        _metadata_value(metadata, "AttributionRequired").casefold() in {"true", "yes"}
                    ),
                )
        return {title: self._commons_cache[title] for title in requested}

    def fetch_commons_audio(self, file_titles: Iterable[str]) -> dict[str, CommonsAudio]:
        """Resolve selected Commons audio with the same attribution requirements as images."""
        requested = list(dict.fromkeys(file_titles))
        result: dict[str, CommonsAudio] = {}
        for batch in _chunks(requested, MAX_BATCH_SIZE):
            payload = self._request_json(
                COMMONS_API,
                {
                    "action": "query",
                    "prop": "imageinfo|info",
                    "iiprop": "url|mime|size|extmetadata",
                    "inprop": "url",
                    "titles": "|".join(batch),
                },
            )
            query = payload.get("query")
            if not isinstance(query, dict) or not isinstance(query.get("pages"), list):
                raise MediaWikiError("Commons audio response has no pages list.")
            pages_by_title = {
                page.get("title"): page
                for page in query["pages"]
                if isinstance(page, dict) and isinstance(page.get("title"), str)
            }
            normalized = {
                item.get("from"): item.get("to")
                for item in query.get("normalized", [])
                if isinstance(item, dict)
                and isinstance(item.get("from"), str)
                and isinstance(item.get("to"), str)
            }
            for requested_title in batch:
                page = pages_by_title.get(normalized.get(requested_title, requested_title))
                imageinfo = page.get("imageinfo") if isinstance(page, dict) else None
                info = imageinfo[0] if isinstance(imageinfo, list) and imageinfo else None
                if not isinstance(info, dict) or not isinstance(page, dict) or page.get("missing") is not None:
                    raise ResolutionError(f"Commons audio metadata is missing: {requested_title}")
                metadata = info.get("extmetadata")
                if not isinstance(metadata, dict):
                    raise ResolutionError(f"Commons audio attribution metadata is missing: {requested_title}")
                file_url = trusted_wikimedia_url(page.get("fullurl"))
                source_url = trusted_wikimedia_url(info.get("url"))
                mime = info.get("mime")
                license_url = safe_https_url(_metadata_value(metadata, "LicenseUrl"))
                if not all((file_url, source_url, license_url, isinstance(mime, str))):
                    raise ResolutionError(f"Commons audio URLs or MIME are missing: {requested_title}")
                if not mime.casefold().startswith("audio/"):
                    raise ResolutionError(f"Commons source is not an audio file: {requested_title}")
                author = _metadata_value(metadata, "Artist") or _metadata_value(metadata, "Author")
                credit = _metadata_value(metadata, "Credit")
                license_name = _metadata_value(metadata, "LicenseShortName")
                license_terms = _metadata_value(metadata, "UsageTerms")
                if not (_is_meaningful(author) or _is_meaningful(credit)):
                    raise ResolutionError(f"Commons audio lacks meaningful attribution: {requested_title}")
                if not _is_meaningful(license_name) or not _is_meaningful(license_terms):
                    raise ResolutionError(f"Commons audio lacks license terms: {requested_title}")
                result[requested_title] = CommonsAudio(
                    title=page["title"],
                    file_url=file_url,
                    source_url=source_url,
                    mime=mime.casefold(),
                    size=info.get("size") if isinstance(info.get("size"), int) else None,
                    author=author,
                    credit=credit,
                    license_name=license_name,
                    license_url=license_url,
                    license_terms=license_terms,
                    attribution_required=(
                        _metadata_value(metadata, "AttributionRequired").casefold() in {"true", "yes"}
                    ),
                )
        return {title: result[title] for title in requested}

    def _thumbnail_is_rendered_image(self, thumbnail_url: str) -> bool:
        if thumbnail_url in self._thumbnail_cache:
            return self._thumbnail_cache[thumbnail_url]
        valid = False
        try:
            response = self.session.get(
                thumbnail_url,
                headers={"User-Agent": self.user_agent, "Accept": "image/*"},
                timeout=REQUEST_TIMEOUT,
                stream=True,
            )
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].casefold()
            valid = response.status_code == 200 and content_type.startswith("image/")
            close = getattr(response, "close", None)
            if callable(close):
                close()
        except requests.RequestException:
            valid = False
        self._thumbnail_cache[thumbnail_url] = valid
        return valid

    def _find_still_fallback(self, article: ArticlePage) -> CommonsImage:
        payload = self._request_json(
            EN_WIKIPEDIA_API,
            {
                "action": "query",
                "prop": "images",
                "imlimit": str(MAX_FALLBACK_IMAGES),
                "titles": article.title,
            },
        )
        query = payload.get("query")
        pages = query.get("pages") if isinstance(query, dict) else None
        page = pages[0] if isinstance(pages, list) and pages else None
        images = page.get("images") if isinstance(page, dict) else None
        file_titles = [
            item["title"]
            for item in images or []
            if isinstance(item, dict) and isinstance(item.get("title"), str)
        ][:MAX_FALLBACK_IMAGES]
        if not file_titles:
            raise ResolutionError(f"No still-image fallback exists for {article.requested_title}")
        candidates = self.fetch_commons_images(file_titles)
        for file_title in file_titles:
            candidate = candidates[file_title]
            if candidate.is_static_image:
                return candidate
        raise ResolutionError(f"No usable static-image fallback exists for {article.requested_title}")

    @staticmethod
    def _provenance(image: CommonsImage, page_image_title: str) -> ImageProvenance:
        return ImageProvenance(
            thumbnail_url=image.thumbnail_url,
            source_url=image.source_url,
            commons_file_url=image.file_url,
            commons_file_title=image.title,
            author=image.author,
            credit=image.credit,
            license_name=image.license_name,
            license_url=image.license_url,
            license_terms=image.license_terms,
            attribution_required=image.attribution_required,
            mime=image.mime,
            width=image.width,
            height=image.height,
            thumbnail_width=image.thumbnail_width,
            thumbnail_height=image.thumbnail_height,
            requested_thumbnail_width=THUMBNAIL_WIDTH,
            page_image_title=page_image_title,
        )

    def resolve_targets(self, targets: Iterable[Target]) -> tuple[ResolvedTarget, ...]:
        """Resolve every target, rejecting the complete build on any bad node."""
        targets_tuple = tuple(targets)
        articles = self.fetch_articles(target.title for target in targets_tuple)
        primary_files = [articles[target.title].page_image_title for target in targets_tuple]
        primary_images = self.fetch_commons_images(primary_files)
        resolved: list[ResolvedTarget] = []
        for target in targets_tuple:
            article = articles[target.title]
            image = primary_images[article.page_image_title]
            if not image.is_static_image and not self._thumbnail_is_rendered_image(image.thumbnail_url):
                image = self._find_still_fallback(article)
            if not image.is_static_image and not self._thumbnail_is_rendered_image(image.thumbnail_url):
                raise ResolutionError(
                    f"Page image is non-static and its rendered thumbnail is unusable: {target.title}"
                )
            resolved.append(
                ResolvedTarget(
                    target=target,
                    article=article,
                    image=self._provenance(image, article.page_image_title),
                )
            )
        return tuple(resolved)
