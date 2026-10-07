import copy
import json

import pytest

from golem2.manifest import Target
from golem2.mediawiki import MediaWikiResolver, ResolutionError

from .support import FIXTURES, FixtureSession


def _target() -> Target:
    return Target(id="ada", title="Ada", category="People and ideas", color="#EAB308")


def test_resolver_follows_redirect_and_persists_full_commons_provenance() -> None:
    resolver = MediaWikiResolver(session=FixtureSession(), retry_delay_seconds=0)

    item = resolver.resolve_targets([_target()])[0]

    assert item.article.title == "Ada Lovelace"
    assert item.article.source_revision_id == 123456
    assert item.image.commons_file_title == "File:Ada_Lovelace_portrait.jpg"
    assert item.image.thumbnail_width == 600
    assert item.image.author == "Example photographer"
    assert item.image.license_url == "https://creativecommons.org/licenses/by-sa/4.0/"


def test_resolver_rejects_missing_page_image() -> None:
    en = json.loads((FIXTURES / "en_article_redirect.json").read_text())
    del en["query"]["pages"][0]["pageimage"]
    del en["query"]["pages"][0]["thumbnail"]
    resolver = MediaWikiResolver(session=FixtureSession(en_payload=en), retry_delay_seconds=0)

    with pytest.raises(ResolutionError, match="no page image"):
        resolver.resolve_targets([_target()])


def test_resolver_rejects_commons_without_meaningful_attribution() -> None:
    commons = json.loads((FIXTURES / "commons_ada.json").read_text())
    metadata = commons["query"]["pages"][0]["imageinfo"][0]["extmetadata"]
    metadata["Artist"]["value"] = "<b>Unknown</b>"
    metadata["Credit"]["value"] = " "
    resolver = MediaWikiResolver(
        session=FixtureSession(commons_payload=copy.deepcopy(commons)),
        retry_delay_seconds=0,
    )

    with pytest.raises(ResolutionError, match="meaningful attribution"):
        resolver.fetch_commons_images(["File:Ada_Lovelace_portrait.jpg"])


def test_audio_resolver_normalizes_legacy_ogg_and_creative_commons_license() -> None:
    commons = json.loads((FIXTURES / "commons_audio.json").read_text())
    info = commons["query"]["pages"][0]["imageinfo"][0]
    info["mime"] = "application/ogg"
    info["extmetadata"]["LicenseUrl"]["value"] = "http://creativecommons.org/licenses/by-sa/3.0/"
    resolver = MediaWikiResolver(session=FixtureSession(commons_payload=commons), retry_delay_seconds=0)

    audio = resolver.fetch_commons_audio(["File:Ada Lovelace.ogg"])["File:Ada Lovelace.ogg"]

    assert audio.mime == "audio/ogg"
    assert audio.license_url == "https://creativecommons.org/licenses/by-sa/3.0/"


def test_audio_resolver_allows_url_less_public_domain_license() -> None:
    commons = json.loads((FIXTURES / "commons_audio.json").read_text())
    metadata = commons["query"]["pages"][0]["imageinfo"][0]["extmetadata"]
    del metadata["LicenseUrl"]
    metadata["LicenseShortName"]["value"] = "Public domain"
    metadata["UsageTerms"]["value"] = "Public domain"
    resolver = MediaWikiResolver(session=FixtureSession(commons_payload=commons), retry_delay_seconds=0)

    audio = resolver.fetch_commons_audio(["File:Ada Lovelace.ogg"])["File:Ada Lovelace.ogg"]

    assert audio.license_url is None
