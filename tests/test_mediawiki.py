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
        resolver.resolve_targets([_target()])
