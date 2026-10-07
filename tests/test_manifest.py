from golem2.manifest import CANONICAL_TITLES, load_manifest


def test_target_manifest_has_the_exact_unique_fifty_titles() -> None:
    targets = load_manifest()

    assert len(targets) == 50
    assert len({target.id for target in targets}) == 50
    assert len({target.title for target in targets}) == 50
    assert {target.title for target in targets} == set(CANONICAL_TITLES)
    assert {target.category for target in targets} == {
        "People and ideas",
        "History and civilizations",
        "Physical sciences and space",
        "Life and Earth sciences",
        "Computing and systems",
        "Culture and society",
    }
