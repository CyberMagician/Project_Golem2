"""The strict, versioned 50-page source manifest."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_MANIFEST_PATH, MANIFEST_SCHEMA_VERSION


CANONICAL_TITLES = (
    "Ada Lovelace",
    "Alan Turing",
    "Albert Einstein",
    "Ancient Egypt",
    "Architecture",
    "Astronomy",
    "Astrophysics",
    "Basketball",
    "Black hole",
    "Botany",
    "Chemistry",
    "Chess",
    "Climate change",
    "Computer science",
    "Cryptography",
    "Cybernetics",
    "DNA",
    "Data science",
    "Earth",
    "Ecology",
    "Evolution",
    "General relativity",
    "Genome",
    "Geology",
    "Immune system",
    "Industrial Revolution",
    "Internet",
    "Isaac Newton",
    "Julius Caesar",
    "Leonardo da Vinci",
    "Literature",
    "Marie Curie",
    "Mathematics",
    "Medicine",
    "Mount Everest",
    "Music theory",
    "Neuron",
    "Ocean",
    "Painting",
    "Philosophy",
    "Photography",
    "Quantum mechanics",
    "Renaissance",
    "Robotics",
    "Roman Empire",
    "Smartphone",
    "Solar System",
    "Thermodynamics",
    "World Wide Web",
    "Cleopatra",
)

_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_COLOR_PATTERN = re.compile(r"^#[0-9A-F]{6}$")


class ManifestError(ValueError):
    """The fixed input set has been changed or malformed."""


@dataclass(frozen=True)
class Target:
    id: str
    title: str
    category: str
    color: str


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> tuple[Target, ...]:
    """Load only the canonical, complete target set."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"Cannot read target manifest at {path}: {error}") from error

    if not isinstance(document, dict) or document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError("Target manifest has an unsupported schema version.")
    entries = document.get("targets")
    if not isinstance(entries, list):
        raise ManifestError("Target manifest must contain a targets list.")

    targets: list[Target] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"Target {index} must be an object.")
        values = {name: entry.get(name) for name in ("id", "title", "category", "color")}
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ManifestError(f"Target {index} has incomplete metadata.")
        if not _ID_PATTERN.fullmatch(values["id"]):
            raise ManifestError(f"Target {index} has an unsafe stable id.")
        if not _COLOR_PATTERN.fullmatch(values["color"]):
            raise ManifestError(f"Target {index} has an invalid color.")
        targets.append(Target(**values))

    ids = [target.id for target in targets]
    titles = [target.title for target in targets]
    if len(targets) != 50:
        raise ManifestError(f"Expected exactly 50 targets, found {len(targets)}.")
    if len(ids) != len(set(ids)) or len(titles) != len(set(titles)):
        raise ManifestError("Target ids and titles must both be unique.")
    if set(titles) != set(CANONICAL_TITLES):
        missing = sorted(set(CANONICAL_TITLES) - set(titles))
        unexpected = sorted(set(titles) - set(CANONICAL_TITLES))
        raise ManifestError(
            f"Target titles differ from the canonical set; missing={missing}, unexpected={unexpected}."
        )
    return tuple(targets)
