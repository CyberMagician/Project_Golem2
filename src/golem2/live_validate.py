"""Optional network-only verification that every canonical target has image data."""

from __future__ import annotations

from .manifest import load_manifest
from .mediawiki import MediaWikiResolver


def main() -> int:
    targets = load_manifest()
    resolved = MediaWikiResolver().resolve_targets(targets)
    if len(resolved) != 50:
        raise RuntimeError(f"Expected 50 resolved pages, got {len(resolved)}.")
    print("Validated image provenance for all 50 canonical Wikipedia targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
