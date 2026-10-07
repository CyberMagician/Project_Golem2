"""Download and verify the exact EmbeddingGemma 2 revision used by Project Golem 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from golem2.config import DEFAULT_MODEL_DIR, MODEL_SPEC  # noqa: E402


def main() -> int:
    destination = DEFAULT_MODEL_DIR
    api = HfApi()
    info = api.model_info(MODEL_SPEC.model_id, revision=MODEL_SPEC.revision)
    resolved_sha = getattr(info, "sha", None)
    if resolved_sha != MODEL_SPEC.revision:
        raise RuntimeError(
            f"Hugging Face resolved {resolved_sha!r}, not pinned revision {MODEL_SPEC.revision}."
        )
    snapshot_download(
        repo_id=MODEL_SPEC.model_id,
        revision=MODEL_SPEC.revision,
        local_dir=destination,
    )
    if not (destination / "config.json").is_file():
        raise RuntimeError("Snapshot completed without config.json; model cache is incomplete.")
    marker = {
        "model_id": MODEL_SPEC.model_id,
        "revision": MODEL_SPEC.revision,
        "resolved_sha": resolved_sha,
    }
    (destination / ".golem2-model.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Downloaded and verified {MODEL_SPEC.model_id}@{MODEL_SPEC.revision} at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
