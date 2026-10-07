# Project Golem 2

Project Golem 2 is a local Flask and Three.js semantic visualization of a
fixed, attributable 50-page Wikipedia corpus. It replaces the dynamic,
unattributed legacy ingestion flow with reproducible sources, a pinned
EmbeddingGemma 2 revision, validated `50 x 768` normalized vectors, and a
safe provenance popover for every graph node.

The implementation is a clean successor to Apache-2.0 Project Golem; the
required upstream notice is retained in [`NOTICE`](NOTICE).

## What it does

1. Resolves exactly the 50 titles in [`data/targets.json`](data/targets.json)
   using the English Wikipedia and Wikimedia Commons **Action APIs**. It does
   not scrape HTML.
2. Rejects an entire build when any target lacks a summary, page image,
   Commons metadata, meaningful author/credit, license data, or a safe source
   URL.
3. Embeds each titled summary and its vetted, persisted Wikimedia Commons
   thumbnail together with
   [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
   at immutable revision
   `914f7f89142e33e77833254d9c9b90c3cef7303b`.
4. Writes ignored local artifacts with exactly 50 nodes, a finite and
   L2-normalized `(50, 768)` matrix, deterministic PCA positions, stable
   neighbor topology, checksums, model ID, revision, dimensions, and prompt
   metadata.
5. Serves an interactive Three.js graph. Hovering a point uses
   `THREE.Raycaster`; focused graph controls support arrow-key inspection.
   The accessible popover includes the persisted image, article and Commons
   links, author/credit, and license terms.

## Requirements

- Python 3.11 or later.
- Internet access for the model download and corpus build. Normal tests never
  contact the network.
- Roughly 4 GB free disk space and several GB of memory for the 740M
  EmbeddingGemma 2 snapshot. CPU inference is supported but slower.
- A modern browser with WebGL.

EmbeddingGemma 2 uses a shared native **768-dimensional** vector space. Corpus
items are genuine text-plus-image inputs (`title: … | text: … <|image|>`)
using a resolver-verified Commons thumbnail; search remains text-only with
the model card's `prompt_name="SearchQuery"` task prompt in that same shared
space. This project never configures `float16`: it uses `bfloat16` only when
CUDA reports native support and otherwise uses `float32`. It does not use
`trust_remote_code`.

### Optional curated audio

[`data/audio_targets.json`](data/audio_targets.json) deliberately enriches
only three nodes—Ada Lovelace, Astronomy, and Music theory—with directly
licensed Wikimedia Commons audio. It is not an assertion that every topic has
audio. During corpus construction the selected Commons file metadata is
resolved through the Action API and must include an audio MIME type, secure
source/file/license URLs, meaningful author or credit, and license terms.
Accepted clips are bounded to 32 MiB, checksummed, cached under ignored
`data/audio/`, and used as `<|audio|>` inputs alongside image and titled text.
The model keeps its audio encoder enabled; all other nodes remain text plus
image embeddings in the same native 768-D space.

## Installation

Run commands from the repository root.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`pyproject.toml` applies the same compatible upper bounds for editable
development installs:

```bash
python -m pip install -e ".[dev]"
```

## Download the pinned model

The model cache is intentionally repository-local and ignored by Git. The
downloader verifies that Hugging Face resolves the requested immutable commit
before it writes its local completion marker:

```bash
python scripts/download_model.py
```

This downloads `google/embeddinggemma-2` to
`models/embeddinggemma-2` and verifies revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`. No model weights, Hugging Face
cache files, or generated artifacts should be committed. The command uses
public access; do not put an access token in the repository or shell history.

## Validate and build the corpus

Run the optional network-only validation before downloading or embedding to
confirm that all 50 canonical pages currently provide acceptable image
provenance:

```bash
python -m golem2.live_validate
```

Build the local corpus after downloading the model:

```bash
python -m golem2.ingest
```

The resolver uses a descriptive `User-Agent`, bounded batches of at most 50,
`maxlag=5`, explicit connection/read timeouts, in-process request caching, and
retries only transient network faults, `maxlag`, and retryable HTTP status
codes. It requests 600px Commons thumbnails. If a page image is WebM or
another non-image source, its rendered thumbnail must be confirmed as an
image; otherwise the resolver searches a bounded list of article images for a
still-image fallback. A build stops rather than silently emitting incomplete
nodes. Corpus construction then downloads only those persisted,
HTTPS-Wikimedia thumbnail URLs, verifies response MIME type, byte/pixel
bounds, and image decoding before passing RGB images alongside the titled
text to EmbeddingGemma 2. It applies the analogous required provenance and
cache checks to the three curated audio files. Image and audio bytes are not
committed.

Generated files are written under `data/artifacts/`:

| File | Purpose |
| --- | --- |
| `corpus.json` | The 50 graph nodes, summaries, deterministic coordinates/topology, Wikipedia revision details, and full image provenance. |
| `vectors.npy` | The finite, pre-normalized `(50, 768)` float matrix. |
| `metadata.json` | Schema, fixed model identity/revision/prompts/dimension, count, and SHA-256 checksums. |

The server validates all of these on every graph or query request. Any
different model revision, dimension, corrupted checksum, non-finite value, or
non-normalized vector is rejected.

## Run and query

```bash
python -m golem2.server
```

Open <http://127.0.0.1:8000>. Drag to orbit, scroll to zoom, hover nodes to
inspect provenance, and use the query form to highlight local semantic
matches. Images are only requested by the browser from their already-persisted
URLs; the UI never discovers images at hover time. Where a selected node has
curated audio, its hover popover includes an accessible native player and
on-play Web Audio waveform. It never autoplays; the player receives audio
from a checksum-validated local cache endpoint.

Example local API request:

```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"theory of computation\",\"top_k\":8}"
```

PowerShell equivalent:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/query `
  -ContentType "application/json" `
  -Body '{"query":"theory of computation","top_k":8}'
```

## Tests

```bash
python -m pytest
```

Tests use checked-in, recorded-style Action API fixtures and never make live
requests. They cover exact title cardinality and uniqueness, redirects,
missing image/attribution rejection, text/image/audio model inputs,
model dtype/query prompt behavior, curated audio cache checksum validation,
corpus and vector compatibility, server query validation, and the frontend's
Raycaster/text-content/URL/image-error/no-autoplay contract. The optional
`python -m golem2.live_validate` command is the explicit live test.

## Attribution and licenses

- The source code is Apache-2.0; see [`LICENSE`](LICENSE) and
  [`NOTICE`](NOTICE).
- EmbeddingGemma 2 is Apache-2.0 and is made by Google DeepMind. See its
  [model card](https://huggingface.co/google/embeddinggemma-2) for model terms
  and documentation.
- Wikipedia text and Wikimedia Commons files are fetched at build time, not
  shipped in this repository. Each generated node retains its article URL,
  source revision where provided, Commons file title and URL, source and
  thumbnail URLs, author/credit, license name/URL/terms, attribution flag,
  MIME type, and dimensions. Wikimedia content may carry additional
  obligations; use the per-node popover and Commons source page when
  redistributing it.
- The curated audio layer resolves and exposes equivalent Commons file,
  author/credit, license, source, MIME, checksum, and byte-count metadata for
  its three optional clips. Review each Commons source page before
  redistributing its media.
- Three.js r160 is fetched from a pinned CDN URL and is MIT-licensed. See
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for runtime notices.
