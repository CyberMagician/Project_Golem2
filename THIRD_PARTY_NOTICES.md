# Third-party notices

Project Golem 2 is licensed under Apache-2.0. Its source is a clean
implementation informed by the Apache-2.0 licensed Project Golem and retains
that project's required notice in `NOTICE`.

## Runtime components

| Component | Use | License / notice |
| --- | --- | --- |
| Three.js r160 | Browser 3D renderer and `Raycaster` | MIT; loaded from the pinned `unpkg.com` URL in `src/golem2/static/app.js`. See <https://github.com/mrdoob/three.js/blob/r160/LICENSE>. |
| Flask | Local HTTP server | BSD-3-Clause. |
| NumPy | Local vector storage and similarity | BSD-3-Clause. |
| Sentence Transformers | Embedding API | Apache-2.0. |
| PyTorch | Local model inference | BSD-3-Clause. |
| EmbeddingGemma 2 | Local text embedding model | Apache-2.0, Google DeepMind; model card: <https://huggingface.co/google/embeddinggemma-2>. |

## Wikimedia content

Wikipedia article text and Wikimedia Commons media are not bundled with this
repository. A generated corpus stores each selected file's Commons title,
source URL, author/credit, license data, and article provenance. Rendering
keeps that attribution available in the node popover. Rebuilders and deployers
must comply with the individual file licenses and Wikimedia Terms of Use.
