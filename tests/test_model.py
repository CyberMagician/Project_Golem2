import numpy as np
from pathlib import Path
from types import SimpleNamespace

from golem2.config import MODEL_SPEC
from golem2.model import EmbeddingGemma2Encoder, select_torch_dtype


class _Cuda:
    def __init__(self, available: bool, supports_bf16: bool):
        self.available = available
        self.supports_bf16 = supports_bf16

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.supports_bf16


class _Torch:
    bfloat16 = "bf16"
    float32 = "fp32"

    def __init__(self, available: bool, supports_bf16: bool):
        self.cuda = _Cuda(available, supports_bf16)


class CapturingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append({"sentences": sentences, **kwargs})
        return np.ones((len(sentences), 768), dtype=np.float32)


def test_dtype_never_selects_float16() -> None:
    assert select_torch_dtype(_Torch(True, True)) == "bf16"
    assert select_torch_dtype(_Torch(True, False)) == "fp32"
    assert select_torch_dtype(_Torch(False, False)) == "fp32"


def test_embedding_adapter_uses_multimodal_titled_documents_and_search_prompts() -> None:
    model = CapturingModel()
    encoder = EmbeddingGemma2Encoder.__new__(EmbeddingGemma2Encoder)
    encoder.model = model
    encoder.spec = MODEL_SPEC

    documents = encoder.embed_multimodal_documents(
        ["title: Ada Lovelace | text: mathematician"],
        [object()],
        ["C:\\audio\\ada.ogg"],
    )
    query = encoder.embed_query("mathematician")

    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0)
    assert np.isclose(np.linalg.norm(query), 1.0)
    multimodal_input = model.calls[0]["sentences"][0]
    assert multimodal_input["text"] == (
        "title: Ada Lovelace | text: mathematician <|image|> <|audio|>"
    )
    assert multimodal_input["image"] is not None
    assert multimodal_input["audio"] == "C:\\audio\\ada.ogg"
    assert model.calls[1]["prompt_name"] == "SearchQuery"
    assert model.calls[0]["normalize_embeddings"] is True
    assert model.calls[1]["normalize_embeddings"] is True


def test_model_loader_leaves_audio_encoder_enabled(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr("golem2.model.verify_local_model", lambda *_: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        _Torch(False, False),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    EmbeddingGemma2Encoder(Path("model"))

    assert captured["kwargs"]["model_kwargs"]["torch_dtype"] == "fp32"
    assert "config_kwargs" not in captured["kwargs"]
