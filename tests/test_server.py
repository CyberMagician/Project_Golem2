import numpy as np

from golem2.server import create_app

from .support import build_valid_artifacts


def test_query_endpoint_uses_validated_artifacts_and_reports_clear_input_errors(tmp_path) -> None:
    artifact_dir = build_valid_artifacts(tmp_path / "artifacts")

    def query_encoder(_: str) -> np.ndarray:
        vector = np.zeros(768, dtype=np.float32)
        vector[0] = 1
        return vector

    app = create_app(artifact_dir=artifact_dir, model_dir=tmp_path / "model", query_encoder=query_encoder)
    client = app.test_client()

    graph = client.get("/api/graph")
    assert graph.status_code == 200
    assert graph.get_json()["node_count"] == 50

    response = client.post("/api/query", json={"query": "counting machines", "top_k": 3})
    assert response.status_code == 200
    assert len(response.get_json()["matches"]) == 3
    assert response.get_json()["matches"][0]["id"] == "ada-lovelace"

    invalid = client.post("/api/query", json={"query": "", "top_k": 99})
    assert invalid.status_code == 400
    assert "non-empty" in invalid.get_json()["error"]
