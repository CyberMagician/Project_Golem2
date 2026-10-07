from pathlib import Path


def test_frontend_uses_raycaster_text_content_safe_urls_and_image_failure_state() -> None:
    source = (Path(__file__).parents[1] / "src" / "golem2" / "static" / "app.js").read_text()

    assert "THREE.Raycaster" in source
    assert "textContent" in source
    assert "safeHttpUrl" in source
    assert 'nodeImage.addEventListener("error"' in source
    assert 'nodeAudio.addEventListener("play", startWaveform)' in source
    assert 'nodeAudio.pause();' in source
    assert "audio.autoplay" not in source
    assert "innerHTML" not in source
