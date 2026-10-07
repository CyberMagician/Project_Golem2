from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest


def test_frontend_uses_raycaster_text_content_safe_urls_and_image_failure_state() -> None:
    source = (Path(__file__).parents[1] / "src" / "golem2" / "static" / "app.js").read_text()

    assert "THREE.Raycaster" in source
    assert "textContent" in source
    assert "safeHttpUrl" in source
    assert 'nodeImage.addEventListener("error"' in source
    assert 'nodeAudio.addEventListener("play", startWaveform)' in source
    assert 'nodeAudio.pause();' in source
    assert 'graphCanvas.addEventListener("click"' in source
    assert 'event.key === "Escape"' in source
    assert "audio.autoplay" not in source
    assert "innerHTML" not in source


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser-module smoke testing")
def test_frontend_module_initializes_without_waveform_scope_errors() -> None:
    source = (Path(__file__).parents[1] / "src" / "golem2" / "static" / "app.js").read_text()
    source = source.replace(
        'import * as THREE from "three";\nimport { OrbitControls } from "three/addons/controls/OrbitControls.js";\n',
        "",
    )
    source = source.replace("loadGraph();\nanimate();", "")
    harness = """
const element = () => ({
  style: {}, hidden: false, clientWidth: 300, clientHeight: 80, width: 300, height: 80,
  paused: true, addEventListener() {}, append() {}, setAttribute() {}, removeAttribute() {},
  pause() { this.paused = true; }, getContext() {
    return { clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {} };
  },
});
const elements = new Proxy({}, { get: (items, key) => items[key] ||= element() });
globalThis.document = { getElementById: (id) => elements[id], createElement: () => element() };
globalThis.window = { devicePixelRatio: 1, innerWidth: 1024, innerHeight: 768, addEventListener() {} };
globalThis.cancelAnimationFrame = () => {};
globalThis.requestAnimationFrame = () => 1;
globalThis.THREE = {
  WebGLRenderer: class { setPixelRatio() {} setClearColor() {} setSize() {} render() {} },
  Scene: class { add() {} },
  FogExp2: class {},
  PerspectiveCamera: class { constructor() { this.position = { set() {} }; } updateProjectionMatrix() {} },
  Raycaster: class { constructor() { this.params = { Points: {} }; } setFromCamera() {} intersectObject() { return []; } },
  Vector2: class {},
  Color: class {},
  BufferGeometry: class {},
  Float32BufferAttribute: class {},
  Points: class {},
  PointsMaterial: class {},
  LineSegments: class {},
  LineBasicMaterial: class {},
};
globalThis.OrbitControls = class { constructor() {} };
"""
    assertion = """
if (typeof startWaveform !== "function" || typeof stopWaveform !== "function" || typeof showAudio !== "function") {
  throw new Error("Waveform helpers were not initialized at module scope.");
}
"""
    with tempfile.TemporaryDirectory() as temporary_directory:
        script = Path(temporary_directory) / "frontend-smoke.js"
        script.write_text(harness + source + assertion, encoding="utf-8")
        result = subprocess.run(
            ["node", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0, result.stderr
