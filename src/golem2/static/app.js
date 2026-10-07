import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const graphCanvas = document.getElementById("graph-canvas");
const graphRoot = document.getElementById("graph-root");
const status = document.getElementById("status");
const queryForm = document.getElementById("query-form");
const queryInput = document.getElementById("query-input");
const legend = document.getElementById("legend");
const popover = document.getElementById("node-popover");
const nodeImage = document.getElementById("node-image");
const imageFallback = document.getElementById("image-fallback");
const nodeCategory = document.getElementById("node-category");
const nodeTitle = document.getElementById("node-title");
const nodeSummary = document.getElementById("node-summary");
const nodeCredit = document.getElementById("node-credit");
const articleLink = document.getElementById("article-link");
const commonsLink = document.getElementById("commons-link");
const licenseLink = document.getElementById("license-link");
const audioPanel = document.getElementById("audio-panel");
const nodeAudio = document.getElementById("node-audio");
const audioWaveform = document.getElementById("audio-waveform");
const audioCredit = document.getElementById("audio-credit");
const audioCommonsLink = document.getElementById("audio-commons-link");

const renderer = new THREE.WebGLRenderer({ canvas: graphCanvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x050914, 1);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050914, 0.012);
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
camera.position.set(0, 3, 22);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.35;

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.65;
const pointer = new THREE.Vector2();
let nodes = [];
let points = null;
let pointColors = null;
let baselineColors = null;
let selectedIndex = -1;
let audioContext = null;
let audioAnalyser = null;
let audioSource = null;
let waveformAnimation = null;

function safeHttpUrl(value) {
  if (typeof value !== "string") {
    return null;
  }
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : null;
  } catch {
    return null;
  }
}

function setLink(link, value, text) {
  const safeUrl = safeHttpUrl(value);
  if (!safeUrl) {
    link.hidden = true;
    link.removeAttribute("href");
    return;
  }
  link.href = safeUrl;
  link.textContent = text;
  link.hidden = false;
}

function setPopoverPosition(clientX, clientY) {
  const width = 336;
  const height = 420;
  popover.style.left = `${Math.max(12, Math.min(clientX + 16, window.innerWidth - width - 12))}px`;
  popover.style.top = `${Math.max(12, Math.min(clientY + 16, window.innerHeight - height - 12))}px`;
}

function hidePopover() {
  selectedIndex = -1;
  popover.hidden = true;
  popover.setAttribute("aria-hidden", "true");
  nodeAudio.pause();
}

function showImage(node) {
  const thumbnail = safeHttpUrl(node.image.thumbnail_url);
  nodeImage.removeAttribute("src");
  nodeImage.hidden = true;
  imageFallback.hidden = false;
  imageFallback.textContent = "Image unavailable.";
  if (thumbnail) {
    nodeImage.alt = `Thumbnail for ${node.title}`;
    nodeImage.src = thumbnail;
    nodeImage.hidden = false;
    imageFallback.hidden = true;
  }

  function stopWaveform() {
    if (waveformAnimation !== null) {
      cancelAnimationFrame(waveformAnimation);
      waveformAnimation = null;
    }
  }

  function drawWaveform() {
    if (!audioAnalyser || nodeAudio.paused) {
      stopWaveform();
      return;
    }
    const context = audioWaveform.getContext("2d");
    const width = audioWaveform.width;
    const height = audioWaveform.height;
    const samples = new Uint8Array(audioAnalyser.fftSize);
    audioAnalyser.getByteTimeDomainData(samples);
    context.clearRect(0, 0, width, height);
    context.strokeStyle = "#7dd3fc";
    context.lineWidth = 2;
    context.beginPath();
    samples.forEach((sample, index) => {
      const x = (index / (samples.length - 1)) * width;
      const y = (sample / 255) * height;
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
    waveformAnimation = requestAnimationFrame(drawWaveform);
  }

  async function startWaveform() {
    try {
      if (!audioContext) {
        audioContext = new AudioContext();
        audioSource = audioContext.createMediaElementSource(nodeAudio);
        audioAnalyser = audioContext.createAnalyser();
        audioAnalyser.fftSize = 1024;
        audioSource.connect(audioAnalyser);
        audioAnalyser.connect(audioContext.destination);
      }
      await audioContext.resume();
      stopWaveform();
      drawWaveform();
    } catch {
      stopWaveform();
    }
  }

  function showAudio(node) {
    const audio = node.audio;
    nodeAudio.pause();
    nodeAudio.removeAttribute("src");
    stopWaveform();
    if (!audio) {
      audioPanel.hidden = true;
      return;
    }
    audioPanel.hidden = false;
    nodeAudio.src = `/api/audio/${encodeURIComponent(node.id)}`;
    audioCredit.textContent = `${audio.author || audio.credit} — ${audio.license_name}: ${audio.license_terms}`;
    setLink(audioCommonsLink, audio.commons_file_url, "Commons audio source");
  }
}

function showPopover(index, clientX, clientY) {
  const node = nodes[index];
  if (!node) {
    hidePopover();
    return;
  }
  selectedIndex = index;
  nodeCategory.textContent = node.category;
  nodeTitle.textContent = node.title;
  nodeSummary.textContent = node.summary;
  nodeCredit.textContent = node.image.author || node.image.credit;
  setLink(articleLink, node.article_url, "Wikipedia article");
  setLink(commonsLink, node.image.commons_file_url, "Commons source");
  setLink(
    licenseLink,
    node.image.license_url,
    `${node.image.license_name}: ${node.image.license_terms}`,
  );
  showImage(node);
  showAudio(node);
  setPopoverPosition(clientX, clientY);
  popover.hidden = false;
  popover.setAttribute("aria-hidden", "false");
}

nodeImage.addEventListener("error", () => {
  nodeImage.hidden = true;
  nodeImage.removeAttribute("src");
  imageFallback.hidden = false;
  imageFallback.textContent = "Image could not be loaded.";
});

nodeAudio.addEventListener("play", startWaveform);
nodeAudio.addEventListener("pause", stopWaveform);
nodeAudio.addEventListener("ended", stopWaveform);
nodeAudio.addEventListener("error", () => {
  stopWaveform();
  audioCredit.textContent = "Audio could not be loaded.";
});

function validNode(node) {
  return (
    node
    && typeof node.id === "string"
    && typeof node.title === "string"
    && Array.isArray(node.position)
    && node.position.length === 3
    && node.position.every(Number.isFinite)
    && typeof node.color === "string"
    && node.image
  );
}

function buildLegend() {
  const categories = new Map();
  for (const node of nodes) {
    categories.set(node.category, node.color);
  }
  for (const [category, color] of [...categories.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    const row = document.createElement("div");
    row.className = "legend-row";
    const dot = document.createElement("span");
    dot.className = "legend-dot";
    dot.style.backgroundColor = color;
    const label = document.createElement("span");
    label.textContent = category;
    row.append(dot, label);
    legend.append(row);
  }
}

function buildGraph() {
  const positions = [];
  const colors = [];
  const indexById = new Map();
  nodes.forEach((node, index) => {
    indexById.set(node.id, index);
    positions.push(node.position[0] * 3.2, node.position[1] * 3.2, node.position[2] * 3.2);
    const color = new THREE.Color(node.color);
    colors.push(color.r, color.g, color.b);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  pointColors = geometry.getAttribute("color");
  baselineColors = Float32Array.from(pointColors.array);
  points = new THREE.Points(
    geometry,
    new THREE.PointsMaterial({
      size: 0.38,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.94,
      vertexColors: true,
    }),
  );
  scene.add(points);

  const linePositions = [];
  const emitted = new Set();
  nodes.forEach((node, index) => {
    for (const neighborId of node.neighbors || []) {
      const neighborIndex = indexById.get(neighborId);
      const edgeKey = [index, neighborIndex].sort((a, b) => a - b).join(":");
      if (neighborIndex === undefined || emitted.has(edgeKey)) {
        continue;
      }
      emitted.add(edgeKey);
      const from = node.position;
      const to = nodes[neighborIndex].position;
      linePositions.push(
        from[0] * 3.2, from[1] * 3.2, from[2] * 3.2,
        to[0] * 3.2, to[1] * 3.2, to[2] * 3.2,
      );
    }
  });
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
  scene.add(
    new THREE.LineSegments(
      lineGeometry,
      new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.28 }),
    ),
  );
  buildLegend();
}

function highlight(matches) {
  if (!pointColors || !baselineColors) {
    return;
  }
  pointColors.array.set(baselineColors);
  const scoreById = new Map(matches.map((match) => [match.id, match.score]));
  nodes.forEach((node, index) => {
    if (!scoreById.has(node.id)) {
      return;
    }
    const offset = index * 3;
    pointColors.array[offset] = 1;
    pointColors.array[offset + 1] = 1;
    pointColors.array[offset + 2] = 1;
  });
  pointColors.needsUpdate = true;
}

function checkHit(event) {
  if (!points) {
    return;
  }
  const bounds = graphCanvas.getBoundingClientRect();
  pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
  pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(points, false)[0];
  if (!hit || hit.index === undefined) {
    hidePopover();
    return;
  }
  showPopover(hit.index, event.clientX, event.clientY);
}

graphCanvas.addEventListener("pointermove", checkHit);
graphCanvas.addEventListener("pointerleave", hidePopover);
graphCanvas.addEventListener("keydown", (event) => {
  if (!nodes.length || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
    return;
  }
  event.preventDefault();
  const step = event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1;
  const index = (selectedIndex + step + nodes.length) % nodes.length;
  showPopover(index, window.innerWidth / 2, window.innerHeight / 2);
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    status.textContent = "Enter a topic to search.";
    return;
  }
  status.textContent = "Embedding query locally…";
  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 8 }),
    });
    const payload = await response.json();
    if (!response.ok || !Array.isArray(payload.matches)) {
      throw new Error(payload.error || "Search failed.");
    }
    highlight(payload.matches);
    status.textContent = `Highlighted ${payload.matches.length} semantic matches.`;
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Search failed.";
  }
});

function resize() {
  const width = graphRoot.clientWidth;
  const height = graphRoot.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
  audioWaveform.width = Math.max(1, Math.floor(audioWaveform.clientWidth * window.devicePixelRatio));
  audioWaveform.height = Math.max(1, Math.floor(audioWaveform.clientHeight * window.devicePixelRatio));
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

async function loadGraph() {
  try {
    const response = await fetch("/api/graph");
    const payload = await response.json();
    if (!response.ok || !Array.isArray(payload.nodes)) {
      throw new Error(payload.error || "Could not load the graph.");
    }
    nodes = payload.nodes.filter(validNode);
    if (nodes.length !== 50) {
      throw new Error("The server did not provide exactly 50 valid graph nodes.");
    }
    buildGraph();
    status.textContent = "50 attributed nodes loaded.";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Could not load the graph.";
  }
}

window.addEventListener("resize", resize);
resize();
loadGraph();
animate();
