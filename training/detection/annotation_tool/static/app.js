// Solar AI panel annotation tool — frontend logic.
// Boxes are always stored/sent in NORMALIZED [0,1] coordinates
// (cx, cy, w, h), matching the YOLO label format written server-side —
// the canvas display scale never leaks into saved data.

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let state = {
  images: [],        // [{filename, annotated, box_count}]
  index: 0,
  boxes: [],         // current image's boxes, normalized
  selected: -1,
  dirty: false,
  img: null,         // current HTMLImageElement
  draw: null,        // {x0,y0,x1,y1} in canvas pixel coords while dragging
};

function currentFilename() {
  return state.images[state.index] ? state.images[state.index].filename : null;
}

async function loadProgress() {
  const res = await fetch("/api/progress");
  const data = await res.json();
  state.images = data.images;
  renderSidebar();
  renderProgress(data);
}

function renderProgress(data) {
  document.getElementById("progress").textContent =
    `${data.annotated} / ${data.total} annotated  ·  ${data.total_boxes_so_far} boxes so far`;
}

function renderSidebar() {
  const el = document.getElementById("sidebar");
  el.innerHTML = "";
  state.images.forEach((im, i) => {
    const row = document.createElement("div");
    row.className = "side-item " + (im.annotated ? "done" : "pending") + (i === state.index ? " current" : "");
    row.textContent = im.filename;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = im.annotated ? `${im.box_count}` : "";
    row.appendChild(count);
    row.onclick = () => gotoIndex(i);
    el.appendChild(row);
  });
}

function confirmDiscardIfDirty() {
  if (!state.dirty) return true;
  return confirm("You have unsaved changes on this image. Discard them?");
}

async function gotoIndex(i) {
  if (i < 0 || i >= state.images.length) return;
  if (!confirmDiscardIfDirty()) return;
  state.index = i;
  state.selected = -1;
  state.dirty = false;
  await loadCurrentImage();
  renderSidebar();
}

async function loadCurrentImage() {
  const filename = currentFilename();
  document.getElementById("filename").textContent = filename;

  const res = await fetch(`/api/boxes?filename=${encodeURIComponent(filename)}`);
  const data = await res.json();
  state.boxes = data.boxes || [];

  const img = new Image();
  img.onload = () => {
    state.img = img;
    fitCanvas();
    redraw();
  };
  img.src = `/images/${encodeURIComponent(filename)}?t=${Date.now()}`;
}

function fitCanvas() {
  const maxW = Math.min(window.innerWidth - 320, 1100);
  const maxH = Math.min(window.innerHeight - 180, 760);
  const ratio = Math.min(maxW / state.img.naturalWidth, maxH / state.img.naturalHeight, 1);
  canvas.width = Math.round(state.img.naturalWidth * ratio);
  canvas.height = Math.round(state.img.naturalHeight * ratio);
}

function normToCanvas(b) {
  return {
    x: (b.cx - b.w / 2) * canvas.width,
    y: (b.cy - b.h / 2) * canvas.height,
    w: b.w * canvas.width,
    h: b.h * canvas.height,
  };
}

function canvasToNorm(x0, y0, x1, y1) {
  const xa = Math.min(x0, x1), xb = Math.max(x0, x1);
  const ya = Math.min(y0, y1), yb = Math.max(y0, y1);
  const cx = (xa + xb) / 2 / canvas.width;
  const cy = (ya + yb) / 2 / canvas.height;
  const w = (xb - xa) / canvas.width;
  const h = (yb - ya) / canvas.height;
  return { class_id: 0, cx, cy, w, h };
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);

  state.boxes.forEach((b, i) => {
    const r = normToCanvas(b);
    ctx.lineWidth = i === state.selected ? 3 : 2;
    ctx.strokeStyle = i === state.selected ? "#ff5050" : "#30d050";
    ctx.fillStyle = i === state.selected ? "rgba(255,80,80,0.15)" : "rgba(48,208,80,0.12)";
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = "12px sans-serif";
    ctx.fillText(`solar_panel #${i}`, r.x + 3, Math.max(12, r.y - 4));
  });

  if (state.draw) {
    const { x0, y0, x1, y1 } = state.draw;
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#40a0ff";
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    ctx.setLineDash([]);
  }
}

function hitTest(x, y) {
  // Topmost (last-drawn) box wins on overlap.
  for (let i = state.boxes.length - 1; i >= 0; i--) {
    const r = normToCanvas(state.boxes[i]);
    if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return i;
  }
  return -1;
}

function canvasPos(evt) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (evt.clientX - rect.left) * (canvas.width / rect.width),
    y: (evt.clientY - rect.top) * (canvas.height / rect.height),
  };
}

canvas.addEventListener("mousedown", (evt) => {
  const { x, y } = canvasPos(evt);
  const hit = hitTest(x, y);
  if (hit >= 0) {
    state.selected = hit;
    redraw();
    return;
  }
  state.selected = -1;
  state.draw = { x0: x, y0: y, x1: x, y1: y };
});

canvas.addEventListener("mousemove", (evt) => {
  if (!state.draw) return;
  const { x, y } = canvasPos(evt);
  state.draw.x1 = x;
  state.draw.y1 = y;
  redraw();
});

window.addEventListener("mouseup", (evt) => {
  if (!state.draw) return;
  const { x0, y0, x1, y1 } = state.draw;
  state.draw = null;
  // Ignore accidental clicks/tiny drags (< 4px) - not a real box.
  if (Math.abs(x1 - x0) < 4 || Math.abs(y1 - y0) < 4) {
    redraw();
    return;
  }
  const box = canvasToNorm(x0, y0, x1, y1);
  state.boxes.push(box);
  state.selected = state.boxes.length - 1;
  state.dirty = true;
  updateDirtyFlag();
  redraw();
});

function deleteSelected() {
  if (state.selected < 0) return;
  state.boxes.splice(state.selected, 1);
  state.selected = -1;
  state.dirty = true;
  updateDirtyFlag();
  redraw();
}

function updateDirtyFlag() {
  document.getElementById("dirtyFlag").textContent = state.dirty ? "● unsaved changes" : "";
}

async function saveCurrent() {
  const filename = currentFilename();
  const res = await fetch(`/api/boxes?filename=${encodeURIComponent(filename)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ boxes: state.boxes }),
  });
  if (!res.ok) {
    const err = await res.json();
    alert("Save failed: " + (err.error || res.statusText));
    return;
  }
  const data = await res.json();
  state.dirty = false;
  updateDirtyFlag();
  const im = state.images[state.index];
  im.annotated = true;
  im.box_count = data.saved_box_count;
  renderSidebar();
  const progRes = await fetch("/api/progress");
  renderProgress(await progRes.json());
}

document.getElementById("prevBtn").onclick = () => gotoIndex(state.index - 1);
document.getElementById("nextBtn").onclick = () => gotoIndex(state.index + 1);
document.getElementById("deleteBtn").onclick = deleteSelected;
document.getElementById("saveBtn").onclick = saveCurrent;
document.getElementById("zeroBtn").onclick = () => {
  if (state.boxes.length > 0 && !confirm("This clears all boxes on this image and marks it as confirmed zero-panel. Continue?")) return;
  state.boxes = [];
  state.selected = -1;
  state.dirty = true;
  updateDirtyFlag();
  redraw();
};

window.addEventListener("keydown", (evt) => {
  if (evt.key === "ArrowRight") gotoIndex(state.index + 1);
  else if (evt.key === "ArrowLeft") gotoIndex(state.index - 1);
  else if (evt.key === "s" || evt.key === "S") saveCurrent();
  else if (evt.key === "Delete" || evt.key === "Backspace") deleteSelected();
  else if (evt.key === "Escape") { state.selected = -1; redraw(); }
});

window.addEventListener("beforeunload", (evt) => {
  if (state.dirty) { evt.preventDefault(); evt.returnValue = ""; }
});

(async function init() {
  await loadProgress();
  await loadCurrentImage();
})();
