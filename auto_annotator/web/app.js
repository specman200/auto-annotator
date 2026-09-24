/* Auto Annotator GUI: review and correct locally-predicted boxes. */
'use strict';

const $ = (id) => document.getElementById(id);
const HANDLE = 11;         // drawn size of a resize handle, in screen pixels
const HANDLE_GRAB = 6;     // half-extent of its active area — kept close to the
                           // drawn size, so the target is where the eye says
const MIN_SIDE_FOR_EDGE_HANDLES = 30;  // narrower than this and a mid-edge handle
                           // cannot be told apart from the corners, so it is not offered
const MIN_BOX = 3;         // ignore drags smaller than this (screen pixels)
const MIN_BOX_PX = 2;      // ...and smaller than this in image pixels once clamped
const SAVE_DELAY = 500;    // debounce for autosave

const state = {
  root: '',
  images: [],
  classes: [],
  current: null,      // relative path of the open image
  record: null,       // { path, width, height, status, annotations: [] }
  selectedId: null,
  filter: 'all',
  query: '',
  conf: 0.25,
  merge: 'keep-human',
  view: { scale: 1, x: 0, y: 0 },
  drag: null,         // active pointer interaction
  hoveredId: null,    // box under the cursor, highlighted for discoverability
  drawMode: false,    // draw new boxes even when the drag starts inside one
  cursor: null,       // last pointer position, in canvas pixels
  undo: [],
  redo: [],
  dirty: false,
  revision: 0,        // bumped on every local edit, to spot stale save replies
  saveTimer: null,
  job: null,
};

const img = new Image();
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

/* ------------------------------------------------------------------ utils */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

let toastTimer = null;
function toast(message, isError = false) {
  const el = $('toast');
  el.textContent = message;
  el.classList.toggle('error', isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, isError ? 6000 : 2600);
}

function labelColor(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i++) hash = (hash * 31 + label.charCodeAt(i)) | 0;
  return `hsl(${Math.abs(hash) % 360}, 70%, 58%)`;
}

const uid = () => Math.random().toString(16).slice(2, 14);

/* Encode a project-relative path for a URL.
   encodeURI leaves #, ? and & alone, so "photo #3.jpg" would arrive at the
   server truncated — segment-wise encodeURIComponent keeps the separators
   while escaping everything else. */
const encodePath = (path) => path.split('/').map(encodeURIComponent).join('/');

/* ------------------------------------------------------------ data loading */

async function loadProject() {
  const data = await api('/api/project');
  state.root = data.root;
  state.classes = data.classes;
  state.images = data.images;
  state.conf = data.model.conf ?? state.conf;
  $('project-root').textContent = `${data.root}  ·  ${data.model.kind}` +
    (data.model.weights ? `:${data.model.weights}` : '');
  $('conf').value = state.conf;
  $('conf-value').textContent = Number(state.conf).toFixed(2);
  renderStats(data.stats);
  renderImageList();
  renderClasses();
  if (!state.current && state.images.length) openImage(state.images[0].path);
}

async function openImage(path) {
  if (state.dirty) await save();
  const record = await api(`/api/images/${encodePath(path)}`);
  state.current = path;
  state.record = record;
  state.selectedId = null;
  state.undo = [];
  state.redo = [];
  img.onload = () => { fitView(); render(); };
  img.onerror = () => toast(`could not load image: ${path}`, true);
  img.src = `/api/file/${encodePath(path)}`;
  $('canvas-empty').hidden = true;
  renderImageList();
  renderAnnotations();
  renderPosition();
}

function visibleImages() {
  return state.images.filter((item) => {
    if (state.filter !== 'all' && item.status !== state.filter) return false;
    if (state.query && !item.path.toLowerCase().includes(state.query)) return false;
    return true;
  });
}

/* --------------------------------------------------------------- rendering */

function renderImageList() {
  const list = $('image-list');
  list.innerHTML = '';
  for (const item of visibleImages()) {
    const li = document.createElement('li');
    li.className = item.path === state.current ? 'active' : '';
    li.innerHTML =
      `<span class="dot ${item.status}"></span>` +
      `<span class="name" title="${item.path}">${item.name}</span>` +
      `<span class="badge">${item.count || ''}</span>`;
    li.onclick = () => openImage(item.path);
    list.appendChild(li);
  }
}

function renderStats(stats) {
  if (!stats) return;
  const { by_status: status = {} } = stats;
  $('project-stats').textContent =
    `${stats.images} images · ${stats.boxes} boxes · ` +
    `${status.reviewed || 0} reviewed · ${status.predicted || 0} predicted`;
  for (const item of state.images) {
    if (item.path === state.current && state.record) {
      item.status = state.record.status;
      item.count = state.record.annotations.length;
    }
  }
}

function renderClasses() {
  const holder = $('class-list');
  holder.innerHTML = '';
  state.classes.forEach((name, index) => {
    const pill = document.createElement('div');
    pill.className = 'class-pill';
    pill.innerHTML =
      `<span class="swatch" style="background:${labelColor(name)}"></span>` +
      `<span>${name}</span>` +
      (index < 9 ? `<span class="key">${index + 1}</span>` : '');
    pill.onclick = () => applyLabel(name);
    holder.appendChild(pill);
  });
}

function renderAnnotations() {
  const list = $('annotation-list');
  list.innerHTML = '';
  const annotations = state.record ? state.record.annotations : [];
  $('box-count').textContent = annotations.length;

  for (const annotation of annotations) {
    const li = document.createElement('li');
    li.className = annotation.id === state.selectedId ? 'selected' : '';
    li.dataset.id = annotation.id;
    li.onclick = () => selectAnnotation(annotation.id);

    const source = document.createElement('span');
    source.className = `src ${annotation.source}`;
    source.title = annotation.source === 'model' ? 'predicted' : 'edited by hand';

    const select = document.createElement('select');
    const options = state.classes.includes(annotation.label)
      ? state.classes
      : [annotation.label, ...state.classes];
    for (const name of options) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      option.selected = name === annotation.label;
      select.appendChild(option);
    }
    select.onchange = () => {
      pushUndo();
      annotation.label = select.value;
      annotation.source = 'human';
      markDirty();
      renderAnnotations();
      render();
    };
    // Clicking the dropdown still selects the row — without rebuilding the
    // list, which would close the dropdown the user just opened.
    select.onclick = (event) => {
      event.stopPropagation();
      selectAnnotation(annotation.id, false);
    };

    const score = document.createElement('span');
    score.className = 'score';
    score.textContent = annotation.score == null ? '—' : annotation.score.toFixed(2);

    const remove = document.createElement('button');
    remove.className = 'del';
    remove.textContent = '✕';
    remove.title = 'delete box';
    remove.onclick = (event) => { event.stopPropagation(); deleteAnnotation(annotation.id); };

    li.append(source, select, score, remove);
    list.appendChild(li);
  }
}

function selectAnnotation(id, rebuild = true) {
  state.selectedId = id;
  if (rebuild) {
    renderAnnotations();
    $('annotation-list').querySelector(`[data-id="${id}"]`)
      ?.scrollIntoView({ block: 'nearest' });
  } else {
    for (const row of $('annotation-list').children) {
      row.classList.toggle('selected', row.dataset.id === id);
    }
  }
  render();
}

function renderPosition() {
  const list = visibleImages();
  const index = list.findIndex((item) => item.path === state.current);
  $('image-position').textContent = state.current
    ? `${index + 1} / ${list.length} · ${state.current}`
    : '–';
}

/* -------------------------------------------------------------- view maths */

function imageToScreen(x, y) {
  return { x: x * state.view.scale + state.view.x, y: y * state.view.scale + state.view.y };
}

function screenToImage(x, y) {
  return { x: (x - state.view.x) / state.view.scale, y: (y - state.view.y) / state.view.scale };
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return screenToImage(event.clientX - rect.left, event.clientY - rect.top);
}

function fitView() {
  if (!img.naturalWidth) return;
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight) * 0.96;
  state.view.scale = scale;
  state.view.x = (rect.width - img.naturalWidth * scale) / 2;
  state.view.y = (rect.height - img.naturalHeight * scale) / 2;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  // Scale by what the buffer actually is over what CSS shows, not by the
  // nominal ratio: the two differ once rounding is involved, and the
  // difference stretches everything drawn away from the cursor.
  ctx.setTransform(canvas.width / rect.width, 0, 0, canvas.height / rect.height, 0, 0);
  render();
}

/* Annotation boxes are normalized; drawing needs image pixels. */
function boxPixels(box) {
  return {
    x1: box.x1 * img.naturalWidth,
    y1: box.y1 * img.naturalHeight,
    x2: box.x2 * img.naturalWidth,
    y2: box.y2 * img.naturalHeight,
  };
}

/* A box clamped against the image edge can collapse to nothing; drop those. */
function isDegenerate(box) {
  return (box.x2 - box.x1) * img.naturalWidth < MIN_BOX_PX ||
         (box.y2 - box.y1) * img.naturalHeight < MIN_BOX_PX;
}

function toNormalized(x1, y1, x2, y2) {
  const w = img.naturalWidth, h = img.naturalHeight;
  return {
    x1: Math.max(0, Math.min(x1, x2) / w),
    y1: Math.max(0, Math.min(y1, y2) / h),
    x2: Math.min(1, Math.max(x1, x2) / w),
    y2: Math.min(1, Math.max(y1, y2) / h),
  };
}

/* ----------------------------------------------------------------- drawing */

function render() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!img.naturalWidth) return;

  const { scale, x, y } = state.view;
  ctx.imageSmoothingEnabled = scale < 3;
  ctx.drawImage(img, x, y, img.naturalWidth * scale, img.naturalHeight * scale);

  for (const annotation of state.record ? state.record.annotations : []) {
    drawBox(annotation, annotation.id === state.selectedId,
            annotation.id === state.hoveredId);
  }
  drawGuides();

  if (state.drag && state.drag.mode === 'draw' && state.drag.box) {
    const { x1, y1, x2, y2 } = state.drag.box;
    const a = imageToScreen(x1, y1), b = imageToScreen(x2, y2);
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = '#4c9aff';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
    ctx.restore();
  }
}

/* In draw mode the exact anchor matters, so show it: full-height and
   full-width lines through the cursor, and a marker on the corner a drag
   has already pinned. */
function drawGuides() {
  const drawing = state.drag && state.drag.mode === 'draw';
  if (!state.drawMode && !drawing) return;

  const rect = canvas.getBoundingClientRect();
  ctx.save();
  ctx.strokeStyle = 'rgba(76, 154, 255, 0.55)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);

  if (state.cursor) {
    const x = Math.round(state.cursor.x) + 0.5;   // crisp 1px lines
    const y = Math.round(state.cursor.y) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, rect.height);
    ctx.moveTo(0, y);
    ctx.lineTo(rect.width, y);
    ctx.stroke();
  }

  if (drawing) {
    const anchor = imageToScreen(state.drag.start.x, state.drag.start.y);
    ctx.setLineDash([]);
    ctx.fillStyle = '#4c9aff';
    ctx.fillRect(anchor.x - 3, anchor.y - 3, 6, 6);
    ctx.strokeStyle = '#0d1117';
    ctx.strokeRect(anchor.x - 3.5, anchor.y - 3.5, 7, 7);
  }
  ctx.restore();
}

function drawBox(annotation, selected, hovered = false) {
  const px = boxPixels(annotation.box);
  const a = imageToScreen(px.x1, px.y1);
  const b = imageToScreen(px.x2, px.y2);
  const w = b.x - a.x, h = b.y - a.y;
  const color = labelColor(annotation.label);

  ctx.save();
  ctx.lineWidth = selected ? 2.5 : (hovered ? 2 : 1.5);
  ctx.strokeStyle = color;
  if (hovered && !selected) {
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
  }
  if (annotation.source === 'model') ctx.setLineDash([6, 3]);
  ctx.strokeRect(a.x, a.y, w, h);
  ctx.setLineDash([]);

  if (selected) {
    ctx.fillStyle = color.replace('hsl', 'hsla').replace(')', ', 0.14)');
    ctx.fillRect(a.x, a.y, w, h);
  }

  // The label goes on before the handles: drawn after, it covered the top-left
  // handle, hiding a control that still responded to clicks. It also starts
  // clear of that handle so the two never sit on each other.
  const score = annotation.score == null ? '' : ` ${annotation.score.toFixed(2)}`;
  const text = `${annotation.label}${score}`;
  ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
  const width = ctx.measureText(text).width + 8;
  const inset = selected ? HANDLE / 2 + 2 : 0;
  const top = a.y > 16 ? a.y - 15 - inset : a.y + 1 + inset;
  ctx.fillStyle = color;
  ctx.fillRect(a.x + inset, top, width, 14);
  ctx.fillStyle = '#0d1117';
  ctx.fillText(text, a.x + inset + 4, top + 11);

  if (selected) {
    // Outlined so the exact centre is unmistakable at any zoom.
    for (const [, hx, hy] of handleSpecs(a, b)) {
      ctx.fillStyle = color;
      ctx.fillRect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
      ctx.strokeStyle = '#0d1117';
      ctx.lineWidth = 1;
      ctx.strokeRect(hx - HANDLE / 2 + 0.5, hy - HANDLE / 2 + 0.5, HANDLE - 1, HANDLE - 1);
    }
  }
  ctx.restore();
}

/* The handles a box actually offers, as [name, x, y] in screen pixels.

   Corners are always there. Mid-edge handles appear only when that edge is
   long enough for them to be distinguishable — on a small box they would sit
   almost on top of the corners, which is what made grabbing one a lottery. */
function handleSpecs(a, b) {
  const width = Math.abs(b.x - a.x);
  const height = Math.abs(b.y - a.y);
  const midX = (a.x + b.x) / 2;
  const midY = (a.y + b.y) / 2;

  const specs = [
    ['nw', a.x, a.y], ['ne', b.x, a.y], ['sw', a.x, b.y], ['se', b.x, b.y],
  ];
  if (width >= MIN_SIDE_FOR_EDGE_HANDLES) {
    specs.push(['n', midX, a.y], ['s', midX, b.y]);
  }
  if (height >= MIN_SIDE_FOR_EDGE_HANDLES) {
    specs.push(['w', a.x, midY], ['e', b.x, midY]);
  }
  return specs;
}

/* How far off a handle still counts as grabbing it: never more than a third of
   the box, so two handles can never claim the same pixel. */
function grabRadius(a, b) {
  const shortest = Math.min(Math.abs(b.x - a.x), Math.abs(b.y - a.y));
  return Math.max(3, Math.min(HANDLE_GRAB, shortest / 3));
}

/* ------------------------------------------------------------ hit  testing */

function handleAt(point) {
  const annotation = selectedAnnotation();
  if (!annotation) return null;
  const px = boxPixels(annotation.box);
  const a = imageToScreen(px.x1, px.y1);
  const b = imageToScreen(px.x2, px.y2);
  const screen = imageToScreen(point.x, point.y);
  const radius = grabRadius(a, b);

  // Nearest wins: with "first within tolerance" a click closer to the top edge
  // handle could still grab the corner one.
  let best = null;
  let bestDistance = Infinity;
  for (const [name, x, y] of handleSpecs(a, b)) {
    if (Math.abs(screen.x - x) > radius || Math.abs(screen.y - y) > radius) continue;
    const distance = Math.hypot(screen.x - x, screen.y - y);
    if (distance < bestDistance) {
      best = name;
      bestDistance = distance;
    }
  }
  return best;
}

function annotationsAt(point) {
  const annotations = state.record ? state.record.annotations : [];
  return annotations
    .filter((annotation) => {
      const px = boxPixels(annotation.box);
      return point.x >= px.x1 && point.x <= px.x2 &&
             point.y >= px.y1 && point.y <= px.y2;
    })
    .sort((a, b) => boxArea(a) - boxArea(b));   // innermost (smallest) first
}

function boxArea(annotation) {
  const box = annotation.box;
  return (box.x2 - box.x1) * (box.y2 - box.y1);
}

function annotationAt(point) {
  return annotationsAt(point)[0] || null;   // innermost wins
}

/* Clicking the same spot again steps to the next box under the cursor, so a
   box completely covered by another is still reachable on the canvas. */
function pickAnnotation(point) {
  const candidates = annotationsAt(point);
  if (!candidates.length) { state.lastPick = null; return null; }

  const tolerance = 4 / state.view.scale;
  const repeated = state.lastPick &&
    Math.abs(point.x - state.lastPick.x) < tolerance &&
    Math.abs(point.y - state.lastPick.y) < tolerance;
  state.lastPick = { x: point.x, y: point.y };

  if (!repeated) return candidates[0];
  const current = candidates.findIndex((a) => a.id === state.selectedId);
  const next = candidates[(current + 1) % candidates.length];
  if (candidates.length > 1 && current >= 0) {
    toast(`box ${((current + 1) % candidates.length) + 1} of ${candidates.length} here`);
  }
  return next;
}

function selectedAnnotation() {
  return (state.record ? state.record.annotations : [])
    .find((a) => a.id === state.selectedId) || null;
}

/* ----------------------------------------------------------- mouse  events */

let spaceDown = false;

canvas.addEventListener('pointerdown', (event) => {
  if (!state.record || !img.naturalWidth) return;
  canvas.setPointerCapture(event.pointerId);
  const point = pointerPosition(event);

  if (spaceDown || event.button === 1) {
    state.drag = { mode: 'pan', startX: event.clientX, startY: event.clientY,
                   originX: state.view.x, originY: state.view.y };
    canvas.classList.add('panning');
    return;
  }

  // Drawing a box inside another one is how nested labels get made (a wheel
  // inside a car), so a forced draw has to beat the move and resize gestures.
  const forceDraw = event.shiftKey || state.drawMode;

  const handle = forceDraw ? null : handleAt(point);
  if (handle) {
    pushUndo();
    state.drag = { mode: 'resize', handle, annotation: selectedAnnotation() };
    return;
  }

  const hit = forceDraw ? null : pickAnnotation(point);
  if (hit) {
    pushUndo();
    state.drag = { mode: 'move', annotation: hit, start: point,
                   origin: { ...hit.box } };
    selectAnnotation(hit.id);
    return;
  }

  state.selectedId = null;
  state.drag = { mode: 'draw', start: point, box: null, startClient: { x: event.clientX, y: event.clientY } };
  renderAnnotations();
  render();
});

const HANDLE_CURSORS = {
  nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize',
  n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize',
};

function updateCursor(point, shiftKey = false) {
  if (spaceDown) { canvas.style.cursor = 'grab'; return; }
  if (shiftKey || state.drawMode) { canvas.style.cursor = 'crosshair'; return; }
  const handle = handleAt(point);
  if (handle) { canvas.style.cursor = HANDLE_CURSORS[handle]; return; }
  canvas.style.cursor = annotationAt(point) ? 'move' : 'crosshair';
}

canvas.addEventListener('pointermove', (event) => {
  const point = pointerPosition(event);
  const rect = canvas.getBoundingClientRect();
  state.cursor = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  if (img.naturalWidth) {
    $('cursor-readout').textContent =
      `${Math.round(point.x)}, ${Math.round(point.y)} px`;
  }
  const drag = state.drag;
  if (!drag) {
    updateCursor(point, event.shiftKey);
    const hovered = annotationAt(point);
    const hoveredId = hovered ? hovered.id : null;
    if (hoveredId !== state.hoveredId) {
      state.hoveredId = hoveredId;
      for (const row of $('annotation-list').children) {
        row.classList.toggle('hovered', row.dataset.id === hoveredId);
      }
      render();
    }
    return;
  }

  if (drag.mode === 'pan') {
    state.view.x = drag.originX + (event.clientX - drag.startX);
    state.view.y = drag.originY + (event.clientY - drag.startY);
  } else if (drag.mode === 'draw') {
    drag.box = {
      x1: drag.start.x, y1: drag.start.y, x2: point.x, y2: point.y,
    };
  } else if (drag.mode === 'move' && drag.annotation) {
    const dx = (point.x - drag.start.x) / img.naturalWidth;
    const dy = (point.y - drag.start.y) / img.naturalHeight;
    const width = drag.origin.x2 - drag.origin.x1;
    const height = drag.origin.y2 - drag.origin.y1;
    const x1 = Math.max(0, Math.min(1 - width, drag.origin.x1 + dx));
    const y1 = Math.max(0, Math.min(1 - height, drag.origin.y1 + dy));
    drag.annotation.box = { x1, y1, x2: x1 + width, y2: y1 + height };
    drag.annotation.source = 'human';
  } else if (drag.mode === 'resize' && drag.annotation) {
    resizeBox(drag.annotation, drag.handle, point);
    drag.annotation.source = 'human';
  }
  render();
});

canvas.addEventListener('pointerup', (event) => {
  const drag = state.drag;
  state.drag = null;
  canvas.classList.remove('panning');
  if (!drag) return;

  if (drag.mode === 'draw') {
    const moved = drag.startClient &&
      (Math.abs(event.clientX - drag.startClient.x) > MIN_BOX ||
       Math.abs(event.clientY - drag.startClient.y) > MIN_BOX);
    if (moved && drag.box) {
      const point = pointerPosition(event);
      const box = toNormalized(drag.start.x, drag.start.y, point.x, point.y);
      if (isDegenerate(box)) {
        toast('box too small — drag inside the image');
      } else {
        addAnnotation(box);
      }
    }
  } else if (drag.mode === 'move' || drag.mode === 'resize') {
    if (drag.mode === 'resize' && isDegenerate(drag.annotation.box)) {
      restore(state.undo, state.redo);   // undo the collapse we pushed on grab
      toast('box too small — resize undone');
    } else {
      markDirty();
      renderAnnotations();
    }
  }
  render();
});

canvas.addEventListener('pointerleave', () => {
  state.cursor = null;
  render();
});

canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const cursorX = event.clientX - rect.left, cursorY = event.clientY - rect.top;
  const before = screenToImage(cursorX, cursorY);
  const factor = Math.exp(-event.deltaY * 0.0015);
  state.view.scale = Math.max(0.05, Math.min(40, state.view.scale * factor));
  const after = imageToScreen(before.x, before.y);
  state.view.x += cursorX - after.x;   // keep the point under the cursor fixed
  state.view.y += cursorY - after.y;
  render();
}, { passive: false });

function resizeBox(annotation, handle, point) {
  const box = { ...annotation.box };
  const nx = Math.max(0, Math.min(1, point.x / img.naturalWidth));
  const ny = Math.max(0, Math.min(1, point.y / img.naturalHeight));
  if (handle.includes('w')) box.x1 = nx;
  if (handle.includes('e')) box.x2 = nx;
  if (handle.includes('n')) box.y1 = ny;
  if (handle.includes('s')) box.y2 = ny;
  annotation.box = {
    x1: Math.min(box.x1, box.x2), y1: Math.min(box.y1, box.y2),
    x2: Math.max(box.x1, box.x2), y2: Math.max(box.y1, box.y2),
  };
}

/* ------------------------------------------------------------------- edits */

function pushUndo() {
  if (!state.record) return;
  state.undo.push(JSON.stringify(state.record.annotations));
  if (state.undo.length > 60) state.undo.shift();
  state.redo = [];
}

function restore(stack, counterStack) {
  if (!state.record || !stack.length) return;
  counterStack.push(JSON.stringify(state.record.annotations));
  state.record.annotations = JSON.parse(stack.pop());
  state.selectedId = null;
  markDirty();
  renderAnnotations();
  render();
}

function addAnnotation(box) {
  pushUndo();
  const label = state.classes[0] || 'object';
  const annotation = { id: uid(), label, box, score: null, source: 'human' };
  state.record.annotations.push(annotation);
  state.selectedId = annotation.id;
  if (!state.classes.length) { state.classes.push(label); renderClasses(); }
  markDirty();
  renderAnnotations();
  render();
}

function deleteAnnotation(id) {
  pushUndo();
  state.record.annotations = state.record.annotations.filter((a) => a.id !== id);
  if (state.selectedId === id) state.selectedId = null;
  markDirty();
  renderAnnotations();
  render();
}

/* Arrow keys move the selected box; with Shift they move its bottom-right
   corner, which is the way to adjust a box whose handles are off-screen or
   buried under other boxes. */
function nudgeSelection(key, resize) {
  const annotation = selectedAnnotation();
  if (!annotation || !img.naturalWidth) return;
  const stepX = 1 / img.naturalWidth;
  const stepY = 1 / img.naturalHeight;
  const dx = (key === 'ArrowRight' ? stepX : key === 'ArrowLeft' ? -stepX : 0);
  const dy = (key === 'ArrowDown' ? stepY : key === 'ArrowUp' ? -stepY : 0);

  if (!state.nudging) {   // one undo entry per burst of arrow presses
    pushUndo();
    state.nudging = true;
    clearTimeout(state.nudgeTimer);
  }
  clearTimeout(state.nudgeTimer);
  state.nudgeTimer = setTimeout(() => { state.nudging = false; }, 700);

  const box = { ...annotation.box };
  if (resize) {
    box.x2 = Math.min(1, Math.max(box.x1 + stepX, box.x2 + dx));
    box.y2 = Math.min(1, Math.max(box.y1 + stepY, box.y2 + dy));
  } else {
    const width = box.x2 - box.x1, height = box.y2 - box.y1;
    box.x1 = Math.max(0, Math.min(1 - width, box.x1 + dx));
    box.y1 = Math.max(0, Math.min(1 - height, box.y1 + dy));
    box.x2 = box.x1 + width;
    box.y2 = box.y1 + height;
  }
  annotation.box = box;
  annotation.source = 'human';
  markDirty();
  render();
}

function applyLabel(label) {
  const annotation = selectedAnnotation();
  if (!annotation) { toast('select a box first'); return; }
  pushUndo();
  annotation.label = label;
  annotation.source = 'human';
  markDirty();
  renderAnnotations();
  render();
}

function setDrawMode(enabled) {
  state.drawMode = enabled;
  $('btn-draw-mode').classList.toggle('active', enabled);
  canvas.style.cursor = 'crosshair';
  render();
  if (enabled) toast('draw mode: drags make new boxes, even inside existing ones');
}

function markDirty() {
  state.dirty = true;
  state.revision++;
  $('save-state').textContent = 'unsaved';
  $('save-state').classList.add('dirty');
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(save, SAVE_DELAY);
}

async function save() {
  if (!state.record || !state.dirty) return;
  clearTimeout(state.saveTimer);
  const payload = { annotations: state.record.annotations };
  // Touching a prediction means a human has looked at it.
  if (state.record.status === 'new') payload.status = 'predicted';

  // Remember what we are saving: edits made while the request is in flight
  // must not be thrown away when the reply lands.
  const revision = state.revision;
  const path = state.current;

  try {
    const data = await api(`/api/images/${encodePath(path)}/annotations`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    });
    state.classes = data.classes;
    renderStats(data.stats);
    renderClasses();
    renderImageList();

    if (state.current !== path) return;   // the user has moved on to another image

    if (state.revision !== revision) {
      // Edited again mid-save (an undo, say). Keep the newer boxes and save them.
      state.record.status = data.image.status;
      markDirty();
      return;
    }
    state.record = data.image;
    state.dirty = false;
    $('save-state').textContent = 'saved';
    $('save-state').classList.remove('dirty');
  } catch (error) {
    toast(`save failed: ${error.message}`, true);
  }
}

/* --------------------------------------------------------------- inference */

async function predictCurrent() {
  if (!state.current) return;
  await save();
  const button = $('btn-predict');
  button.disabled = true;
  button.textContent = 'running…';
  try {
    const data = await api(`/api/images/${encodePath(state.current)}/predict`, {
      method: 'POST',
      body: JSON.stringify({ conf: state.conf, merge: state.merge }),
    });
    state.record = data.image;
    state.classes = data.classes;
    state.selectedId = null;
    state.undo = [];
    renderStats(data.stats);
    renderClasses();
    renderAnnotations();
    renderImageList();
    render();
    toast(`${data.image.annotations.length} boxes on this image`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = 'Run on image';
  }
}

async function predictAll() {
  await save();
  try {
    const job = await api('/api/jobs/predict', {
      method: 'POST',
      body: JSON.stringify({ conf: state.conf, merge: state.merge, skip_reviewed: true }),
    });
    state.job = job.id;
    $('job-modal').hidden = false;
    $('job-close').hidden = true;
    $('job-cancel').hidden = false;
    pollJob();
  } catch (error) {
    toast(error.message, true);
  }
}

async function pollJob() {
  if (!state.job) return;
  try {
    const job = await api(`/api/jobs/${state.job}`);
    const percent = job.total ? Math.round((job.done / job.total) * 100) : 0;
    $('job-bar').style.width = `${percent}%`;
    $('job-text').textContent =
      job.state === 'running'
        ? `${job.done} / ${job.total} — ${job.current || ''}`
        : describeJob(job);

    if (job.state === 'running') {
      setTimeout(pollJob, 400);
      return;
    }
    state.job = null;
    $('job-cancel').hidden = true;
    $('job-close').hidden = false;
    await loadProject();
    if (state.current) await openImage(state.current);
  } catch (error) {
    state.job = null;
    $('job-text').textContent = error.message;
    $('job-cancel').hidden = true;
    $('job-close').hidden = false;
  }
}

function describeJob(job) {
  if (job.state === 'error') return `failed: ${job.error}`;
  if (job.state === 'cancelled') return 'cancelled';
  const result = job.result || {};
  return `done — ${result.processed} images, ${result.boxes} boxes` +
    (result.skipped ? `, ${result.skipped} skipped` : '') +
    (result.errors && result.errors.length ? `, ${result.errors.length} failed` : '');
}

/* ------------------------------------------------------------- navigation */

async function step(delta) {
  const list = visibleImages();
  const index = list.findIndex((item) => item.path === state.current);
  const next = list[Math.min(list.length - 1, Math.max(0, index + delta))];
  if (next && next.path !== state.current) await openImage(next.path);
}

async function markReviewed() {
  if (!state.current) return;
  await save();
  const data = await api(`/api/images/${encodePath(state.current)}/status`, {
    method: 'POST',
    body: JSON.stringify({ status: 'reviewed' }),
  });
  const entry = state.images.find((item) => item.path === state.current);
  if (entry) entry.status = data.image.status;
  if (state.record) state.record.status = data.image.status;
  renderStats(data.stats);
  renderImageList();
  await step(1);
}

/* ------------------------------------------------------------------ layout */
/* Pane sizes and collapsed panels are per-browser conveniences: if storage is
   unavailable the layout just starts at its defaults. */

const LAYOUT_KEY = 'auto-annotator.layout';
const MIN_CANVAS = 320;    // the image always keeps at least this much width
const MIN_SIDEBAR = 160;
const MIN_PANEL_BODY = 90; // a panel below a resized one stays usable
const HEAD_HEIGHT = 34;

function loadLayout() {
  try {
    return JSON.parse(localStorage.getItem(LAYOUT_KEY)) || {};
  } catch (_) {
    return {};
  }
}

function saveLayout(patch) {
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify({ ...loadLayout(), ...patch }));
  } catch (_) {
    /* private window or blocked storage: sizing simply will not persist */
  }
}

/* Clamp a sidebar width so the two sidebars can never crowd out the image. */
function clampWidth(width, otherWidth) {
  const room = window.innerWidth - otherWidth - MIN_CANVAS;
  return Math.round(Math.min(Math.max(width, MIN_SIDEBAR), Math.max(MIN_SIDEBAR, room)));
}

function clampPanelHeight(panel, height) {
  const sidebar = $('sidebar-right').clientHeight;
  return Math.round(Math.min(Math.max(height, HEAD_HEIGHT), sidebar));
}

/* Keep the right-hand panels within the sidebar.

   Clamping each panel on its own is not enough: what a resized panel may take
   depends on how tall the *content* of the others is, not on some minimum, so
   the sized panels are shrunk together until everything fits. Without this a
   stored height pushes the last panel off the bottom of the screen.
*/
function fitPanels() {
  const sidebar = $('sidebar-right');
  const panels = [...sidebar.querySelectorAll('.panel')];
  if (!panels.length || !sidebar.clientHeight) return;

  const splitters = sidebar.querySelectorAll('.splitter.horizontal').length * 5;
  const sized = panels.filter((panel) => (
    panel.style.flexBasis &&
    !panel.classList.contains('collapsed') &&
    !panel.classList.contains('grow')
  ));

  const untouched = panels
    .filter((panel) => !sized.includes(panel))
    .reduce((total, panel) => {
      if (panel.classList.contains('collapsed')) return total + HEAD_HEIGHT;
      if (panel.classList.contains('grow')) return total + HEAD_HEIGHT + MIN_PANEL_BODY;
      return total + panel.scrollHeight;   // its natural content height
    }, 0);

  const budget = sidebar.clientHeight - splitters - untouched;
  const requested = sized.reduce((total, panel) => total + parseFloat(panel.style.flexBasis), 0);
  if (requested <= budget) return;

  const factor = Math.max(0, budget) / requested;
  for (const panel of sized) {
    const shrunk = Math.max(HEAD_HEIGHT, parseFloat(panel.style.flexBasis) * factor);
    panel.style.flexBasis = `${Math.round(shrunk)}px`;
  }
}

function applyLayout() {
  const layout = loadLayout();
  const root = document.documentElement;
  const left = layout.leftWidth || $('sidebar-left').getBoundingClientRect().width;
  const right = layout.rightWidth || $('sidebar-right').getBoundingClientRect().width;
  root.style.setProperty('--left-width', `${clampWidth(left, right)}px`);
  root.style.setProperty('--right-width', `${clampWidth(right, left)}px`);

  for (const id of layout.collapsed || []) {
    document.getElementById(id)?.classList.add('collapsed');
  }
  for (const [id, height] of Object.entries(layout.panels || {})) {
    const panel = document.getElementById(id);
    if (panel && !panel.classList.contains('grow')) {
      panel.style.flexBasis = `${clampPanelHeight(panel, height)}px`;
    }
  }
  fitPanels();
}

/* A window that shrinks (or a layout stored on a bigger screen) must not leave
   the image or a panel squeezed to nothing. */
function reclampLayout() {
  const root = document.documentElement;
  const left = $('sidebar-left').getBoundingClientRect().width;
  const right = $('sidebar-right').getBoundingClientRect().width;
  root.style.setProperty('--left-width', `${clampWidth(left, right)}px`);
  root.style.setProperty('--right-width', `${clampWidth(right, left)}px`);
  for (const panel of document.querySelectorAll('.sidebar.right .panel')) {
    if (panel.style.flexBasis && !panel.classList.contains('collapsed')) {
      const current = parseFloat(panel.style.flexBasis);
      panel.style.flexBasis = `${clampPanelHeight(panel, current)}px`;
    }
  }
  fitPanels();
}

function resetLayout() {
  try {
    localStorage.removeItem(LAYOUT_KEY);
  } catch (_) { /* nothing stored to clear */ }
  const root = document.documentElement;
  root.style.removeProperty('--left-width');
  root.style.removeProperty('--right-width');
  for (const panel of document.querySelectorAll('.panel')) {
    panel.style.flexBasis = '';
    panel.classList.remove('collapsed');
  }
  resizeCanvas();
  toast('layout reset');
}

function setUpSidebarResize(splitter, sidebarId, key, fromRight) {
  const sidebar = $(sidebarId);
  splitter.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    splitter.setPointerCapture(event.pointerId);
    splitter.classList.add('dragging');
    const startX = event.clientX;
    const startWidth = sidebar.getBoundingClientRect().width;

    const move = (moveEvent) => {
      const delta = fromRight ? startX - moveEvent.clientX : moveEvent.clientX - startX;
      const otherWidth = (fromRight ? $('sidebar-left') : $('sidebar-right'))
        .getBoundingClientRect().width;
      const width = clampWidth(startWidth + delta, otherWidth);
      document.documentElement.style.setProperty(
        fromRight ? '--right-width' : '--left-width', `${width}px`
      );
      resizeCanvas();
    };
    const up = () => {
      splitter.classList.remove('dragging');
      splitter.removeEventListener('pointermove', move);
      splitter.removeEventListener('pointerup', up);
      saveLayout({ [key]: sidebar.getBoundingClientRect().width });
      resizeCanvas();
    };
    splitter.addEventListener('pointermove', move);
    splitter.addEventListener('pointerup', up);
  });
}

function setUpPanelResize(splitter) {
  const panel = document.getElementById(splitter.dataset.resizes);
  if (!panel) return;
  const fromBottom = splitter.hasAttribute('data-from-bottom');

  splitter.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    splitter.setPointerCapture(event.pointerId);
    splitter.classList.add('dragging');
    if (panel.classList.contains('collapsed')) togglePanel(panel);
    const startY = event.clientY;
    const startHeight = panel.getBoundingClientRect().height;

    const move = (moveEvent) => {
      const delta = fromBottom ? startY - moveEvent.clientY : moveEvent.clientY - startY;
      panel.style.flexBasis = `${clampPanelHeight(panel, startHeight + delta)}px`;
      fitPanels();
    };
    const up = () => {
      splitter.classList.remove('dragging');
      splitter.removeEventListener('pointermove', move);
      splitter.removeEventListener('pointerup', up);
      const panels = { ...(loadLayout().panels || {}) };
      panels[panel.id] = panel.getBoundingClientRect().height;
      saveLayout({ panels });
    };
    splitter.addEventListener('pointermove', move);
    splitter.addEventListener('pointerup', up);
  });
}

function togglePanel(panel) {
  panel.classList.toggle('collapsed');
  if (panel.classList.contains('collapsed')) {
    panel.style.flexBasis = '';
  }
  fitPanels();
  const collapsed = [...document.querySelectorAll('.panel.collapsed')].map((p) => p.id);
  saveLayout({ collapsed });
}

function setUpLayout() {
  applyLayout();
  $('btn-reset-layout').onclick = resetLayout;
  setUpSidebarResize($('split-left'), 'sidebar-left', 'leftWidth', false);
  setUpSidebarResize($('split-right'), 'sidebar-right', 'rightWidth', true);
  for (const splitter of document.querySelectorAll('.splitter.horizontal')) {
    setUpPanelResize(splitter);
  }
  for (const head of document.querySelectorAll('.panel-head')) {
    head.onclick = () => togglePanel(head.closest('.panel'));
  }
}

/* ------------------------------------------------------------------ wiring */

$('conf').addEventListener('input', (event) => {
  state.conf = Number(event.target.value);
  $('conf-value').textContent = state.conf.toFixed(2);
});
$('conf').addEventListener('change', () => {
  api('/api/model', { method: 'PUT', body: JSON.stringify({ conf: state.conf }) })
    .catch(() => {});
});
$('merge').addEventListener('change', (event) => { state.merge = event.target.value; });
$('btn-predict').onclick = predictCurrent;
$('btn-predict-all').onclick = predictAll;
$('btn-prev').onclick = () => step(-1);
$('btn-next').onclick = () => step(1);
$('btn-review').onclick = markReviewed;
$('btn-fit').onclick = () => { fitView(); render(); };
$('btn-draw-mode').onclick = () => setDrawMode(!state.drawMode);
$('job-cancel').onclick = () => {
  if (state.job) api(`/api/jobs/${state.job}/cancel`, { method: 'POST' }).catch(() => {});
};
$('job-close').onclick = () => { $('job-modal').hidden = true; };

$('btn-export').onclick = async () => {
  const format = $('export-format').value;
  try {
    const data = await api('/api/export', {
      method: 'POST',
      body: JSON.stringify({ format }),
    });
    toast(`exported ${format} → ${data.path}`);
  } catch (error) {
    toast(error.message, true);
  }
};

$('search').addEventListener('input', (event) => {
  state.query = event.target.value.trim().toLowerCase();
  renderImageList();
  renderPosition();
});

for (const chip of $('status-filters').querySelectorAll('.chip')) {
  chip.onclick = () => {
    $('status-filters').querySelectorAll('.chip').forEach((c) => c.classList.remove('active'));
    chip.classList.add('active');
    state.filter = chip.dataset.status;
    renderImageList();
    renderPosition();
  };
}

$('add-class-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = $('new-class');
  const name = input.value.trim();
  if (!name || state.classes.includes(name)) { input.value = ''; return; }
  state.classes.push(name);
  input.value = '';
  renderClasses();
  await api('/api/classes', {
    method: 'PUT',
    body: JSON.stringify({ classes: state.classes }),
  }).catch((error) => toast(error.message, true));
});

document.addEventListener('keydown', (event) => {
  const inField = event.target.matches('input, textarea');
  const inSelect = event.target.matches('select');
  if (inField) return;
  // A select keeps its own arrow/typeahead keys, but Delete means "drop the box".
  if (inSelect && event.key !== 'Delete' && event.key !== 'Backspace') return;
  if (event.code === 'Space') { spaceDown = true; return; }

  const ctrl = event.ctrlKey || event.metaKey;
  if (ctrl && event.key.toLowerCase() === 's') { event.preventDefault(); save(); return; }
  if (ctrl && event.key.toLowerCase() === 'z') {
    event.preventDefault();
    event.shiftKey ? restore(state.redo, state.undo) : restore(state.undo, state.redo);
    return;
  }

  if (event.key.startsWith('Arrow') && selectedAnnotation() && !inSelect) {
    event.preventDefault();
    nudgeSelection(event.key, event.shiftKey);
    return;
  }

  switch (event.key) {
    case 'Delete': case 'Backspace':
      if (state.selectedId) { event.preventDefault(); deleteAnnotation(state.selectedId); }
      break;
    case 'Escape':
      if (state.drawMode) setDrawMode(false);
      state.selectedId = null; renderAnnotations(); render(); break;
    case 'd': case 'D': setDrawMode(!state.drawMode); break;
    case 'f': case 'F': fitView(); render(); break;
    case 'n': case 'N': case 'ArrowRight': step(1); break;
    case 'p': case 'P': case 'ArrowLeft': step(-1); break;
    case 'r': case 'R': predictCurrent(); break;
    case 'v': case 'V': markReviewed(); break;
    default:
      if (/^[1-9]$/.test(event.key)) {
        const label = state.classes[Number(event.key) - 1];
        if (label) applyLabel(label);
      }
  }
});

document.addEventListener('keyup', (event) => {
  if (event.code === 'Space') spaceDown = false;
});

window.addEventListener('beforeunload', (event) => {
  if (state.dirty) { save(); event.preventDefault(); event.returnValue = ''; }
});

window.addEventListener('resize', () => { reclampLayout(); resizeCanvas(); });

setUpLayout();
resizeCanvas();
loadProject().catch((error) => toast(`could not load project: ${error.message}`, true));
