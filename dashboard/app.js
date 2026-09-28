/* Enhanced-sampling demo dashboard. Static: everything comes from data/ (see SCHEMA.md). */
"use strict";

const state = {
  index: null,
  view: null,            // "comparison" or a method key
  method: null,          // loaded <key>.json
  methodCache: {},
};
const traj = {
  key: null, meta: null, coords: null, nFrames: 0, nAtoms: 0,
  viewer: null, model: null, atoms: null, frame: 0, timer: null,
  hbPairs: [], topology: null, color: "#2a78d6",
};

const TEXT = "#0b0b0b", TEXT2 = "#52514e", MUTED = "#85847f", GRID = "#ebeae6";
const SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];

function el(id) { return document.getElementById(id); }
function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
async function fetchJSON(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}
async function fetchText(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.text();
}
function isObj(x) { return x && typeof x === "object" && !Array.isArray(x); }
function merge(base, over) {
  const out = { ...base };
  for (const [k, v] of Object.entries(over || {})) {
    out[k] = isObj(v) && isObj(base[k]) ? merge(base[k], v) : v;
  }
  return out;
}
function axis(extra) {
  return merge({
    gridcolor: GRID, zerolinecolor: GRID, linecolor: "#c9c8c3", tickcolor: "#c9c8c3",
    tickfont: { color: TEXT2, size: 13 }, title: { font: { color: TEXT2, size: 14 } },
    automargin: true,
  }, extra || {});
}
function baseLayout() {
  return {
    font: { family: '-apple-system, "Segoe UI", Helvetica, Arial, sans-serif', size: 14, color: TEXT },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    margin: { l: 60, r: 20, t: 16, b: 50 },
    xaxis: axis(), yaxis: axis(),
    legend: { font: { color: TEXT2, size: 13 }, bgcolor: "rgba(255,255,255,0.8)" },
    hovermode: "closest",
    hoverlabel: { font: { size: 13 } },
  };
}
/* Every axis the panel layout mentions (xaxis2, yaxis3, ...) gets the base axis style too. */
function panelLayout(over) {
  let lay = baseLayout();
  for (const k of Object.keys(over || {})) {
    if (/^[xy]axis\d*$/.test(k) && !lay[k]) lay[k] = axis();
  }
  return merge(lay, over || {});
}
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

/* ================================================================ sidebar */

function renderSidebar() {
  const idx = state.index;
  el("system-blurb").textContent = idx.system?.blurb || "";
  el("generated").textContent = idx.generated ? `data generated ${idx.generated}` : "";
  const host = el("method-list");
  host.innerHTML = "";
  for (const m of idx.methods) {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.key = m.key;
    const dF = (m.dF === null || m.dF === undefined) ? "ΔF —"
      : `ΔF ${Number(m.dF).toFixed(1)}${m.dF_err != null ? " ± " + Number(m.dF_err).toFixed(1) : ""} kcal/mol`;
    const ns = m.total_ns != null ? `${Number(m.total_ns).toFixed(m.total_ns < 10 ? 1 : 0)} ns` : "";
    card.innerHTML = `
      <div class="card-head">
        <span class="swatch" style="background:${esc(m.color)}"></span>
        <span class="card-name">${esc(m.name)}</span>
      </div>
      <div class="card-meta">${esc(ns)}${ns ? " · " : ""}${esc(dF)}</div>`;
    const lin = idx.lineage?.[m.key];
    if (lin && lin.nodes?.length) card.appendChild(lineageSVG(m, lin));
    card.addEventListener("click", () => selectMethod(m.key));
    host.appendChild(card);
  }
}

function highlightSidebar() {
  el("nav-comparison").classList.toggle("active", state.view === "comparison");
  el("nav-intro").classList.toggle("active", state.view === "intro");
  el("nav-estimators").classList.toggle("active", state.view === "estimators");
  el("intro-section").style.display = state.view === "intro" ? "" : "none";
  document.querySelectorAll(".card").forEach(c => c.classList.toggle("active", c.dataset.key === state.view));
  document.querySelectorAll(".lineage .node").forEach(n =>
    n.classList.toggle("current", !!traj.key && n.dataset.traj === traj.key && n.dataset.method === state.view));
}

/* Layered DAG: layer = longest path from a root. Big layers become thin bars. */
function lineageSVG(method, lin) {
  const NS = "http://www.w3.org/2000/svg";
  const nodes = lin.nodes, byId = {};
  nodes.forEach(n => { byId[n.id] = n; });
  const edges = (lin.edges || []).filter(([a, b]) => byId[a] && byId[b]);
  const parents = {};
  edges.forEach(([a, b]) => { (parents[b] = parents[b] || []).push(a); });
  const depth = {};
  const depthOf = (id, seen = new Set()) => {
    if (depth[id] !== undefined) return depth[id];
    if (seen.has(id)) return 0;
    seen.add(id);
    const ps = parents[id] || [];
    depth[id] = ps.length ? 1 + Math.max(...ps.map(p => depthOf(p, seen))) : 0;
    return depth[id];
  };
  nodes.forEach(n => depthOf(n.id));
  const layers = [];
  nodes.forEach(n => { (layers[depth[n.id]] = layers[depth[n.id]] || []).push(n); });
  for (let i = 0; i < layers.length; i++) layers[i] = layers[i] || [];

  const W = 320, GAP = 12;
  const weights = layers.map(L => L.some(n => n.strip) ? 3.2 : 1);
  const unit = (W - GAP * (layers.length - 1)) / weights.reduce((a, b) => a + b, 0);
  const BOX_H = 18, VSP = 5;
  const layerH = layers.map(L => L.some(n => n.strip) ? 46 : L.length > 8 ? Math.min(110, Math.max(46, L.length * 2.2)) : L.length * (BOX_H + VSP) - VSP);
  const H = Math.max(30, ...layerH) + 6;

  const pos = {};
  let x = 0;
  layers.forEach((L, i) => {
    const w = weights[i] * unit, lh = layerH[i], y0 = (H - lh) / 2;
    if (L.length > 8 && !L.some(n => n.strip)) {
      const step = lh / L.length, bh = Math.max(1, step - (step > 3 ? 1 : 0.3));
      L.forEach((n, j) => { pos[n.id] = { x, y: y0 + j * step, w, h: bh, thin: true }; });
    } else {
      L.forEach((n, j) => {
        const h = n.strip ? lh : BOX_H;
        pos[n.id] = { x, y: y0 + j * (BOX_H + VSP), w, h };
      });
    }
    x += w + GAP;
  });

  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "lineage");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("height", H);
  const mk = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    (parent || svg).appendChild(e);
    return e;
  };
  const defs = mk("defs", {});
  const mid = `arrow-${method.key}`;
  const marker = mk("marker", { id: mid, viewBox: "0 0 6 6", refX: 5.5, refY: 3, markerWidth: 5, markerHeight: 5, orient: "auto" }, defs);
  mk("path", { d: "M0,0 L6,3 L0,6 z", fill: "#9a9994" }, marker);

  const many = edges.length > 20;
  edges.forEach(([a, b]) => {
    const p = pos[a], q = pos[b];
    const x1 = p.x + p.w, y1 = p.y + p.h / 2, x2 = q.x - 1, y2 = q.y + q.h / 2;
    const mx = (x1 + x2) / 2;
    mk("path", {
      class: "edge", d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`,
      "stroke-width": many ? 0.5 : 1.2, "stroke-opacity": many ? 0.45 : 0.9,
      ...(many ? {} : { "marker-end": `url(#${mid})` }),
    });
  });

  nodes.forEach(n => {
    const p = pos[n.id];
    const g = mk("g", { class: "node" + (n.traj ? " clickable" : "") });
    g.dataset.traj = n.traj || "";
    g.dataset.method = method.key;
    const shared = ["build", "minimize", "equil"].includes(n.kind);
    const fill = shared ? "#e9e8e4" : method.color;
    mk("title", {}, g).textContent = `${n.label}${n.t_ps ? ` · ${fmtPs(n.t_ps)}` : ""}${n.traj ? " · click to watch" : ""}`;
    if (n.strip) {
      mk("rect", { x: p.x, y: p.y, width: p.w, height: p.h, rx: 3, fill: "#fff", stroke: "#c9c8c3" }, g);
      const ns = n.strip.n || [], dm = n.strip.dmax || [];
      const nMax = Math.max(1, ...ns), dLo = Math.min(...dm), dHi = Math.max(...dm);
      const bw = (p.w - 4) / Math.max(1, ns.length);
      ns.forEach((v, i) => {
        const h = (p.h - 4) * v / nMax;
        const t = dHi > dLo ? (dm[i] - dLo) / (dHi - dLo) : 0.5;
        mk("rect", { x: p.x + 2 + i * bw, y: p.y + p.h - 2 - h, width: Math.max(0.6, bw - (bw > 2 ? 0.4 : 0)), height: h,
                     fill: SEQ[Math.min(SEQ.length - 1, Math.floor(t * SEQ.length))] }, g);
      });
    } else {
      mk("rect", { x: p.x, y: p.y, width: p.w, height: p.h, rx: p.thin ? 0.5 : 3, fill,
                   stroke: p.thin ? "none" : "#fff", "stroke-width": 1 }, g);
      if (!p.thin && p.w > 34) {
        const maxChars = Math.floor(p.w / 5.2);
        const txt = shortLabel(n, maxChars);
        mk("text", { x: p.x + 4, y: p.y + p.h / 2 + 3, fill: shared ? TEXT : "#fff",
                     style: shared ? "" : "fill:#fff" }, g).textContent = txt;
      }
    }
    if (n.traj) {
      g.addEventListener("click", ev => {
        ev.stopPropagation();
        selectMethod(method.key, n.traj);
      });
    }
  });
  return svg;
}

function shortLabel(n, maxChars) {
  const s = n.kind === "prod" || n.kind === "equil" || n.kind === "build" || n.kind === "minimize"
    ? (n.kind === "prod" ? n.label.split(",")[0] : n.kind) : n.label;
  return s.length > maxChars ? s.slice(0, Math.max(1, maxChars - 1)) + "…" : s;
}
function fmtPs(ps) {
  return ps >= 1000 ? `${(ps / 1000).toFixed(ps >= 10000 ? 0 : 1)} ns` : `${Math.round(ps)} ps`;
}

/* ================================================================ views */

function setHeader(title, blurb, feHow, stats, note) {
  el("title").textContent = title;
  el("blurb").textContent = blurb || "";
  el("fe-how").textContent = feHow || "";
  el("stats").innerHTML = (stats || []).map(s => `<div class="stat"><b>${esc(s.value)}</b><span>${esc(s.label)}</span></div>`).join("");
  el("note").textContent = note || "";
}

function renderPanels(panels) {
  const host = el("panels");
  host.querySelectorAll(".plot").forEach(d => { try { Plotly.purge(d); } catch (e) { /* ignore */ } });
  host.innerHTML = "";
  (panels || []).forEach((p, i) => {
    const box = document.createElement("div");
    box.className = "panel" + (p.wide ? " wide" : "");
    box.innerHTML = `<h3>${esc(p.title)}${p.hint ? `<span class="hint">${esc(p.hint)}</span>` : ""}</h3>`;
    const div = document.createElement("div");
    div.className = "plot";
    div.id = `panel-${p.id || i}`;
    if (p.layout?.height) div.style.height = p.layout.height + "px";
    box.appendChild(div);
    host.appendChild(box);
    try {
      Plotly.react(div, p.data || [], panelLayout(p.layout), PLOT_CONFIG);
    } catch (err) {
      div.innerHTML = `<p class="note">could not draw: ${esc(err)}</p>`;
    }
  });
}

/* ================================================================ ensemble grid */

const ens = { key: null, grid: null, viewers: [], atoms: [], models: [], coords: [], metas: [], n: 0, frame: 0, timer: null };
const TINT = ["#ffffff", "#eef5fd", "#dde9fa", "#cde2fb", "#b7d3f6", "#9ec5f4"];
const PAIR = ["#eda100", "#e87ba4", "#1baf7a", "#eb6834", "#4a3aa7"];

function renderViewToggles() {
  const e = state.method?.ensemble;
  ["view-toggle-single", "view-toggle-grid"].forEach((id, i) => {
    const host = el(id);
    if (!e || !e.members?.length) { host.innerHTML = ""; return; }
    host.innerHTML = `<button class="${i === 0 ? "active" : ""}" data-v="single">one trajectory</button>` +
      `<button class="${i === 1 ? "active" : ""}" data-v="grid">all ${e.members.length} at once${e.exchange ? " (with exchanges)" : ""}</button>`;
    host.querySelectorAll("button").forEach(b => b.addEventListener("click", () => b.dataset.v === "grid" ? showEnsemble() : showSingle()));
  });
}
function showSingle() {
  stopEns();
  el("ens-section").style.display = "none";
  el("traj-section").style.display = "";
  if (traj.viewer) traj.viewer.resize();
}
function hideEnsemble() { stopEns(); el("ens-section").style.display = "none"; }

async function showEnsemble() {
  const e = state.method?.ensemble;
  if (!e) return;
  stopPlay();
  el("traj-section").style.display = "none";
  el("ens-section").style.display = "";
  el("ens-title").textContent = e.title;
  el("ens-hint").textContent = e.hint || "";
  if (ens.key === state.view) { showEnsFrame(ens.frame); return; }
  el("ens-readout").textContent = "loading…";
  const M = e.members.length;
  const cols = M <= 10 ? Math.min(5, M) : 4, rows = Math.ceil(M / cols);
  const host = el("ens-grid");
  host.innerHTML = "";
  host.style.height = `${rows * 250}px`;
  const top = await ensureTopology();
  const loaded = await Promise.all(e.members.map(async m => {
    const meta = await fetchJSON(`data/traj/${m.key}.json`);
    const buf = await (await fetch(`data/traj/${meta.bin || m.key + ".bin"}`, { cache: "no-store" })).arrayBuffer();
    return [meta, new Float32Array(buf)];
  }));
  ens.grid = $3Dmol.createViewerGrid("ens-grid", { rows, cols, control_all: true }, { backgroundColor: "white" });
  ens.viewers = []; ens.atoms = []; ens.models = []; ens.metas = []; ens.coords = [];
  const carbon = traj.color;
  const colorfunc = atom => {
    const x = (atom.elem || "").toUpperCase();
    return x === "C" ? carbon : x === "O" ? "#e34948" : x === "N" ? "#3a4fc4" : "#d8d7d2";
  };
  loaded.forEach(([meta, c], i) => {
    const v = ens.grid[Math.floor(i / cols)][i % cols];
    const model = v.addModel(top, "pdb", { keepH: true });
    v.setStyle({ elem: "H", invert: true }, { stick: { radius: 0.2, colorfunc }, sphere: { scale: 0.28, colorfunc } });
    ens.viewers.push(v); ens.models.push(model); ens.atoms.push(model.selectedAtoms({}));
    ens.metas.push(meta); ens.coords.push(c);
  });
  ens.n = Math.min(...ens.metas.map(m => m.n_frames));
  ens.key = state.view;
  el("ens-frame").max = String(Math.max(0, ens.n - 1));
  showEnsFrame(0, true);
  ens.viewers.forEach(v => { v.zoomTo(); v.zoom(0.9); v.render(); });
}

function showEnsFrame(f, first) {
  const e = state.method?.ensemble;
  if (!e || !ens.n) return;
  f = Math.max(0, Math.min(ens.n - 1, f));
  ens.frame = f;
  const pairs = (e.events && e.events[f]) || [];
  const pairOf = {};
  pairs.forEach(([a, b], k) => { pairOf[a] = [b, PAIR[k % PAIR.length]]; pairOf[b] = [a, PAIR[k % PAIR.length]]; });
  ens.viewers.forEach((v, m) => {
    const c = ens.coords[m], n = ens.atoms[m].length, base = f * n * 3;
    ens.atoms[m].forEach((at, a) => { at.x = c[base + 3 * a]; at.y = c[base + 3 * a + 1]; at.z = c[base + 3 * a + 2]; });
    ens.models[m].molObj = null;
    const t = e.tint ? e.tint[m][f] : null;
    const bg = pairOf[m] ? pairOf[m][1] + "55" : t !== null ? TINT[Math.round(t * (TINT.length - 1))] : "#ffffff";
    v.setBackgroundColor(bg.length > 7 ? bg.slice(0, 7) : bg, bg.length > 7 ? 0.35 : 1);
    v.removeAllLabels();
    const meta = ens.metas[m];
    const txt = (meta.frame_label?.[f] || e.members[m].label) + (meta.d_ee ? ` · d_ee ${Number(meta.d_ee[f]).toFixed(1)} Å` : "") +
      (pairOf[m] ? `  ⇄ ${e.members[pairOf[m][0]].label}` : "");
    v.addLabel(txt, { position: { x: 6, y: 6, z: 0 }, useScreen: true, fontSize: 12, fontColor: "#0b0b0b",
      backgroundColor: pairOf[m] ? pairOf[m][1] : "#ffffff", backgroundOpacity: pairOf[m] ? 0.9 : 0.75, borderThickness: 0 });
    v.render();
  });
  el("ens-frame").value = String(f);
  const t = ens.metas[0].t_ns?.[f];
  el("ens-readout").textContent = `${f + 1}/${ens.n}` + (t !== undefined ? ` · ${fmtNs(t)}` : "");
  el("ens-events").textContent = e.exchange ? (pairs.length ? `${pairs.length} swap${pairs.length > 1 ? "s" : ""} since the previous frame` : "no swaps since the previous frame") : "";
}
function toggleEns() {
  if (ens.timer) { stopEns(); return; }
  if (!ens.n) return;
  ens.timer = setInterval(() => showEnsFrame((ens.frame + 1) % ens.n), Math.max(120, Number(el("speed").value) * 2));
  el("ens-play").innerHTML = "&#10073;&#10073; pause";
}
function stopEns() {
  if (ens.timer) { clearInterval(ens.timer); ens.timer = null; }
  el("ens-play").innerHTML = "&#9654; play";
}

/* ================================================================ intro */

const introViewers = [];
async function showIntro() {
  stopPlay(); hideEnsemble(); stopTimeline();
  state.view = "intro";
  state.method = null;
  highlightSidebar();
  el("traj-section").style.display = "none";
  const intro = state.index.intro;
  if (!intro) { setHeader("The system", state.index.system?.blurb, "", [], "running… intro not built yet"); renderPanels([]); return; }
  setHeader(intro.title, state.index.system?.name || "", "", [], "");
  el("intro-facts").innerHTML = intro.facts.map(f => `<li>${f}</li>`).join("");
  renderPanels(intro.panels || []);
  const host = el("intro-structs");
  if (host.dataset.built) return;
  host.dataset.built = "1";
  const top = await ensureTopology();
  for (const s of intro.structures) {
    const card = document.createElement("div");
    card.className = "intro-struct";
    card.innerHTML = `<h3>${esc(s.label)}</h3><div class="meta">d<sub>ee</sub> = ${Number(s.d_ee).toFixed(1)} Å · ${s.n_hb} helical H-bonds</div><div class="intro-viewer"></div>`;
    host.appendChild(card);
    const [meta, buf] = await Promise.all([fetchJSON(`data/traj/${s.traj}.json`),
      fetch(`data/traj/${s.traj}.bin`, { cache: "no-store" }).then(r => r.arrayBuffer())]);
    const c = new Float32Array(buf);
    const v = $3Dmol.createViewer(card.querySelector(".intro-viewer"), { backgroundColor: "white" });
    const model = v.addModel(top, "pdb", { keepH: true });
    const atoms = model.selectedAtoms({});
    atoms.forEach((a, i) => { a.x = c[3 * i]; a.y = c[3 * i + 1]; a.z = c[3 * i + 2]; });
    model.molObj = null;
    const colorfunc = atom => {
      const e = (atom.elem || "").toUpperCase();
      return e === "C" ? "#85847f" : e === "O" ? "#e34948" : e === "N" ? "#3a4fc4" : "#d8d7d2";
    };
    v.setStyle({ elem: "H", invert: true }, { stick: { radius: 0.2, colorfunc }, sphere: { scale: 0.28, colorfunc } });
    const [i, j] = intro.d_pair;
    const p = { x: c[3 * i], y: c[3 * i + 1], z: c[3 * i + 2] }, q = { x: c[3 * j], y: c[3 * j + 1], z: c[3 * j + 2] };
    v.addCylinder({ start: p, end: q, radius: 0.18, color: "#2a78d6", dashed: true, dashLength: 0.6, gapLength: 0.35, fromCap: 1, toCap: 1 });
    [p, q].forEach(pt => v.addSphere({ center: pt, radius: 0.55, color: "#2a78d6" }));
    v.addLabel(`d_ee = ${Number(s.d_ee).toFixed(1)} Å`, { position: { x: (p.x + q.x) / 2, y: (p.y + q.y) / 2, z: (p.z + q.z) / 2 },
      backgroundColor: "white", backgroundOpacity: 0.85, fontColor: "#0b0b0b", fontSize: 14, borderColor: "#2a78d6", borderThickness: 1 });
    // hbonds
    hbondPairs(atoms).forEach(([o, h]) => {
      const a = { x: c[3 * o], y: c[3 * o + 1], z: c[3 * o + 2] }, b = { x: c[3 * h], y: c[3 * h + 1], z: c[3 * h + 2] };
      if (Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) < 2.6)
        v.addCylinder({ start: a, end: b, radius: 0.07, color: "#c98500", dashed: true, dashLength: 0.25, gapLength: 0.18 });
    });
    v.zoomTo(); v.zoom(0.95); v.render();
    introViewers.push(v);
  }
}

async function showEstimators() {
  stopPlay(); hideEnsemble(); stopTimeline();
  state.view = "estimators";
  state.method = null;
  highlightSidebar();
  el("traj-section").style.display = "none";
  const e = state.index.estimators;
  if (!e) { setHeader("Estimators", "", "", [], "not built yet"); renderPanels([]); return; }
  setHeader(e.title, e.blurb, e.fe_how, e.stats, "");
  renderPanels(e.panels);
}

async function showComparison() {
  stopPlay(); hideEnsemble(); stopTimeline();
  state.view = "comparison";
  state.method = null;
  highlightSidebar();
  el("traj-section").style.display = "none";
  const n = state.index.methods.length;
  setHeader("Comparison: all methods",
    state.index.system?.blurb,
    "every method's estimate of the PMF along d_ee, on the same axes, against the umbrella-sampling reference.",
    [{ value: String(n), label: "methods" },
     { value: `${state.index.methods.reduce((a, m) => a + (Number(m.total_ns) || 0), 0).toFixed(0)} ns`, label: "total simulated" }],
    state.index.comparison?.panels?.length ? "" : "running… comparison not built yet");
  renderPanels(state.index.comparison?.panels || []);
  renderTimeline(state.index.comparison?.timeline);
}

/* "Play time forward": every method's PMF using at most B ns of its own sampling (all copies). */
const tl = { timer: null, pos: 0, N: 240 };
function renderTimeline(t) {
  if (!t || !t.methods?.length) return;
  const host = el("panels");
  const box = document.createElement("div");
  box.className = "panel wide";
  box.innerHTML = `<h3>Free-energy profiles as sampling accumulates
      <span class="hint">slider = sampling budget per method (total ns over all its copies, log scale); each curve is that method's estimate using at most that much; blank = no estimate yet</span></h3>
    <div class="controls"><button id="tl-play">&#9654; play</button>
      <input type="range" id="tl-slider" min="0" max="${tl.N}" value="${tl.N}" style="flex:1">
      <span id="tl-readout" class="readout"></span></div>
    <div id="tl-plot" class="plot" style="height:460px"></div>`;
  host.insertBefore(box, host.firstChild);
  const traces = [];
  if (t.ref) traces.push({ type: "scatter", mode: "lines", x: t.x, y: t.ref, name: "reference (US+MBAR)", line: { color: TEXT2, width: 2, dash: "dash" } });
  t.methods.forEach(m => traces.push({ type: "scatter", mode: "lines", x: t.x, y: t.x.map(() => null), name: m.name,
    line: { color: m.color, width: 2.5, ...(m.dash ? { dash: m.dash } : {}) } }));
  Plotly.react("tl-plot", traces, panelLayout({
    xaxis: { title: { text: "d_ee (Å)" }, range: [3, 36] }, yaxis: { title: { text: "F (kcal/mol)" }, range: [0, 30] },
    legend: { x: 1.01, y: 1 }, margin: { l: 60, r: 20, t: 10, b: 50 },
  }), PLOT_CONFIG);
  el("tl-play").addEventListener("click", () => {
    if (tl.timer) { stopTimeline(); return; }
    if (tl.pos >= tl.N) tl.pos = 0;
    el("tl-play").innerHTML = "&#10073;&#10073; pause";
    tl.timer = setInterval(() => { if (tl.pos >= tl.N) { stopTimeline(); return; } showTimeline(t, tl.pos + 1); }, 60);
  });
  el("tl-slider").addEventListener("input", e => { stopTimeline(); showTimeline(t, Number(e.target.value)); });
  showTimeline(t, tl.N);
}
function stopTimeline() {
  if (tl.timer) { clearInterval(tl.timer); tl.timer = null; }
  const b = el("tl-play"); if (b) b.innerHTML = "&#9654; play";
}
function showTimeline(t, pos) {
  tl.pos = pos;
  const lo = Math.log10(t.ns_min * 0.9), hi = Math.log10(t.ns_max);
  const B = Math.pow(10, lo + (hi - lo) * pos / tl.N);
  const ys = t.methods.map(m => {
    let k = -1;
    while (k + 1 < m.ns.length && m.ns[k + 1] <= B * (1 + 1e-9)) k++;
    return k >= 0 ? m.F[k] : t.x.map(() => null);
  });
  const idx = t.methods.map((_, i) => i + (t.ref ? 1 : 0));
  Plotly.restyle("tl-plot", { y: ys }, idx);
  el("tl-slider").value = String(pos);
  el("tl-readout").textContent = `budget ${B < 1 ? B.toFixed(2) : B < 10 ? B.toFixed(1) : B.toFixed(0)} ns per method`;
}

async function selectMethod(key, trajKey) {
  const meta = state.index.methods.find(m => m.key === key);
  if (!meta) return;
  hideEnsemble(); stopTimeline();
  const switching = state.view !== key;
  state.view = key;
  highlightSidebar();
  el("traj-section").style.display = "";
  if (switching) {
    stopPlay();
    setHeader(meta.name, meta.blurb, meta.fe_how, [], "loading…");
    let m = state.methodCache[key];
    if (!m) {
      try { m = await fetchJSON(`data/${key}.json`); state.methodCache[key] = m; }
      catch (err) { m = null; }
    }
    if (state.view !== key) return;
    state.method = m;
    traj.color = meta.color || "#2a78d6";
    if (!m) {
      setHeader(meta.name, meta.blurb, meta.fe_how, [], "running… no results exported for this method yet");
      el("traj-chips").innerHTML = "";
      renderPanels([]);
      clearViewer("running…");
      return;
    }
    setHeader(m.name || meta.name, meta.blurb, meta.fe_how, m.stats, "");
    renderPanels(m.panels);
    drawSync();
    hideEnsemble();
    renderViewToggles();
    renderChips();
  }
  const m = state.method;
  if (!m) return;
  const want = trajKey || (switching ? m.default_traj || m.trajectories?.[0]?.key : traj.key);
  if (want && (switching || want !== traj.key)) await loadTraj(want);
  highlightSidebar();
}

/* ================================================================ trajectory */

function renderChips() {
  const host = el("traj-chips");
  host.innerHTML = "";
  const list = state.method?.trajectories || [];
  if (!list.length) { host.innerHTML = `<span class="hint">no trajectories exported</span>`; return; }
  const groups = [];
  const byGroup = {};
  list.forEach(t => {
    const g = t.group || "";
    if (!byGroup[g]) { byGroup[g] = []; groups.push(g); }
    byGroup[g].push(t);
  });
  groups.forEach(g => {
    const row = document.createElement("div");
    row.className = "chip-row";
    if (g) row.innerHTML = `<span class="row-label">${esc(g)}</span>`;
    byGroup[g].forEach(t => {
      const b = document.createElement("button");
      b.className = "chip" + (t.key === traj.key ? " active" : "");
      b.textContent = t.label;
      b.dataset.key = t.key;
      b.addEventListener("click", () => loadTraj(t.key));
      row.appendChild(b);
    });
    host.appendChild(row);
  });
}

function clearViewer(msg) {
  stopPlay();
  traj.key = null; traj.coords = null; traj.nFrames = 0;
  if (traj.viewer) { traj.viewer.clear(); traj.model = null; traj.atoms = null; traj.viewer.render(); }
  el("traj-title").textContent = msg || "";
  el("readout").textContent = "";
  el("frame-label").textContent = "";
  try { Plotly.purge("ts-plot"); } catch (e) { /* ignore */ }
}

async function ensureTopology() {
  if (traj.topology) return traj.topology;
  traj.topology = await fetchText("data/topology.pdb");
  return traj.topology;
}

async function loadTraj(key) {
  stopPlay();
  const methodAtLoad = state.view;
  el("traj-title").textContent = "loading…";
  let meta, buf, top;
  try {
    [meta, top] = await Promise.all([fetchJSON(`data/traj/${key}.json`), ensureTopology()]);
    buf = await (await fetch(`data/traj/${meta.bin || key + ".bin"}`, { cache: "no-store" })).arrayBuffer();
  } catch (err) {
    if (state.view === methodAtLoad) clearViewer(`running… trajectory ${key} not exported yet`);
    return;
  }
  if (state.view !== methodAtLoad) return;
  const coords = new Float32Array(buf);
  traj.key = key;
  traj.meta = meta;
  traj.coords = coords;
  traj.nAtoms = meta.n_atoms;
  traj.nFrames = Math.min(meta.n_frames, Math.floor(coords.length / (meta.n_atoms * 3)));

  let keptView = null;
  if (traj.viewer) {
    keptView = traj.model ? traj.viewer.getView() : null;
    traj.viewer.clear();
  } else {
    traj.viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });
  }
  traj.model = traj.viewer.addModel(top, "pdb", { keepH: true });
  traj.atoms = traj.model.selectedAtoms({});
  if (traj.atoms.length !== traj.nAtoms) {
    console.warn(`topology has ${traj.atoms.length} atoms, trajectory ${traj.nAtoms}`);
  }
  traj.hbPairs = hbondPairs(traj.atoms);
  applyStyle();
  el("frame").max = String(Math.max(0, traj.nFrames - 1));
  el("traj-title").textContent = `${meta.title || key} · ${traj.nFrames} frames`;
  document.querySelectorAll("#traj-chips .chip").forEach(c => c.classList.toggle("active", c.dataset.key === key));
  drawTimeSeries();
  showFrame(0);
  if (keptView) traj.viewer.setView(keptView); else resetView();
  highlightSidebar();
}

/* i -> i+4 backbone O ... H-N pairs, by residue order in the topology. */
function hbondPairs(atoms) {
  const residues = [];
  const seen = {};
  atoms.forEach((a, i) => {
    const rk = `${a.chain}:${a.resi}`;
    if (!(rk in seen)) { seen[rk] = residues.length; residues.push({}); }
    residues[seen[rk]][a.atom] = i;
  });
  const pairs = [];
  for (let r = 0; r + 4 < residues.length; r++) {
    const o = residues[r].O, h = residues[r + 4].H;
    if (o !== undefined && h !== undefined) pairs.push([o, h]);
  }
  return pairs;
}

function applyStyle() {
  if (!traj.viewer || !traj.model) return;
  const showH = el("opt-h").checked;
  const cartoon = el("opt-cartoon").checked;
  const v = traj.viewer;
  v.setStyle({}, {});
  const carbon = traj.color;
  const colorfunc = atom => {
    const e = (atom.elem || "").toUpperCase();
    return e === "C" ? carbon : e === "O" ? "#e34948" : e === "N" ? "#3a4fc4" : e === "H" ? "#d8d7d2" : "#999";
  };
  v.setStyle({ elem: "H", invert: true }, { stick: { radius: 0.2, colorfunc }, sphere: { scale: 0.28, colorfunc } });
  if (showH) v.addStyle({ elem: "H" }, { stick: { radius: 0.1, colorfunc }, sphere: { scale: 0.15, colorfunc } });
  if (cartoon) v.addStyle({}, { cartoon: { color: carbon, opacity: 0.9, thickness: 0.4 } });
  if (traj.coords) showFrame(traj.frame);
}

function writeFrame(f) {
  const n = traj.nAtoms, base = f * n * 3, c = traj.coords;
  const m = Math.min(n, traj.atoms.length);
  for (let a = 0; a < m; a++) {
    const at = traj.atoms[a];
    at.x = c[base + 3 * a]; at.y = c[base + 3 * a + 1]; at.z = c[base + 3 * a + 2];
  }
  traj.model.molObj = null;   // 3Dmol caches built geometry; drop it so the new coordinates draw
}

function drawHbonds(f) {
  const v = traj.viewer;
  v.removeAllShapes();
  if (!el("opt-hb").checked) return;
  const c = traj.coords, base = f * traj.nAtoms * 3;
  traj.hbPairs.forEach(([o, h]) => {
    const p = { x: c[base + 3 * o], y: c[base + 3 * o + 1], z: c[base + 3 * o + 2] };
    const q = { x: c[base + 3 * h], y: c[base + 3 * h + 1], z: c[base + 3 * h + 2] };
    const d = Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z);
    if (d < 2.6) {
      v.addCylinder({ start: p, end: q, radius: 0.07, color: "#c98500", dashed: true, dashLength: 0.25, gapLength: 0.18, fromCap: 1, toCap: 1 });
    }
  });
}

function showFrame(f) {
  if (!traj.viewer || !traj.coords || !traj.nFrames) return;
  f = Math.max(0, Math.min(traj.nFrames - 1, f));
  traj.frame = f;
  writeFrame(f);
  drawHbonds(f);
  traj.viewer.render();
  el("frame").value = String(f);
  const m = traj.meta;
  const t = m.t_ns?.[f], d = m.d_ee?.[f], h = m.n_hb?.[f];
  el("readout").textContent = `${f + 1}/${traj.nFrames}` +
    (t !== undefined ? ` · ${fmtNs(t)}` : "") +
    (d !== undefined ? ` · d_ee ${Number(d).toFixed(1)} Å` : "") +
    (h !== undefined ? ` · ${h} H-bonds` : "");
  el("frame-label").textContent = m.frame_label?.[f] || "";
  moveCursor(t);
  updateSync(t, d);
}

/* Live panels synced to the movie: each is a curve over d_ee that is replaced by the snapshot
   whose time is the latest one <= the frame's time (metad bias V(s,t); FE estimate so far). */
function syncList() {
  const m = state.method;
  if (!m) return [];
  return m.syncs || (m.sync ? [m.sync] : []);
}
function drawSync() {
  const list = syncList();
  const box = el("sync-box");
  box.querySelectorAll(".plot").forEach(d => { try { Plotly.purge(d); } catch (e) { /* ignore */ } });
  box.innerHTML = "";
  box.style.display = list.length ? "" : "none";
  state.syncIdx = list.map(() => -2);
  list.forEach((s, i) => {
    const h = document.createElement("h3");
    h.innerHTML = `${esc(s.title)} <span class="hint">${esc(s.hint || "")}</span>`;
    const div = document.createElement("div");
    div.className = "plot sync-plot"; div.id = `sync-plot-${i}`;
    box.appendChild(h); box.appendChild(div);
    const pmf = s.kind === "pmf";
    const last = s.V[s.V.length - 1] || [];
    const vmax = pmf ? (s.yrange || [0, 30])[1] : (Math.max(...last.filter(v => v !== null)) * 1.1 || 1);
    const traces = [
      pmf ? { type: "scatter", mode: "lines", x: s.x, y: s.V[0], name: "estimate so far", line: { color: s.color, width: 2.5 } }
          : { type: "scatter", mode: "lines", x: s.x, y: s.V[0], fill: "tozeroy", name: "bias V(s,t)",
              line: { color: s.color, width: 2 }, fillcolor: s.color + "33" },
      { type: "scatter", mode: "markers", x: [null], y: [null], name: "walker", marker: { size: 11, color: "#0b0b0b" } },
    ];
    if (pmf && s.ref) traces.push({ type: "scatter", mode: "lines", x: s.x, y: s.ref, name: "reference", line: { color: TEXT2, width: 2, dash: "dash" } });
    Plotly.react(div, traces, panelLayout({
      margin: { l: 60, r: 20, t: 10, b: 45 }, showlegend: false,
      xaxis: { title: { text: "d_ee (Å)" }, range: s.xrange || [3, 36] },
      yaxis: { title: { text: s.ylabel || "V (kcal/mol)" }, range: [0, vmax] },
      annotations: [{ xref: "paper", yref: "paper", x: 0.99, y: 0.97, xanchor: "right", showarrow: false, text: "", font: { color: TEXT2, size: 13 } }],
    }), PLOT_CONFIG);
  });
}
function updateSync(t, d) {
  const list = syncList();
  if (!list.length || t === undefined) return;
  list.forEach((s, i) => {
    const div = el(`sync-plot-${i}`);
    if (!div || !div.data) return;
    let k = -1;
    while (k + 1 < s.t_ns.length && s.t_ns[k + 1] <= t + 1e-9) k++;
    const V = k >= 0 ? s.V[k] : s.x.map(() => null);
    let y = null;
    if (d !== undefined && k >= 0) {
      let j = 0; while (j + 1 < s.x.length && s.x[j + 1] <= d) j++;
      y = V[j];
    }
    if (k !== state.syncIdx[i]) {
      Plotly.restyle(div, { y: [V] }, [0]);
      state.syncIdx[i] = k;
      Plotly.relayout(div, { "annotations[0].text": k >= 0 ? `as of ${fmtNs(s.t_ns[k])}` : "no estimate yet" });
    }
    Plotly.restyle(div, { x: [[d ?? null]], y: [[y]] }, [1]);
  });
}
function fmtNs(t) {
  return t < 1 ? `${(t * 1000).toFixed(0)} ps` : `${Number(t).toFixed(t < 10 ? 2 : 1)} ns`;
}

function drawTimeSeries() {
  const m = traj.meta;
  if (!m || !m.t_ns) { try { Plotly.purge("ts-plot"); } catch (e) { /* ignore */ } return; }
  const color = traj.color;
  const data = [
    { x: m.t_ns, y: m.d_ee, type: "scattergl", mode: "lines", line: { color, width: 2 }, name: "d_ee", xaxis: "x", yaxis: "y",
      hovertemplate: "%{x:.3f} ns<br>d_ee %{y:.1f} Å<extra></extra>" },
    { x: m.t_ns, y: m.n_hb, type: "scattergl", mode: "lines", line: { color: TEXT2, width: 1.5, shape: "hv" }, name: "H-bonds",
      xaxis: "x", yaxis: "y2", hovertemplate: "%{x:.3f} ns<br>%{y} H-bonds<extra></extra>" },
    { x: [m.t_ns[0]], y: [m.d_ee[0]], type: "scatter", mode: "markers", xaxis: "x", yaxis: "y", hoverinfo: "skip",
      marker: { size: 12, color: "#fff", line: { color: TEXT, width: 2.5 } }, showlegend: false },
  ];
  const layout = panelLayout({
    showlegend: false,
    margin: { l: 64, r: 16, t: 10, b: 50 },
    grid: { rows: 2, columns: 1, roworder: "top to bottom" },
    xaxis: { title: { text: "time (ns)" }, anchor: "y2" },
    yaxis: { title: { text: "d_ee (Å)" }, domain: [0.38, 1] },
    yaxis2: { title: { text: "i→i+4 H-bonds" }, domain: [0, 0.28], rangemode: "tozero", dtick: 2 },
    shapes: [cursorShape(m.t_ns[0])],
    hovermode: "x",
  });
  Plotly.react("ts-plot", data, layout, PLOT_CONFIG).then(div => {
    div.removeAllListeners?.("plotly_click");
    div.on("plotly_click", ev => {
      const x = ev.points?.[0]?.x;
      if (x === undefined) return;
      stopPlay();
      showFrame(nearestIndex(m.t_ns, x));
    });
  });
}
function cursorShape(t) {
  return { type: "line", xref: "x", yref: "paper", x0: t, x1: t, y0: 0, y1: 1, line: { color: TEXT, width: 1.5, dash: "dot" } };
}
let cursorPending = null;
function moveCursor(t) {
  if (t === undefined) return;
  const div = el("ts-plot");
  if (!div.data) return;
  cursorPending = t;
  if (moveCursor.busy) return;
  moveCursor.busy = true;
  requestAnimationFrame(() => {
    const tt = cursorPending, f = traj.frame, m = traj.meta;
    Plotly.relayout(div, { shapes: [cursorShape(tt)] });
    Plotly.restyle(div, { x: [[tt]], y: [[m.d_ee?.[f]]] }, [2]);
    moveCursor.busy = false;
  });
}
function nearestIndex(arr, x) {
  let best = 0, bd = Infinity;
  for (let i = 0; i < arr.length; i++) { const d = Math.abs(arr[i] - x); if (d < bd) { bd = d; best = i; } }
  return best;
}

function togglePlay() {
  if (traj.timer) { stopPlay(); return; }
  if (!traj.nFrames) return;
  traj.timer = setInterval(() => showFrame((traj.frame + 1) % traj.nFrames), Number(el("speed").value));
  el("play").innerHTML = "&#10073;&#10073; pause";
}
function stopPlay() {
  if (traj.timer) { clearInterval(traj.timer); traj.timer = null; }
  el("play").innerHTML = "&#9654; play";
}
function resetView() {
  if (!traj.viewer) return;
  traj.viewer.zoomTo();
  traj.viewer.zoom(0.9);
  traj.viewer.render();
}

/* ================================================================ wiring */

el("nav-comparison").addEventListener("click", showComparison);
el("nav-intro").addEventListener("click", showIntro);
el("nav-estimators").addEventListener("click", showEstimators);
el("ens-play").addEventListener("click", toggleEns);
el("ens-frame").addEventListener("input", e => { stopEns(); showEnsFrame(Number(e.target.value)); });
el("play").addEventListener("click", togglePlay);
el("frame").addEventListener("input", e => { stopPlay(); showFrame(Number(e.target.value)); });
el("speed").addEventListener("input", () => { if (traj.timer) { stopPlay(); togglePlay(); } });
el("reset-view").addEventListener("click", resetView);
["opt-h", "opt-cartoon"].forEach(id => el(id).addEventListener("change", applyStyle));
el("opt-hb").addEventListener("change", () => showFrame(traj.frame));
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox") return;
  if (el("ens-section").style.display !== "none") {
    if (e.code === "Space") { e.preventDefault(); toggleEns(); }
    else if (e.code === "ArrowRight") { e.preventDefault(); stopEns(); showEnsFrame(ens.frame + 1); }
    else if (e.code === "ArrowLeft") { e.preventDefault(); stopEns(); showEnsFrame(ens.frame - 1); }
    return;
  }
  if (["comparison", "intro", "estimators"].includes(state.view) || !traj.nFrames) return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  else if (e.code === "ArrowRight") { e.preventDefault(); stopPlay(); showFrame(traj.frame + 1); }
  else if (e.code === "ArrowLeft") { e.preventDefault(); stopPlay(); showFrame(traj.frame - 1); }
});
window.addEventListener("resize", () => { if (traj.viewer) traj.viewer.resize(); });

(async function init() {
  try {
    state.index = await fetchJSON("data/index.json");
  } catch (err) {
    el("title").textContent = "No data yet";
    el("note").textContent = `Could not load data/index.json (${err}). Run analysis/build_data.py, then serve this directory over HTTP (file:// is blocked by fetch).`;
    el("traj-section").style.display = "none";
    return;
  }
  renderSidebar();
  const hash = location.hash.replace("#", "");
  if (hash && state.index.methods.some(m => m.key === hash)) selectMethod(hash);
  else if (hash === "comparison") showComparison();
  else if (hash === "estimators") showEstimators();
  else showIntro();
})();
