// Flat world map for the live monitor. Each detected attack becomes a curved
// arc, coloured by attack type, that draws itself towards the protected network.
//
// Streamlit calls this function again on every rerun with fresh `data`, so the
// map is built once, kept on its host element, and only updated after that.

const WORLD = "app/static/world-map.json";
const SVG = "http://www.w3.org/2000/svg";
const ARC_LIFETIME_MS = 3600;
const FADE_MS = 600;
const MAX_ARCS_PER_TYPE = 5;   // one arc per ~10 flows, capped so bursts stay readable
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function node(tag, attributes = {}, parent = null) {
  const element = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  if (parent) parent.appendChild(element);
  return element;
}

// A dot that keeps growing and fading: the original component's "ping".
function pulse(parent, x, y, color) {
  node("circle", { cx: x, cy: y, r: 2, fill: color }, parent);
  if (reduceMotion) return;
  const ring = node("circle", { cx: x, cy: y, r: 2, fill: color, opacity: 0.5 }, parent);
  node("animate", { attributeName: "r", from: 2, to: 8, dur: "1.5s", repeatCount: "indefinite" }, ring);
  node("animate", { attributeName: "opacity", from: 0.5, to: 0, dur: "1.5s", repeatCount: "indefinite" }, ring);
}

async function createMap(host, data) {
  const world = await (await fetch(WORLD)).json();
  const { width: W, height: H, lat_top: top, lat_bottom: bottom } = world;
  const project = ({ lat, lng }) => ({ x: (lng + 180) * (W / 360), y: (top - lat) * (H / (top - bottom)) });

  const svg = node("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "World map with detected attacks drawn as arcs to the protected network" });
  const defs = node("defs", {}, svg);
  // Fade the land out at the top and bottom edges, as in the original design.
  const fade = node("linearGradient", { id: "nids-map-fade", x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
  [[0, 0], [0.1, 1], [0.9, 1], [1, 0]].forEach(([offset, opacity]) =>
    node("stop", { offset, "stop-color": "white", "stop-opacity": opacity }, fade));
  const mask = node("mask", { id: "nids-map-mask" }, defs);
  node("rect", { width: W, height: H, fill: "url(#nids-map-fade)" }, mask);

  const land = node("g", { fill: data.land, "fill-opacity": 0.75, mask: "url(#nids-map-mask)" }, svg);
  for (const [x, y] of world.dots) node("circle", { cx: x, cy: y, r: world.radius }, land);
  const arcs = node("g", {}, svg);
  const target = project(data.target);
  pulse(node("g", {}, svg), target.x, target.y, data.accent);
  host.appendChild(svg);

  const state = { svg, defs, arcs, project, target, origins: world.origins, items: [], lastTick: -1,
                  gradients: new Set() };

  // Fade out and remove old arcs; stop once Streamlit removes the element.
  const timer = setInterval(() => {
    if (!host.isConnected) return clearInterval(timer);
    const now = performance.now();
    state.items = state.items.filter((item) => {
      const age = now - item.born;
      if (age > ARC_LIFETIME_MS) { item.group.remove(); return false; }
      if (age > ARC_LIFETIME_MS - FADE_MS) item.group.classList.add("fading");
      return true;
    });
  }, 200);
  return state;
}

// Transparent at both ends, solid in the middle: one gradient per attack colour.
function gradientFor(state, color) {
  const id = `nids-arc-${color.slice(1)}`;
  if (!state.gradients.has(id)) {
    const gradient = node("linearGradient", { id, x1: "0%", y1: "0%", x2: "100%", y2: "0%" }, state.defs);
    [[0, 0], [0.05, 1], [0.95, 1], [1, 0]].forEach(([offset, opacity]) =>
      node("stop", { offset, "stop-color": color, "stop-opacity": opacity }, gradient));
    state.gradients.add(id);
  }
  return `url(#${id})`;
}

function addArc(state, event, now) {
  const origin = state.origins[Math.floor(Math.random() * state.origins.length)];
  const start = state.project({ lat: origin[0], lng: origin[1] });
  const end = state.target;
  const group = node("g", {}, state.arcs);
  const path = node("path", {
    class: "arc", fill: "none", stroke: gradientFor(state, event.color), "stroke-width": 1.4,
    d: `M ${start.x} ${start.y} Q ${(start.x + end.x) / 2} ${Math.min(start.y, end.y) - 50} ${end.x} ${end.y}`,
  }, group);
  node("title", {}, path).textContent = `${event.category}: ${event.count} flows detected`;
  pulse(group, start.x, start.y, event.color);
  if (!reduceMotion) {
    // Draw the arc from its origin: hide it behind a dash offset, then release it.
    const length = path.getTotalLength();
    path.style.strokeDasharray = `${length}`;
    path.style.strokeDashoffset = `${length}`;
    path.getBoundingClientRect();  // commit the hidden state before animating
    path.style.strokeDashoffset = "0";
  }
  state.items.push({ group, born: now });
}

function addEvents(state, events) {
  const now = performance.now();
  for (const event of events) {
    const count = Math.min(MAX_ARCS_PER_TYPE, Math.ceil(event.count / 10));
    for (let i = 0; i < count; i++) addArc(state, event, now);
  }
}

function applyData(state, data) {
  if (data.tick < state.lastTick) {  // the monitor was reset
    state.items.forEach((item) => item.group.remove());
    state.items = [];
  } else if (data.tick > state.lastTick) {
    addEvents(state, data.events);
  }
  state.lastTick = data.tick;
}

// Autoplay: step through a recorded sequence of per-second detections on a loop,
// without waiting for Streamlit reruns (used on the overview page).
function autoplay(host, state, sequence) {
  let step = 0;
  const timer = setInterval(() => {
    if (!host.isConnected) return clearInterval(timer);
    addEvents(state, sequence[step++ % sequence.length]);
  }, 1000);
}

export default function (component) {
  const { data, parentElement } = component;
  let host = parentElement.querySelector(".nids-map");
  if (!host) {
    host = document.createElement("div");
    host.className = "nids-map";
    parentElement.appendChild(host);
  }
  host.latestData = data;
  if (!host.mapState) {
    host.mapState = createMap(host, data).then((state) => {
      if (data.sequence?.length) autoplay(host, state, data.sequence);
      else applyData(state, host.latestData);
      return state;
    });
  } else if (!data.sequence?.length) {
    host.mapState.then((state) => applyData(state, host.latestData));
  }
}
