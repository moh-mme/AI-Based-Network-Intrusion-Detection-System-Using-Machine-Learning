// Rotating globe for the live monitor. Each detected attack becomes an arc
// that travels to the protected network, coloured by attack type.
//
// Streamlit calls this function again on every rerun with fresh `data`, so the
// globe is created once, kept on its host element, and only updated after that.

const ASSETS = "app/static/vendor/";
const ARC_LIFETIME_MS = 3600;
const RING_LIFETIME_MS = 1800;
const MAX_ARCS_PER_TYPE = 5;   // one arc per ~10 flows, capped so bursts stay readable
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-nids-src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded) resolve();
      else existing.addEventListener("load", () => resolve());
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.dataset.nidsSrc = src;
    script.onload = () => { script.dataset.loaded = "1"; resolve(); };
    script.onerror = () => reject(new Error(`Could not load ${src}`));
    document.head.appendChild(script);
  });
}

function polygonsOf(feature) {
  const g = feature.geometry;
  return !g ? [] : g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : [];
}

// h3, which draws the dotted land, throws on polygons with no area. The 110m map
// has one (a North Korean polygon whose four points coincide), so drop those.
function drawableLand(features) {
  return features.flatMap((feature) => {
    const kept = polygonsOf(feature).filter(([outer]) => {
      const lngs = outer.map((p) => p[0]);
      const lats = outer.map((p) => p[1]);
      return Math.max(...lngs) > Math.min(...lngs) && Math.max(...lats) > Math.min(...lats);
    });
    return kept.length ? [{ ...feature, geometry: { type: "MultiPolygon", coordinates: kept } }] : [];
  });
}

// Rough centre of each country's largest polygon: candidate arc origins.
function countryCentres(features) {
  const centres = [];
  for (const feature of features) {
    const polygons = polygonsOf(feature);
    if (!polygons.length) continue;
    const ring = polygons.reduce((a, b) => (b[0].length > a[0].length ? b : a))[0];
    const lngs = ring.map((p) => p[0]);
    if (Math.max(...lngs) - Math.min(...lngs) > 180) continue;  // straddles the antimeridian
    const lat = ring.reduce((sum, p) => sum + p[1], 0) / ring.length;
    if (lat < -60) continue;  // Antarctica
    centres.push({ lat, lng: lngs.reduce((sum, x) => sum + x, 0) / ring.length });
  }
  return centres;
}

function withAlpha(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

async function createGlobe(host, data) {
  await loadScript(ASSETS + "globe.gl.min.js");
  await loadScript(ASSETS + "topojson-client.min.js");
  const world = await (await fetch(ASSETS + "countries-110m.json")).json();
  const countries = drawableLand(topojson.feature(world, world.objects.countries).features);

  const globe = new Globe(host, { animateIn: false })
    .width(host.clientWidth)
    .height(host.clientHeight)
    .backgroundColor("rgba(0, 0, 0, 0)")
    .showAtmosphere(true)
    .atmosphereColor(data.accent)
    .atmosphereAltitude(0.14)
    .hexPolygonsData(countries)
    .hexPolygonResolution(3)
    .hexPolygonMargin(0.35)
    .hexPolygonUseDots(true)
    .hexPolygonColor(() => data.land)
    .pointsData([data.target])
    .pointColor(() => data.accent)
    .pointAltitude(0.02)
    .pointRadius(0.55)
    .pointLabel(() => data.target.label)
    .arcColor("color")
    .arcLabel("label")
    .arcStroke(0.85)  // thick enough to read on a projector
    .arcAltitudeAutoScale(0.45)
    .arcDashLength(0.45)
    .arcDashGap(1.6)
    .arcDashInitialGap(1)
    .arcDashAnimateTime(reduceMotion ? 0 : 1400)
    .arcsTransitionDuration(0)
    .ringColor(() => (t) => `rgba(255, 107, 111, ${1 - t})`)
    .ringMaxRadius(5)
    .ringPropagationSpeed(3)
    .ringRepeatPeriod(reduceMotion ? 0 : 700);
  globe.globeMaterial().color.set(data.ocean);
  globe.pointOfView({ lat: 25, lng: data.target.lng + 25, altitude: 2.0 });

  const controls = globe.controls();
  controls.enableZoom = false;  // keep the page's scroll wheel for the page
  controls.autoRotate = !reduceMotion;
  controls.autoRotateSpeed = 0.45;

  const state = { globe, origins: countryCentres(countries), arcs: [], rings: [], lastTick: -1 };

  const resize = new ResizeObserver(() => globe.width(host.clientWidth).height(host.clientHeight));
  resize.observe(host);
  // Expire old arcs, and release WebGL resources once Streamlit removes the element.
  const timer = setInterval(() => {
    if (!host.isConnected) {
      clearInterval(timer);
      resize.disconnect();
      globe._destructor();
      return;
    }
    const now = performance.now();
    const arcs = state.arcs.filter((a) => now - a.born < ARC_LIFETIME_MS);
    const rings = state.rings.filter((r) => now - r.born < RING_LIFETIME_MS);
    if (arcs.length !== state.arcs.length) globe.arcsData((state.arcs = arcs));
    if (rings.length !== state.rings.length) globe.ringsData((state.rings = rings));
  }, 250);
  return state;
}

function applyData(state, data) {
  if (data.tick < state.lastTick) {  // the monitor was reset
    state.arcs = [];
    state.rings = [];
  } else if (data.tick > state.lastTick && data.events.length) {
    const now = performance.now();
    const { lat, lng } = data.target;
    for (const event of data.events) {
      const count = Math.min(MAX_ARCS_PER_TYPE, Math.ceil(event.count / 10));
      for (let i = 0; i < count; i++) {
        const origin = state.origins[Math.floor(Math.random() * state.origins.length)];
        state.arcs.push({
          startLat: origin.lat, startLng: origin.lng, endLat: lat, endLng: lng,
          color: [withAlpha(event.color, 0.1), event.color],
          label: `${event.category}: ${event.count} flows detected`,
          born: now,
        });
      }
    }
    state.rings.push({ lat, lng, born: now });
  }
  state.lastTick = data.tick;
  state.globe.arcsData(state.arcs).ringsData(state.rings);
}

export default function (component) {
  const { data, parentElement } = component;
  let host = parentElement.querySelector(".nids-globe");
  if (!host) {
    host = document.createElement("div");
    host.className = "nids-globe";
    host.setAttribute("role", "img");
    host.setAttribute("aria-label", "Rotating globe with detected attacks drawn as arcs to the protected network");
    parentElement.appendChild(host);
  }
  host.latestData = data;
  if (!host.globeState) {
    host.globeState = createGlobe(host, data).then((state) => {
      applyData(state, host.latestData);
      return state;
    });
  } else {
    host.globeState.then((state) => applyData(state, host.latestData));
  }
}
