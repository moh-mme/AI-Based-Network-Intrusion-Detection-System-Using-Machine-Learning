"""Flat world map for the live monitor: a dotted map with each detected attack
drawn as a curved arc that grows towards the protected network.

The look follows a React "World Map" component (dotted-map + framer-motion):
diagonal dot grid, curved arcs that draw in, pulsing end points. It is
re-implemented here without React. The dots come from the same world map as the
globe and use the arcs' projection, so land and arcs line up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib.path import Path as Polygon

from common import attack_payload

STATIC = Path(__file__).parent / "static"
TOPOLOGY = STATIC / "vendor" / "countries-110m.json"
WORLD_MAP = STATIC / "world-map.json"  # generated on first use
# Equirectangular view box cropped to the latitudes with land (Antarctica is left
# out), so the map fills its frame instead of leaving empty polar bands.
LAT_TOP, LAT_BOTTOM = 84, -58
WIDTH = 800
HEIGHT = round(WIDTH * (LAT_TOP - LAT_BOTTOM) / 360)
STEP = 4  # distance between dot rows, in view-box units


def _rings(topology: dict) -> list[list[np.ndarray]]:
    """Each country's polygons as rings of (lng, lat), decoded from TopoJSON."""
    (sx, sy), (tx, ty) = topology["transform"]["scale"], topology["transform"]["translate"]
    arcs = [np.cumsum(np.asarray(arc, dtype=float), axis=0) * (sx, sy) + (tx, ty) for arc in topology["arcs"]]

    def ring(indexes):
        parts = [arcs[i] if i >= 0 else arcs[~i][::-1] for i in indexes]
        return np.concatenate([parts[0]] + [p[1:] for p in parts[1:]])

    countries = []
    for geometry in topology["objects"]["countries"]["geometries"]:
        polygons = {"Polygon": [geometry.get("arcs", [])], "MultiPolygon": geometry.get("arcs", [])}
        countries.append([[ring(r) for r in polygon]
                          for polygon in polygons.get(geometry["type"], []) if polygon])
    return countries


def build_world_map(topology_path: Path = TOPOLOGY, out_path: Path = WORLD_MAP) -> dict:
    """Land dots on a diagonal grid, plus one centre per country for arc origins.

    Polygons crossing the antimeridian (eastern Russia, Fiji) and Antarctica are
    skipped: in flat longitude/latitude they would wrap around the whole map.
    """
    countries = _rings(json.loads(topology_path.read_text(encoding="utf-8")))
    points = np.array([(col * STEP + (STEP / 2 if row % 2 else 0) + STEP / 4, row * STEP + STEP / 2)
                       for row in range(int(HEIGHT / STEP)) for col in range(int(WIDTH / STEP))])
    lnglat = np.column_stack([points[:, 0] * 360 / WIDTH - 180,
                              LAT_TOP - points[:, 1] * (LAT_TOP - LAT_BOTTOM) / HEIGHT])

    land = np.zeros(len(points), dtype=bool)
    origins = []
    for polygons in countries:
        drawable = [rings for rings in polygons
                    if np.ptp(rings[0][:, 0]) <= 180 and rings[0][:, 1].min() > -60]
        for outer, *holes in drawable:
            inside = Polygon(outer).contains_points(lnglat)
            for hole in holes:
                inside &= ~Polygon(hole).contains_points(lnglat)
            land |= inside
        if drawable:
            largest = max((rings[0] for rings in drawable), key=len)
            origins.append([round(float(largest[:, 1].mean()), 2), round(float(largest[:, 0].mean()), 2)])

    world = {"width": WIDTH, "height": HEIGHT, "lat_top": LAT_TOP, "lat_bottom": LAT_BOTTOM,
             "radius": round(STEP * 0.28, 2), "dots": np.round(points[land], 1).tolist(), "origins": origins}
    out_path.write_text(json.dumps(world, separators=(",", ":")), encoding="utf-8")
    return world


_component = st.components.v2.component(
    "attack_map",
    js=(Path(__file__).with_suffix(".js")).read_text(encoding="utf-8"),
    css="""
.nids-map { width: 100%; }
.nids-map svg { width: 100%; height: auto; display: block; overflow: visible; }
.nids-map path.arc { transition: stroke-dashoffset 1s ease-out, opacity 0.6s ease-in; }
.nids-map .fading { opacity: 0; transition: opacity 0.6s ease-in; }
.nids-map.fullscreen { position: fixed; inset: 0; z-index: 0; background: #0A0D12; }
.nids-map.fullscreen svg { height: 100%; }
""",
    isolate_styles=False,
)


def attack_map(tick: int, attacks: dict[str, int], key: str = "attack_map",
               sequence: list[dict[str, int]] | None = None, fullscreen: bool = False) -> None:
    """Draw this tick's detections (attack type -> flow count) as arcs on the world map.

    With `sequence` (one {attack type: flows} per second) the map plays it on a loop
    by itself instead of following `tick`. `fullscreen` fills the window behind the page.
    """
    # Rebuilt whenever this generator is newer than its output.
    if not WORLD_MAP.exists() or WORLD_MAP.stat().st_mtime < Path(__file__).stat().st_mtime:
        build_world_map()
    data = attack_payload(tick, attacks)
    data["fullscreen"] = fullscreen
    if sequence:
        data["sequence"] = [attack_payload(0, step)["events"] for step in sequence]
    _component(key=key, height="content", data=data)
