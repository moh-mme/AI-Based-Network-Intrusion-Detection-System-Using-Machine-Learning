"""Rotating attack globe for the live monitor, built on globe.gl.

The JavaScript lives in attack_globe.js; globe.gl, topojson-client and the
world map are served from dashboard/static/vendor, so it works offline.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from common import attack_payload

_component = st.components.v2.component(
    "attack_globe",
    js=(Path(__file__).with_suffix(".js")).read_text(encoding="utf-8"),
    css=".nids-globe { width: 100%; height: 100%; cursor: grab; }",
    isolate_styles=False,
)


def attack_globe(tick: int, attacks: dict[str, int], height: int = 440, key: str = "attack_globe") -> None:
    """Draw this tick's detections (attack type -> flow count) as arcs on the globe."""
    _component(key=key, height=height, data=attack_payload(tick, attacks))
