"""Shared loaders, theme tokens and helpers for the dashboard pages."""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from nids.config import TASKS, load_config  # noqa: E402
from nids.data import BENIGN  # noqa: E402
from nids.predict import IntrusionDetector  # noqa: E402

TASK_LABELS = {"binary": "Benign vs attack", "multiclass": "Attack type (8 classes)"}

# Theme tokens (mirrored in .streamlit/config.toml).
INK = "#E6EDF5"
MUTED = "#93A4BC"
SURFACE = "#141922"
BORDER = "#232A36"
ACCENT = "#38C6E8"

# Traffic categories. Benign stays a quiet steel blue so attacks stand out;
# attack hues are spread out (and vary in lightness) so neighbours stay
# distinguishable, including for colour-blind viewers.
CATEGORY_COLORS = {
    BENIGN: "#5F86AD",
    "Attack": "#E5484D",
    "PortScan": "#F2C14E",
    "BruteForce": "#F08A4B",
    "DoS": "#E5484D",
    "DDoS": "#D23C7A",
    "WebAttack": "#EDE6D6",
    "Botnet": "#A970FF",
    "Infiltration": "#B6E35A",
}
# Models and metrics use a separate cool palette, so no chart reuses a traffic
# colour with a different meaning. Dash styles keep lines apart without colour.
SERIES_COLORS = ["#38C6E8", "#3FB59A", "#8A99AD", "#9FE3F5", "#2B7A9B", "#C3CFDD"]
SERIES_DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
# Sequential scale for heatmaps such as confusion matrices.
HEAT_SCALE = [[0.0, SURFACE], [0.5, "#2B7A9B"], [1.0, ACCENT]]

# CIC-IDS2017 was recorded on the Canadian Institute for Cybersecurity testbed
# at the University of New Brunswick, Fredericton: the network the attacks target.
TARGET = {"lat": 45.96, "lng": -66.64, "label": "Protected network (CIC-IDS2017 testbed)"}
LAND = "#4B6A8C"

# Columns worth showing next to a prediction, when the input file has them.
CONTEXT_COLUMNS = [
    "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp",
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s",
]

# Lucide icons (ISC licence), inlined so they inherit the surrounding text colour.
_SHIELD = ('M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 '
           '4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z')
ICONS = {
    "brand": [_SHIELD, "M8 12.5h2l1.2-2.5 1.8 4 1.2-2.5H16"],
    "clear": [_SHIELD, "m9 12 2 2 4-4"],
    "alert": [_SHIELD, "M12 8v4", "M12 16h.01"],
}

CSS = f"""
<style>
[data-testid="stMetricValue"] {{
    font-family: 'Barlow Semi Condensed', Barlow, sans-serif;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.01em;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-weight: 500; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED}; opacity: 1; }}
/* White text on the cyan accent fails contrast (2:1); navy on cyan passes (8:1). */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] * {{ color: #0B1626 !important; }}
/* Selected-option chips stay quiet so cyan is kept for actions. */
[data-testid="stMultiSelectTagsContainer"] > span > span {{
    background-color: #22344D !important; border: 1px solid {BORDER};
}}
[data-testid="stMultiSelectTagsContainer"] > span > span * {{ color: {INK} !important; }}
.nids-brand {{ display: flex; align-items: center; gap: 0.7rem; padding: 0.25rem 0 0.9rem; }}
.nids-brand img, .threat-band img {{ flex: none; }}
.nids-brand b {{
    display: block; font-family: 'Barlow Semi Condensed', Barlow, sans-serif;
    font-size: 1.35rem; font-weight: 700; color: {INK}; line-height: 1.1;
}}
.nids-brand span {{ font-size: 0.85rem; color: {MUTED}; }}
.nids-status {{ font-size: 0.85rem; color: {MUTED}; line-height: 1.5; }}
.nids-status b {{ color: {INK}; font-weight: 500; }}
.threat-band {{
    display: flex; align-items: center; gap: 0.8rem; padding: 0.85rem 1.1rem;
    border: 1px solid; border-radius: 6px;
    font-family: 'Barlow Semi Condensed', Barlow, sans-serif; font-size: 1.35rem; font-weight: 600;
    font-variant-numeric: tabular-nums;
}}
.threat-clear {{ background: rgba(95, 134, 173, 0.12); border-color: {BORDER}; color: #CFE0F2; }}
.threat-alert {{ background: rgba(229, 72, 77, 0.16); border-color: #E5484D; color: #FFD9DA; }}
.threat-band small {{ font-family: Barlow, sans-serif; font-size: 0.95rem; font-weight: 400; color: {MUTED}; }}
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


def icon(name: str, size: int = 24, color: str = ACCENT) -> str:
    """A decorative icon as an <img>: st.html strips inline <svg> elements."""
    paths = "".join(f'<path d="{d}"/>' for d in ICONS[name])
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')
    data = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" alt="">'


@st.cache_resource
def get_config():
    return load_config()


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def _read_json(path: str, mtime: float) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if mtime else None


@st.cache_data
def _read_parquet(path: str, mtime: float) -> pd.DataFrame | None:
    return pd.read_parquet(path) if mtime else None


# The file's modification time is part of each cache key, so results refresh
# automatically after `python -m nids.prepare` or `python -m nids.train`.
def read_json(path: Path) -> dict | None:
    return _read_json(str(path), _mtime(path))


def get_profile() -> dict | None:
    path = get_config().paths.profile
    return _read_json(str(path), _mtime(path))


def get_metrics(task: str) -> dict | None:
    path = get_config().paths.task_reports(task) / "metrics.json"
    return _read_json(str(path), _mtime(path))


def get_metadata(task: str) -> dict | None:
    path = get_config().paths.task_models(task) / "metadata.json"
    return _read_json(str(path), _mtime(path))


def get_explorer_sample() -> pd.DataFrame | None:
    path = get_config().paths.explorer_sample
    return _read_parquet(str(path), _mtime(path))


def get_demo_stream() -> pd.DataFrame | None:
    path = get_config().paths.demo_stream
    return _read_parquet(str(path), _mtime(path))


def trained_tasks() -> list[str]:
    return [task for task in TASKS if get_metadata(task)]


@st.cache_resource(max_entries=6)
def _detector(task: str, model: str, mtime: float) -> IntrusionDetector:
    return IntrusionDetector(task, model, get_config())


def get_detector(task: str, model: str) -> IntrusionDetector:
    path = get_config().paths.task_models(task) / "metadata.json"
    return _detector(task, model, _mtime(path))


def compact(n: float) -> str:
    """2497980 -> '2.5M', so large counts fit in narrow metric cards."""
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= size:
            return f"{n / size:.2f}".rstrip("0").rstrip(".") + suffix
    return f"{n:,}"


def when(timestamp: str) -> str:
    """'2026-10-31T23:57:20' -> '31 Oct 2026, 23:57'."""
    return datetime.fromisoformat(timestamp).strftime("%d %b %Y, %H:%M")


def attack_payload(tick: int, attacks: dict[str, int]) -> dict:
    """Data for the attack globe and world map: this tick's detections by attack type.

    `tick` increases once per classified batch. The views add arcs only when it
    changes, and clear them when it goes back to zero after a reset.
    """
    return {
        "tick": tick,
        "events": [{"category": category, "count": count, "color": CATEGORY_COLORS.get(category, MUTED)}
                   for category, count in attacks.items()],
        "target": TARGET,
        "accent": ACCENT,
        "land": LAND,
        "ocean": "#13253B",
    }


def color_map(classes) -> dict[str, str]:
    return {c: CATEGORY_COLORS.get(c, MUTED) for c in classes}


def synthetic_warning() -> None:
    sources = {(get_profile() or {}).get("data_source")}
    sources |= {(get_metadata(task) or {}).get("data_source") for task in TASKS}
    if "synthetic" in sources:
        st.warning(
            "**Synthetic data.** These results come from generated test data, not CIC-IDS2017. "
            "They only show that the pipeline works. Put the real dataset in `data/raw/` and "
            "rerun `python -m nids.prepare` and `python -m nids.train`.",
            icon=":material/warning:",
        )


def require(available, message: str, command: str) -> None:
    """Stop the page with instructions when a pipeline step has not been run yet."""
    if not available:
        st.info(message, icon=":material/info:")
        st.code(command, language="bash")
        st.stop()


def model_picker(key: str, tasks: list[str] | None = None) -> tuple[str, str]:
    """Task and model selectors; the model defaults to the task's best model."""
    tasks = tasks or trained_tasks()
    left, right = st.columns(2)
    task = left.selectbox("Task", tasks, format_func=TASK_LABELS.get, key=f"{key}_task",
                          index=tasks.index("multiclass") if "multiclass" in tasks else 0)
    metadata = get_metadata(task)
    names = list(metadata["models"])
    model = right.selectbox(
        "Model", names, index=names.index(metadata["best_model"]), key=f"{key}_model_{task}",
        format_func=lambda n: metadata["models"][n] + (" (best)" if n == metadata["best_model"] else ""),
    )
    return task, model


def sidebar_status() -> None:
    """What the dashboard is showing: dataset size and when the models were trained."""
    profile = get_profile()
    metadata = next((get_metadata(t) for t in ("multiclass", "binary") if get_metadata(t)), None)
    lines = []
    if profile:
        lines.append(f"Dataset <b>{compact(profile['rows_final'])} flows</b>, {profile['data_source']}")
    if metadata:
        lines.append(f"Models trained <b>{when(metadata['created_at'])}</b>")
    if lines:
        st.html(f'<div class="nids-status">{"<br>".join(lines)}</div>')
