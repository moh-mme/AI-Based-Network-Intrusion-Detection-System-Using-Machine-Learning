import time
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from attack_globe import attack_globe
from attack_map import attack_map
from common import (
    CATEGORY_COLORS,
    INK,
    color_map,
    icon,
    get_demo_stream,
    get_detector,
    model_picker,
    require,
    synthetic_warning,
    trained_tasks,
)
from nids.data import ATTACK_COL, BENIGN, CATEGORY_COL, LABEL_COL

HISTORY_SECONDS = 120
MAX_ALERTS = 300

st.title("Live Monitor")
st.markdown("Held-out test flows, never seen in training, are replayed as a live feed and classified "
            "as they arrive.")
synthetic_warning()
require(trained_tasks(), "No trained models yet.", "python -m nids.train")
stream = get_demo_stream()
require(stream is not None, "The replay stream is created during training.", "python -m nids.train")

# One row of controls keeps the globe above the fold on a 768 px tall screen.
pickers, speed, start_button, reset_button = st.columns([5, 3, 1.3, 1.3], vertical_alignment="bottom")
with pickers:
    task, model = model_picker("live")
detector = get_detector(task, model)


def new_state() -> dict:
    return {"key": (task, model), "position": 0, "processed": 0, "correct": 0, "tick": 0,
            "history": [], "alerts": [], "counts": Counter(), "last_attacks": {}, "latency_ms": 0.0}


if st.session_state.get("live", {}).get("key") != (task, model):
    st.session_state.live = new_state()
    st.session_state.live_running = False


def toggle():
    st.session_state.live_running = not st.session_state.get("live_running", False)


def reset():
    st.session_state.live = new_state()
    st.session_state.live_running = False


running = st.session_state.get("live_running", False)
rate = speed.select_slider("Flows per second", [10, 25, 50, 100, 250, 500], value=50)
start_button.button("Pause" if running else "Start", on_click=toggle, type="primary",
                    icon=":material/pause:" if running else ":material/play_arrow:", width="stretch")
reset_button.button("Reset", on_click=reset, icon=":material/restart_alt:", width="stretch")


def step(state: dict, n: int) -> None:
    """Classify the next `n` flows of the replay stream."""
    positions = [(state["position"] + i) % len(stream) for i in range(n)]
    batch = stream.iloc[positions].reset_index(drop=True)
    state["position"] = (state["position"] + n) % len(stream)

    start = time.perf_counter()
    result = detector.predict(batch)
    state["latency_ms"] = (time.perf_counter() - start) * 1000

    if task == "binary":
        truth = batch[ATTACK_COL].map({0: BENIGN, 1: "Attack"})
    else:
        truth = batch[CATEGORY_COL].astype(str)
    correct = truth.to_numpy() == result["prediction"].to_numpy()
    state["processed"] += n
    state["correct"] += int(correct.sum())
    state["counts"].update(result["prediction"])

    now = datetime.now()
    attacks = result["is_attack"].to_numpy()
    state["tick"] += 1
    state["last_attacks"] = {k: int(v) for k, v in Counter(result["prediction"][attacks]).items()}
    state["history"] = (state["history"] + [{"Time": now, "Flows": n, "Alerts": int(attacks.sum())}])[-HISTORY_SECONDS:]
    new_alerts = [
        {
            "Time": now.strftime("%H:%M:%S"),
            "Detected": result["prediction"].iat[i],
            "Confidence": float(result["confidence"].iat[i]),
            "Dst port": int(batch["Destination Port"].iat[i]) if "Destination Port" in batch else None,
            "Actual label": str(batch[LABEL_COL].iat[i]),
            "Correct": bool(correct[i]),
        }
        for i in reversed(range(n)) if attacks[i]
    ]
    state["alerts"] = (new_alerts + state["alerts"])[:MAX_ALERTS]


def threat_band(level: str, headline: str, detail: str) -> None:
    """Full-width status strip; the one loud element on the page."""
    color = "#FF6B6F" if level == "alert" else "#7FA6CC"
    st.html(f'<div class="threat-band threat-{level}" role="status" aria-live="polite">'
            f'{icon(level, 30, color)}<span>{headline} <small>{detail}</small></span></div>')


def render(state: dict) -> None:
    history = pd.DataFrame(state["history"])
    current_alerts = int(history["Alerts"].iloc[-1]) if len(history) else 0
    alerts_total = sum(c for cls, c in state["counts"].items() if cls != BENIGN)

    map_column, side = st.columns([3, 2], gap="medium")
    with map_column:
        view = st.segmented_control("View", ["World map", "3D globe"], default="World map",
                                    key="live_view", label_visibility="collapsed")
        if view == "3D globe":
            attack_globe(state["tick"], state["last_attacks"])
        else:
            attack_map(state["tick"], state["last_attacks"])
        st.caption("Each arc is an attack the model has just detected, coloured by type and heading for "
                   "the protected network (the CIC-IDS2017 testbed in Fredericton, Canada). The dataset "
                   "records no attacker locations, so arc origins are chosen at random.")
    with side:
        if current_alerts:
            recent = Counter(a["Detected"] for a in state["alerts"][:current_alerts]).most_common(1)[0][0]
            threat_band("alert", f"Under attack: {current_alerts} {recent} flows",
                        f"of {int(history['Flows'].iloc[-1])} in the last second")
        elif state["processed"]:
            threat_band("clear", "All clear", "no attack flows in the last second")
        else:
            threat_band("clear", "Monitor idle", "press Start to replay the held-out traffic")
        top, bottom = st.columns(2), st.columns(2)
        top[0].metric("Flows analysed", f"{state['processed']:,}")
        top[1].metric("Attacks detected", f"{alerts_total:,}")
        bottom[0].metric("Agreement with labels",
                         f"{state['correct'] / state['processed']:.2%}" if state["processed"] else "—",
                         help="Share of flows whose prediction matches the dataset's ground-truth label.")
        bottom[1].metric("Batch latency", f"{state['latency_ms']:.0f} ms" if state["processed"] else "—",
                         help="Time to classify one second's batch of flows.")
        attack_counts = pd.DataFrame(
            [(cls, c) for cls, c in state["counts"].items() if cls != BENIGN], columns=["Type", "Flows"]
        ).sort_values("Flows")
        if len(attack_counts):
            fig = px.bar(attack_counts, x="Flows", y="Type", orientation="h", color="Type", text_auto=",",
                         color_discrete_map=color_map(attack_counts["Type"]), title="Detected attacks")
            fig.update_layout(height=60 + 34 * len(attack_counts), margin=dict(t=30, b=0, l=0, r=10),
                              showlegend=False, yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.caption("Detected attacks are counted here by type.")

    fig = go.Figure()
    if len(history):
        fig.add_scatter(x=history["Time"], y=history["Flows"] - history["Alerts"], name="Benign",
                        stackgroup="traffic", line=dict(color=CATEGORY_COLORS[BENIGN]))
        fig.add_scatter(x=history["Time"], y=history["Alerts"], name="Attack",
                        stackgroup="traffic", line=dict(color=CATEGORY_COLORS["Attack"]))
    fig.update_layout(height=260, margin=dict(t=30, b=10), title="Classified flows per second",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Alert feed** (newest first)")
    if state["alerts"]:
        alerts = pd.DataFrame(state["alerts"][:25])
        st.dataframe(
            alerts.style.map(lambda v: f"color: {CATEGORY_COLORS.get(v, INK)}; font-weight: 600",
                             subset=["Detected"]),
            hide_index=True, width="stretch",
            column_config={
                "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                "Correct": st.column_config.CheckboxColumn(
                    "Correct", help="Prediction matches the dataset's ground-truth label"),
            },
        )
    else:
        st.caption("Alerts appear here as attacks are detected.")


@st.fragment(run_every=1.0 if running else None)
def monitor() -> None:
    state = st.session_state.live
    if st.session_state.get("live_running"):
        step(state, rate)
    render(state)


monitor()
