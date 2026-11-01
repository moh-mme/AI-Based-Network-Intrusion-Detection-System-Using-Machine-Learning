import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    ACCENT,
    BORDER,
    CATEGORY_COLORS,
    INK,
    MUTED,
    SURFACE,
    TASK_LABELS,
    compact,
    get_demo_stream,
    get_metrics,
    get_profile,
    synthetic_warning,
    trained_tasks,
)
from nids.data import BENIGN, CATEGORY_COL

FLOWS_PER_COLUMN = 50

profile = get_profile()
tasks = trained_tasks()
metrics = {task: get_metrics(task) for task in tasks}
n_models = max((len(m["models"]) for m in metrics.values() if m), default=0)

st.title("AI-Based Network Intrusion Detection System")
if profile:
    st.markdown(
        f"{n_models or 'Several'} machine-learning models, trained on **{compact(profile['rows_final'])} "
        f"network flows** from CIC-IDS2017, label each flow as benign or as one of "
        f"**{len(profile['category_counts']) - 1} attack types**."
    )
else:
    st.markdown("Machine-learning models that label network flows as benign or as an attack type.")
synthetic_warning()


def traffic_tape(stream: pd.DataFrame) -> go.Figure:
    """Each column is 50 consecutive flows, stacked by category: a barcode of the traffic."""
    categories = stream[CATEGORY_COL].astype(str).to_numpy()
    counts = pd.crosstab(np.arange(len(categories)) // FLOWS_PER_COLUMN, categories)
    # Attacks along the baseline so bursts rise out of the benign background.
    attacks = [c for c in CATEGORY_COLORS if c in counts.columns and c != BENIGN]
    attacks += [c for c in counts.columns if c not in attacks and c != BENIGN]
    fig = go.Figure()
    for category in attacks + [BENIGN]:
        if category not in counts.columns:
            continue
        fig.add_bar(
            x=counts.index, y=counts[category], name=category, customdata=counts.index * FLOWS_PER_COLUMN,
            marker=dict(color=CATEGORY_COLORS.get(category, MUTED), line_width=0),
            hovertemplate=f"<b>{category}</b>: %{{y}} of {FLOWS_PER_COLUMN} flows<br>"
                          "starting at flow %{customdata:,}<extra></extra>",
        )
    fig.update_layout(
        barmode="stack", bargap=0, height=190, margin=dict(l=0, r=0, t=4, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, FLOWS_PER_COLUMN]),
        legend=dict(orientation="h", yanchor="top", y=-0.06, x=0, traceorder="normal",
                    font=dict(color=MUTED)),
    )
    return fig


stream = get_demo_stream()
if stream is not None:
    st.plotly_chart(traffic_tape(stream), width="stretch", config={"displayModeBar": False})
    attack_share = (stream[CATEGORY_COL].astype(str) != BENIGN).mean()
    st.caption(
        f"The live-monitor replay: {len(stream):,} held-out test flows, {FLOWS_PER_COLUMN} per column, "
        f"with attacks ({attack_share:.0%}) grouped into bursts. Hover a column to see what it holds."
    )

if tasks:
    st.subheader("Detection results")
    for task in tasks:
        report = metrics[task]
        if not report:
            continue
        best = report["models"][report["best_model"]]
        m = best["metrics"]
        # Detection and false alarms lead: with mostly benign traffic, accuracy alone
        # flatters any model, so it is shown as supporting text.
        with st.container(border=True):
            label, *cells = st.columns([1.4, 1, 1, 1], vertical_alignment="center")
            label.markdown(f"**{TASK_LABELS[task]}**  \n{best['display_name']}, "
                           f"accuracy {m['accuracy']:.2%}")
            cells[0].metric("Attacks detected", f"{m['detection_rate']:.2%}")
            cells[1].metric("False alarms", f"{m['false_alarm_rate']:.2%}")
            cells[2].metric("Macro F1", f"{m['f1_macro']:.3f}")
    report = next(r for r in metrics.values() if r)
    benign_share = report["test_class_counts"][BENIGN] / report["n_test"]
    st.caption(
        "Attacks detected: share of attack flows flagged. False alarms: share of benign flows "
        "wrongly flagged. Macro F1: the F1 score averaged over all classes, so rare attacks count "
        f"as much as common ones. Measured on {report['n_test']:,} flows the models never saw "
        f"during training, {benign_share:.0%} of them benign, as in the original capture."
    )

st.subheader("How it works")
st.graphviz_chart(
    f"""
    digraph {{
        rankdir=TB; bgcolor="transparent"; nodesep=0.4; ranksep=0.5;
        node [shape=box, style="rounded,filled", fillcolor="{SURFACE}", color="{BORDER}",
              fontcolor="{INK}", fontname="Barlow", fontsize=12, width=2.1, height=0.75];
        edge [color="{MUTED}", arrowsize=0.7];
        raw [label="CIC-IDS2017\\nflow CSVs"];
        clean [label="Cleaning\\nNaN, infinity, duplicates"];
        split [label="Stratified split\\n80% train, 20% test"];
        sample [label="Undersample\\ntraining set"];
        models [label="6 classifiers\\nLR, DT, RF, KNN\\nXGBoost, MLP", color="{ACCENT}", penwidth=2];
        evaluate [label="Evaluation\\nF1, detection, false alarms"];
        deploy [label="Best model\\ndashboard and CLI"];
        // Two rows (snake layout) so the labels stay readable on narrow screens.
        {{rank=same; raw; clean; split; sample;}}
        {{rank=same; deploy; evaluate; models;}}
        raw -> clean -> split -> sample;
        sample -> models;
        deploy -> evaluate -> models [dir=back];
    }}
    """,
    width="stretch",
)

steps = [
    ("Download CIC-IDS2017 (MachineLearningCSV) and extract the CSV files into `data/raw/`",
     profile is not None, None),
    ("Clean and profile the dataset", profile is not None, "python -m nids.prepare"),
    ("Train and evaluate all models", bool(tasks), "python -m nids.train"),
    ("Classify a CSV file from the command line", bool(tasks), "python -m nids.predict --input flows.csv"),
]
complete = all(done for _, done, _ in steps)
with st.expander("Reproduce these results" if complete else "Getting started", expanded=not complete):
    for text, done, command in steps:
        st.markdown(f"{':material/check_circle:' if done else ':material/radio_button_unchecked:'} {text}")
        if command:
            st.code(command, language="bash")
