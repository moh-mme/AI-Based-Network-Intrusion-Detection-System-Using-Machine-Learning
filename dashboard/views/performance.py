import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    ACCENT,
    BORDER,
    HEAT_SCALE,
    SERIES_COLORS,
    SERIES_DASHES,
    TASK_LABELS,
    get_metrics,
    require,
    synthetic_warning,
    trained_tasks,
    when,
)
from nids.evaluate import METRIC_LABELS

# Short headers so the whole table fits a 1366 px wide screen.
TABLE_COLUMNS = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision",
    "recall_macro": "Recall",
    "f1_macro": "F1",
    "mcc": "MCC",
    "detection_rate": "Detection",
    "false_alarm_rate": "False alarms",
}
LOWER_IS_BETTER = ["False alarms", "Train (s)", "ms / 1k flows"]
HIGHLIGHT = "font-weight: 700; color: #FFFFFF; background-color: rgba(230, 237, 245, 0.08)"
# Shape as well as colour tells the metrics apart.
DOT_METRICS = {"f1_macro": "circle", "precision_macro": "square", "recall_macro": "diamond",
               "mcc": "triangle-up"}

st.title("Model Performance")
synthetic_warning()
tasks = trained_tasks()
require(tasks, "No trained models yet.", "python -m nids.train")

task = st.radio("Task", tasks, format_func=TASK_LABELS.get, horizontal=True,
                index=tasks.index("multiclass") if "multiclass" in tasks else 0)
report = get_metrics(task)
require(report, f"No evaluation report for the {task} task.", f"python -m nids.train --tasks {task}")
models, best = report["models"], report["best_model"]
names = [r["display_name"] for r in models.values()]
st.caption(
    f"Trained on {report['n_train']:,} flows and tested on {report['n_test']:,} held-out flows that keep "
    f"the real class distribution ({report['data_source']}, {when(report['created_at'])})."
)

# --- Comparison table -------------------------------------------------------
table = pd.DataFrame([
    {
        "Model": r["display_name"] + (" (best)" if name == best else ""),
        **{label: r["metrics"].get(key, np.nan) for key, label in TABLE_COLUMNS.items()},
        "Train (s)": r["fit_seconds"],
        "ms / 1k flows": r["predict_ms_per_1k"],
    }
    for name, r in models.items()
]).set_index("Model")
st.dataframe(
    table.style
    .format("{:.4f}", subset=list(TABLE_COLUMNS.values()), na_rep="—")
    .format("{:.1f}", subset=["Train (s)"])
    .format("{:.2f}", subset=["ms / 1k flows"])
    .highlight_max(subset=[c for c in table.columns if c not in LOWER_IS_BETTER], props=HIGHLIGHT)
    .highlight_min(subset=LOWER_IS_BETTER, props=HIGHLIGHT),
    width="stretch",
)
st.caption(
    "Precision, recall and F1 are macro averages, so every class counts equally. Detection and false "
    "alarms treat all attack types as one 'attack' class. The best value in each column is in bold; "
    f"the best model is chosen by {METRIC_LABELS.get(report['selection_metric'], report['selection_metric'])}."
)


def dot_layout(fig: go.Figure, height: int) -> None:
    fig.update_yaxes(autorange="reversed", title=None, showgrid=True, gridcolor=BORDER)
    fig.update_layout(height=height, margin=dict(t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))


scores_tab, roc_tab, speed_tab = st.tabs(["Scores", "ROC curves", "Speed"])
with scores_tab:
    fig = go.Figure()
    for (metric, symbol), color in zip(DOT_METRICS.items(), SERIES_COLORS):
        fig.add_scatter(
            x=[r["metrics"][metric] for r in models.values()], y=names, mode="markers",
            name=METRIC_LABELS[metric], marker=dict(color=color, symbol=symbol, size=13),
            hovertemplate=f"{METRIC_LABELS[metric]}: %{{x:.4f}}<extra>%{{y}}</extra>",
        )
    fig.update_xaxes(title="Score (higher is better)")
    dot_layout(fig, 70 * len(names) + 110)
    st.plotly_chart(fig, width="stretch")
    st.caption("Dots rather than bars, because the axis starts near the lowest score instead of zero: "
               "it zooms in on where the models differ without making shorter bars look like failures.")

with roc_tab:
    fig = go.Figure()
    for r, color, dash in zip(models.values(), SERIES_COLORS, SERIES_DASHES):
        if r["roc"]:
            fig.add_scatter(x=r["roc"]["fpr"], y=r["roc"]["tpr"], mode="lines",
                            line=dict(color=color, dash=dash, width=2),
                            name=f"{r['display_name']} (AUC {r['metrics']['roc_auc']:.4f})")
    fig.update_xaxes(type="log", range=[-5, 0], title="False alarm rate (log scale)")
    fig.update_yaxes(range=[0, 1.01], title="Detection rate")
    fig.update_layout(height=450, margin=dict(t=10), legend=dict(x=0.99, xanchor="right", y=0.02))
    st.plotly_chart(fig, width="stretch")
    st.caption("Attack vs benign, scored by 1 − P(benign). The log axis zooms in on the low "
               "false-alarm region where a deployed IDS has to operate.")

with speed_tab:
    left, right = st.columns(2)
    for column, key, title in ((left, "fit_seconds", "Training time (s)"),
                               (right, "predict_ms_per_1k", "Prediction time (ms per 1,000 flows)")):
        values = [r[key] for r in models.values()]
        fig = go.Figure(go.Scatter(
            x=values, y=names, mode="markers+text", text=[f"{v:,.2f}" for v in values],
            textposition="middle right", cliponaxis=False, marker=dict(color=ACCENT, size=13),
            hovertemplate="%{x:,.2f}<extra>%{y}</extra>",
        ))
        fig.update_xaxes(type="log", title=f"{title}, log scale")
        dot_layout(fig, 60 * len(names) + 90)
        fig.update_layout(margin=dict(t=10, r=60))
        column.plotly_chart(fig, width="stretch")
    st.caption("Lower is better. Times were measured on the training machine.")

# --- Single model ------------------------------------------------------------
st.subheader("Model details")
model_names = list(models)
name = st.selectbox("Model", model_names, index=model_names.index(best),
                    format_func=lambda n: models[n]["display_name"])
result, classes = models[name], report["classes"]

left, right = st.columns([3, 2])
with left:
    normalize = st.toggle("Show share of each actual class", value=True)
    cm = np.asarray(result["confusion_matrix"], dtype=float)
    z = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    fig = px.imshow(z, x=classes, y=classes, color_continuous_scale=HEAT_SCALE,
                    text_auto=".1%" if normalize else ",d", aspect="auto",
                    labels=dict(x="Predicted", y="Actual", color="Share" if normalize else "Flows"))
    fig.update_layout(height=max(380, 55 * len(classes)), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
with right:
    per_class = pd.DataFrame({c: result["report"][c] for c in classes}).T
    per_class["support"] = per_class["support"].astype(int)
    st.markdown("**Per-class results**")
    st.dataframe(per_class.rename(columns=str.capitalize), width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f")
                                for c in ("Precision", "Recall", "F1-score")})
    m = result["metrics"]
    a, b = st.columns(2)
    a.metric("Detection rate", f"{m['detection_rate']:.2%}")
    b.metric("False alarm rate", f"{m['false_alarm_rate']:.3%}")

if result.get("feature_importance"):
    top = pd.DataFrame(result["feature_importance"][:20])
    fig = px.bar(top.iloc[::-1], x="importance", y="feature", orientation="h",
                 title=f"Top {len(top)} features: {result['display_name']}",
                 color_discrete_sequence=[ACCENT])
    fig.update_layout(height=560, margin=dict(t=40), yaxis_title=None, xaxis_title="Importance")
    st.plotly_chart(fig, width="stretch")

with st.expander("Class counts in the training and test sets"):
    st.dataframe(pd.DataFrame({"Train (undersampled)": report["train_class_counts"],
                               "Test (real distribution)": report["test_class_counts"]}),
                 width="stretch")
