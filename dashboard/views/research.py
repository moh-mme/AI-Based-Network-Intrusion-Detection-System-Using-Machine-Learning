import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import ACCENT, CATEGORY_COLORS, HEAT_SCALE, MUTED, get_config, read_json, require, when
from nids.models import DISPLAY_NAMES as MODEL_NAMES
from nids.research_report import SUPERVISED, headline, loao_matrix, primary

st.title("Research Findings")
st.markdown("How much of the reported 99.9% survives a realistic evaluation, and can a hybrid detector "
            "catch attack types it was never trained on?")

results = read_json(get_config().paths.research_reports / "results.json")
require(results, "The research experiments have not been run yet.", "python -m nids.experiments")
settings = results["settings"]
alpha = settings["target_fpr"]
ran = [name for name in ("random", "temporal", "loao", "shap") if name in results]
latest = max(results[name]["created_at"] for name in ran)
quick = [name for name in ran if results[name].get("quick")]
if quick:
    st.warning(f"Quick-run results (small samples) for: {', '.join(quick)}. "
               "Run `python -m nids.experiments` for the final numbers.", icon=":material/warning:")

model, features = primary(results)
items = headline(results)
if items:
    labels = [label for label, _, _ in items]
    fig = go.Figure(go.Bar(
        x=[dr for _, dr, _ in items], y=labels, orientation="h",
        marker_color=["#2B7A9B" if "hybrid" in label else "#5F86AD" if "Random" in label else "#E5484D"
                      for label in labels],
        text=[f"{dr:.1%}  (false alarms {fpr:.2%})" for _, dr, fpr in items], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1.25], tickformat=".0%", title="Share of attack flows detected")
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(height=60 * len(items) + 90, margin=dict(t=10, r=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        f"{MODEL_NAMES.get(model, model)}, binary (benign vs attack). Grey: the usual random split. Red: "
        f"attacks from a family or day the model never saw. Blue: the proposed hybrid, with alarms kept "
        f"within {alpha:.0%} of benign flows. Unseen-family results are averaged over the seven families."
    )

loao_tab, temporal_tab, random_tab, shap_tab, method_tab = st.tabs(
    ["Unseen attack families", "Later days", "Random split", "Shortcut features", "Method"])


def pickers(key: str) -> tuple[str, str]:
    left, right = st.columns(2)
    models = settings["models"]
    chosen_model = left.selectbox("Model", models, index=models.index(model), key=f"{key}_model",
                                  format_func=lambda m: MODEL_NAMES.get(m, m))
    sets = list(settings["feature_sets"])
    chosen_set = right.selectbox("Features", sets, index=sets.index(features), key=f"{key}_features",
                                 format_func=settings["feature_sets"].get)
    return chosen_model, chosen_set


with loao_tab:
    loao = results.get("loao", {}).get("result")
    if not loao:
        st.info("Run `python -m nids.experiments --only loao`.")
    else:
        st.markdown("Each attack family in turn is removed from training, then the detectors are tested "
                    "on it: a controlled stand-in for a zero-day attack.")
        chosen_model, chosen_set = pickers("loao")
        matrix, fpr, detectors = loao_matrix(loao["rows"], loao["families"], chosen_model, chosen_set)
        short = [d.replace("Hybrid: supervised + ", "Hybrid + ") for d in detectors]
        fig = px.imshow(matrix, x=short, y=loao["families"], color_continuous_scale=HEAT_SCALE, zmin=0, zmax=1,
                        text_auto=".0%", aspect="auto",
                        labels=dict(x="Detector", y="Family left out of training", color="Detected"))
        fig.update_layout(height=70 + 48 * len(loao["families"]), margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        summary = pd.DataFrame({"Detector": detectors,
                                "Unseen attacks detected (mean)": np.nanmean(matrix, axis=0),
                                "False alarms (mean)": np.nanmean(fpr, axis=0)})
        st.dataframe(summary, hide_index=True, width="stretch", column_config={
            "Unseen attacks detected (mean)": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                              format="percent"),
            "False alarms (mean)": st.column_config.NumberColumn(format="percent"),
        })

        misses = [r for r in loao["rows"] if r["features"] == chosen_set and r["model"] == chosen_model
                  and r["detector"] == SUPERVISED]
        counts = loao["unseen_scores"].get(chosen_set, {}).get(chosen_model)
        left, right = st.columns([3, 2])
        if counts:
            edges = np.asarray(settings["score_bins"])
            share = np.asarray(counts) / max(sum(counts), 1)
            centers = np.sqrt(np.maximum(edges[:-1], 1e-7) * edges[1:])
            colors = [CATEGORY_COLORS["Attack"] if e <= settings["confident_threshold"] else
                      MUTED if e <= 0.5 else ACCENT for e in edges[1:]]
            fig = go.Figure(go.Bar(x=centers, y=share, marker_color=colors,
                                   hovertemplate="P(attack) ≈ %{x:.0e}: %{y:.1%}<extra></extra>"))
            fig.update_xaxes(type="log", title="Model's P(attack) for flows of the unseen family")
            fig.update_yaxes(tickformat=".0%", title="Share of flows")
            fig.add_vline(x=0.5, line_dash="dash", line_color=MUTED)
            fig.update_layout(height=320, margin=dict(t=30), title="How sure the model is when it misses")
            left.plotly_chart(fig, width="stretch")
        if misses:
            right.markdown(f"**Missed with over {1 - settings['confident_threshold']:.0%} confidence**")
            right.dataframe(pd.DataFrame({"Family left out": [r["held_out"] for r in misses],
                                          "Share of its flows": [r.get("confident_miss") for r in misses]}),
                            hide_index=True, width="stretch",
                            column_config={"Share of its flows": st.column_config.ProgressColumn(
                                min_value=0, max_value=1, format="percent")})
            right.caption("Red bars: attacks the model called benign while almost certain. A confidence "
                          "threshold cannot catch these, which is why the hybrid adds an anomaly detector.")

with temporal_tab:
    temporal = results.get("temporal", {}).get("result")
    if not temporal or not temporal.get("rows"):
        st.info("Run `python -m nids.experiments --only temporal`.")
    else:
        st.markdown(f"Trained on **{', '.join(temporal['train_days'])}**, tested on "
                    f"**{', '.join(temporal['test_days'])}**. Attack types in the test days that never "
                    f"appear in training: {', '.join(temporal['unseen'])}.")
        chosen_model, chosen_set = pickers("temporal")
        rows = [r for r in temporal["rows"] if r["features"] == chosen_set and r["model"] in (chosen_model, "none")]
        frame = pd.DataFrame([{"Detector": r["detector"], "Thresholds set on": r.get("calibration") or "—",
                               "Unseen attacks detected": r["dr_unseen"],
                               **{f"{family}": rate for family, rate in r["by_family"].items()},
                               "False alarms": r["fpr"], "AUC": r.get("auc_unseen")} for r in rows])
        percent = {c: st.column_config.NumberColumn(format="percent") for c in frame.columns
                   if c not in ("Detector", "Thresholds set on", "AUC")}
        st.dataframe(frame, hide_index=True, width="stretch",
                     column_config={**percent, "AUC": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Benign traffic also changes between days, so thresholds set on the training days can "
                   "overshoot the alarm budget. 'Recent benign' re-sets them on benign flows from the test "
                   "days, as an operator would after collecting a little recent normal traffic.")

with random_tab:
    rows = results.get("random", {}).get("result")
    if not rows:
        st.info("Run `python -m nids.experiments --only random`.")
    else:
        st.markdown("The usual protocol: a random 80/20 split, so every attack type appears in training. "
                    "This is where the 99.9% figures in the literature come from.")
        st.dataframe(pd.DataFrame([{
            "Task": r["task"], "Features": settings["feature_sets"][r["features"]],
            "Model": MODEL_NAMES.get(r["model"], r["model"]), "F1 macro": r["f1_macro"],
            "Detection": r["detection_rate"], "False alarms": r["false_alarm_rate"]} for r in rows]),
            hide_index=True, width="stretch", column_config={
                "F1 macro": st.column_config.NumberColumn(format="%.4f"),
                "Detection": st.column_config.NumberColumn(format="percent"),
                "False alarms": st.column_config.NumberColumn(format="percent")})

with shap_tab:
    shap = results.get("shap", {}).get("result", {})
    if "all" not in shap:
        st.info("Run `python -m nids.experiments --only shap`.")
    else:
        st.markdown("Share of the model's decision that comes from each candidate shortcut (TreeSHAP, "
                    "XGBoost, binary, trained with all features). A port names the service targeted in "
                    "this testbed, and TCP window sizes depend on each host's operating system: both can "
                    "identify the testbed's machines rather than how an attack behaves.")
        by_category = shap["all"]["by_category"]
        frame = pd.DataFrame([{"Category": c, "Feature": f, "Share": s}
                              for c, v in by_category.items() for f, s in v["shortcut_shares"].items()])
        fig = px.bar(frame, x="Category", y="Share", color="Feature", barmode="stack",
                     color_discrete_sequence=[ACCENT, "#F2C14E", "#F08A4B"])
        fig.update_yaxes(tickformat=".0%", title="Share of the decision")
        fig.update_layout(height=360, margin=dict(t=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
        st.plotly_chart(fig, width="stretch")
        strictest = list(settings["feature_sets"])[-1]
        without = shap.get(strictest, {}).get("by_category", {})
        st.dataframe(pd.DataFrame([{
            "Category": c, "Top features (all)": ", ".join(f["feature"] for f in v["top_features"][:3]),
            f"Top features ({settings['feature_sets'][strictest].lower()})": ", ".join(
                f["feature"] for f in without.get(c, {}).get("top_features", [])[:3])}
            for c, v in by_category.items()]), hide_index=True, width="stretch")

with method_tab:
    st.markdown(f"""
**Protocols**
- *Random split*: stratified 80/20, the setup behind most published results.
- *Unseen attack family*: one family at a time is removed from training and tested on all its flows.
- *Later days*: train on the first days of the capture week, test on the last; every test attack type is new.

**Detectors**
- *Supervised*: {', '.join(MODEL_NAMES.get(m, m) for m in settings['models'])}, trained on benign and attack flows.
- *Anomaly detectors*: {', '.join(settings['anomaly_detectors'].values())}, trained on benign flows only.
- *Hybrid (proposed)*: raises an alarm when either the supervised model or the anomaly detector finds a
  flow suspicious.

**Alarm budget.** Thresholds are set on {alpha:.0%} of held-out benign flows with conformal p-values: a flow
is flagged when fewer than {alpha:.0%} of benign calibration flows look as suspicious. The hybrid gives each
part half the budget, so together they stay within {alpha:.0%} (union bound).

Results computed {when(latest)}. Full tables: `reports/research/summary.md`.
""")
