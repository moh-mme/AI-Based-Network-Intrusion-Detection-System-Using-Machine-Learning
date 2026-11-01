import io
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    CONTEXT_COLUMNS,
    HEAT_SCALE,
    color_map,
    compact,
    get_config,
    get_demo_stream,
    get_detector,
    model_picker,
    require,
    synthetic_warning,
    trained_tasks,
)
from nids.data import LABEL_COL, read_flow_csv
from nids.predict import true_classes

st.title("Traffic Analyzer")
st.markdown("Upload network flows in **CICFlowMeter / CIC-IDS2017 CSV format** and classify every flow.")
synthetic_warning()
require(trained_tasks(), "No trained models yet.", "python -m nids.train")

task, model = model_picker("analyzer")
upload = st.file_uploader("Flow CSV file", type="csv")
use_sample = st.button("Or try 2,000 held-out test flows", disabled=get_demo_stream() is None)


# Cached by the source's id (not its contents) so that moving a filter widget
# does not re-read or re-classify a large file.
@st.cache_data(show_spinner="Reading file…", max_entries=2)
def read_upload(file_id: str, _upload) -> pd.DataFrame:
    return read_flow_csv(io.BytesIO(_upload.getvalue())).dropna(how="all").reset_index(drop=True)


@st.cache_data(show_spinner="Classifying flows…", max_entries=8)
def classify(source_id: str, task: str, model: str, trained_at: str,
             _flows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    result = get_detector(task, model).predict(_flows)
    return result, time.perf_counter() - start


@st.cache_data(show_spinner="Preparing download…", max_entries=2)
def predictions_csv(source_id: str, task: str, model: str, _flows, _result) -> bytes:
    return pd.concat([_flows, _result], axis=1).to_csv(index=False).encode("utf-8")


if use_sample:
    st.session_state.analyzer_source = "sample"
elif upload is not None:
    st.session_state.analyzer_source = "upload"
source = st.session_state.get("analyzer_source")
if source == "upload" and upload is None:
    source = None

if source is None:
    st.stop()
if source == "sample":
    stream = get_demo_stream()
    source_id = "sample"
    flows = stream.sample(min(2000, len(stream)), random_state=1).reset_index(drop=True)
    st.caption("Sample of held-out test flows (never seen during training).")
else:
    source_id = upload.file_id
    flows = read_upload(source_id, upload)

detector = get_detector(task, model)
try:
    result, elapsed = classify(source_id, task, model, detector.metadata["created_at"], flows)
except ValueError as error:
    st.error(str(error))
    st.stop()

cols = st.columns(4)
cols[0].metric("Flows analysed", f"{len(result):,}")
cols[1].metric("Flagged as attacks", f"{int(result['is_attack'].sum()):,}")
cols[2].metric("Attack share", f"{result['is_attack'].mean():.1%}")
cols[3].metric("Throughput", f"{compact(round(len(result) / max(elapsed, 1e-6)))} flows/s",
               help="Classification speed of the model on this machine.")

counts = result["prediction"].value_counts().rename_axis("Prediction").reset_index(name="Flows")
left, right = st.columns([3, 2])
fig = px.bar(counts, x="Prediction", y="Flows", color="Prediction", text_auto=",",
             color_discrete_map=color_map(counts["Prediction"]), title="Predicted traffic classes")
fig.update_layout(showlegend=False, height=380)
left.plotly_chart(fig, width="stretch")

if LABEL_COL in flows.columns:
    try:
        truth = true_classes(flows[LABEL_COL], task, get_config().data["category_map"])
    except ValueError as error:
        right.info(f"The file's labels could not be compared: {error}")
    else:
        right.metric("Accuracy against the file's labels", f"{(truth == result['prediction']).mean():.4f}")
        classes = [c for c in detector.classes if c in set(truth) | set(result["prediction"])]
        cm = pd.crosstab(truth, result["prediction"]).reindex(index=classes, columns=classes, fill_value=0)
        fig = px.imshow(cm, text_auto=",d", color_continuous_scale=HEAT_SCALE, aspect="auto",
                        labels=dict(x="Predicted", y="Actual", color="Flows"))
        fig.update_layout(height=330, margin=dict(t=10), coloraxis_showscale=False)
        right.plotly_chart(fig, width="stretch")
else:
    right.info("Add a `Label` column to the file to compare predictions with ground truth.")

st.subheader("Flagged flows")
table = pd.concat([result, flows[[c for c in CONTEXT_COLUMNS + [LABEL_COL] if c in flows.columns]]], axis=1)
left, right = st.columns([3, 1])
attack_classes = sorted(table.loc[table["is_attack"], "prediction"].unique())
shown = left.multiselect("Predicted class", attack_classes, default=attack_classes)
min_confidence = right.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05)
flagged = table[table["prediction"].isin(shown) & (table["confidence"] >= min_confidence)]
flagged = flagged.sort_values("confidence", ascending=False)
st.dataframe(flagged.drop(columns="is_attack").head(1000), width="stretch", hide_index=True,
             column_config={"confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.3f")})
st.caption(f"{len(flagged):,} flows match{' (showing the first 1,000)' if len(flagged) > 1000 else ''}.")

st.download_button(
    "Download all predictions (CSV)",
    predictions_csv(source_id, task, model, flows, result),
    file_name=f"predictions_{task}_{model}.csv",
    mime="text/csv",
    type="primary",
)
