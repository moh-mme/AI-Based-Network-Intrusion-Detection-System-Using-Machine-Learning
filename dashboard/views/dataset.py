import pandas as pd
import plotly.express as px
import streamlit as st

from common import color_map, compact, get_explorer_sample, get_profile, require, synthetic_warning, when
from nids.data import BENIGN, CATEGORY_COL

st.title("Dataset Explorer")
synthetic_warning()
profile = get_profile()
require(profile, "The dataset has not been prepared yet.", "python -m nids.prepare")

st.caption(f"{profile['data_source']}, {len(profile['files'])} CSV files, prepared {when(profile['created_at'])}.")
cols = st.columns(5)
for column, (label, key) in zip(cols, [("Rows read", "rows_read"), ("NaN / inf removed", "rows_invalid"),
                                       ("Duplicates removed", "duplicates_removed"), ("Clean flows", "rows_final")]):
    column.metric(label, compact(profile[key]), help=f"{profile[key]:,}")
cols[4].metric("Attack share", f"{profile['attack_ratio']:.1%}")

st.subheader("Class distribution")
categories = pd.Series(profile["category_counts"], name="Flows").rename_axis("Category").reset_index()
left, right = st.columns([2, 1])
with left:
    log_scale = st.toggle("Logarithmic scale", value=True,
                          help="Benign traffic dwarfs the rarer attacks on a linear scale.")
    fig = px.bar(categories, x="Category", y="Flows", color="Category", log_y=log_scale,
                 color_discrete_map=color_map(categories["Category"]), text_auto=",")
    fig.update_layout(showlegend=False, height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
with right:
    attacks = profile["rows_final"] - profile["category_counts"].get(BENIGN, 0)
    split = pd.DataFrame({"Type": [BENIGN, "Attack"], "Flows": [profile["category_counts"].get(BENIGN, 0), attacks]})
    fig = px.pie(split, names="Type", values="Flows", hole=0.55, color="Type",
                 color_discrete_map=color_map(split["Type"]))
    fig.update_layout(height=420, margin=dict(t=40, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

labels = pd.DataFrame(
    [(label, profile["label_to_category"].get(label, "?"), count)
     for label, count in profile["label_counts"].items()],
    columns=["Original label", "Category", "Flows"],
)
labels["Share"] = labels["Flows"] / labels["Flows"].sum()
with st.expander("Original labels and their categories"):
    st.dataframe(labels, hide_index=True, width="stretch",
                 column_config={"Flows": st.column_config.NumberColumn(format="%d"),
                                "Share": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)})

st.subheader("Feature distributions")
sample = get_explorer_sample()
if sample is None:
    st.info("Explorer sample not found; rerun `python -m nids.prepare`.")
else:
    features = [f for f in profile["features"] if f not in profile["constant_features"]]
    default = features.index("Flow Duration") if "Flow Duration" in features else 0
    left, right = st.columns([1, 3])
    feature = left.selectbox("Feature", features, index=default)
    log_y = left.toggle("Logarithmic axis", value=True, key="feature_log")
    left.caption(f"Up to {sample[CATEGORY_COL].value_counts().max():,} sampled flows per category.")
    data = sample[[feature, CATEGORY_COL]].copy()
    data[CATEGORY_COL] = data[CATEGORY_COL].astype(str)
    if log_y:
        data = data[data[feature] > 0]  # a log axis cannot show zero or negative values
    fig = px.box(data, x=CATEGORY_COL, y=feature, color=CATEGORY_COL, log_y=log_y,
                 color_discrete_map=color_map(data[CATEGORY_COL].unique()))
    fig.update_layout(showlegend=False, height=420, margin=dict(t=10, b=10))
    right.plotly_chart(fig, width="stretch")

stats = pd.DataFrame(profile["feature_stats"]).T.rename_axis("Feature").reset_index()
with st.expander(f"Summary statistics for all {len(stats)} features"):
    st.dataframe(stats, hide_index=True, width="stretch")
if profile["constant_features"]:
    with st.expander(f"{len(profile['constant_features'])} constant features (removed before training)"):
        st.write(", ".join(profile["constant_features"]))
