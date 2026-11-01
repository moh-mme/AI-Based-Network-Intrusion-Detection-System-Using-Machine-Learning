"""Dashboard entry point. Run from the project root:

    streamlit run dashboard/app.py
"""

import streamlit as st

from common import CSS, icon, sidebar_status

st.set_page_config(page_title="AI-NIDS", page_icon=":material/shield:", layout="wide")
st.html(CSS)

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/shield:", default=True),
    st.Page("views/attack_map_page.py", title="Attack Map", icon=":material/public:"),
    st.Page("views/dataset.py", title="Dataset Explorer", icon=":material/dataset:"),
    st.Page("views/performance.py", title="Model Performance", icon=":material/monitoring:"),
    st.Page("views/research.py", title="Research Findings", icon=":material/science:"),
    st.Page("views/analyzer.py", title="Traffic Analyzer", icon=":material/manage_search:"),
    st.Page("views/live_monitor.py", title="Live Monitor", icon=":material/sensors:"),
]
navigation = st.navigation(pages, position="hidden")

with st.sidebar:
    st.html(f'<div class="nids-brand">{icon("brand", 34)}'
            '<div><b>AI-NIDS</b><span>Network intrusion detection</span></div></div>')
    for page in pages:
        st.page_link(page)
    st.divider()
    sidebar_status()

navigation.run()
