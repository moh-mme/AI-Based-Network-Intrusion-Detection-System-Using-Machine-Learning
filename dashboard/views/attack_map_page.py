import numpy as np
import streamlit as st

from attack_map import attack_map
from common import get_demo_stream, require, synthetic_warning
from nids.data import BENIGN, CATEGORY_COL

FLOWS_PER_SECOND = 50  # replay speed, as the live monitor's default

st.title("Attack Map")
st.markdown("Attack bursts from the held-out test traffic, replayed on a loop and drawn as arcs "
            "heading for the protected network.")
synthetic_warning()
stream = get_demo_stream()
require(stream is not None, "The replay stream is created during training.", "python -m nids.train")

flows = stream[CATEGORY_COL].astype(str).to_numpy()
seconds = [flows[i:i + FLOWS_PER_SECOND] for i in range(0, len(flows), FLOWS_PER_SECOND)]
sequence = [{c: int((s == c).sum()) for c in np.unique(s) if c != BENIGN} for s in seconds]

with st.container(border=True):
    st.markdown("**Attacks reaching the protected network** "
                ":green-badge[:material/fiber_manual_record: Live replay]")
    attack_map(0, {}, key="attack_map_page", sequence=sequence)
st.caption(f"Held-out test traffic replayed at {FLOWS_PER_SECOND} flows per second: each arc is a burst of "
           "attack flows, coloured by type, heading for the CIC-IDS2017 testbed in Fredericton, Canada. "
           "The dataset records no attacker locations, so arcs start from random countries.")
