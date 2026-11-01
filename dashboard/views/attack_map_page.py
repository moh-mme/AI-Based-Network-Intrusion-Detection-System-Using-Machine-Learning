import numpy as np
import streamlit as st

from attack_map import attack_map
from common import ACCENT, MUTED, get_demo_stream, icon, require
from nids.data import BENIGN, CATEGORY_COL

FLOWS_PER_SECOND = 50  # replay speed, as the live monitor's default

stream = get_demo_stream()
require(stream is not None, "The replay stream is created during training.", "python -m nids.train")

flows = stream[CATEGORY_COL].astype(str).to_numpy()
seconds = [flows[i:i + FLOWS_PER_SECOND] for i in range(0, len(flows), FLOWS_PER_SECOND)]
sequence = [{c: int((s == c).sum()) for c in np.unique(s) if c != BENIGN} for s in seconds]

# The map fills the window behind everything; the logo and name float above it.
st.html(f"""
<style>
.map-hero {{ position: fixed; top: 34px; left: calc(var(--map-left, 0px) + (100vw - var(--map-left, 0px)) / 2); transform: translateX(-50%); z-index: 5;
             text-align: center; pointer-events: none; text-shadow: 0 0 12px #0A0D12, 0 0 4px #0A0D12; }}
.map-hero img {{ filter: drop-shadow(0 0 18px rgba(56, 198, 232, 0.45)); }}
.map-hero b {{ display: block; font-family: 'Barlow Semi Condensed', Barlow, sans-serif; font-size: 2.6rem;
               font-weight: 700; letter-spacing: 0.02em; color: #E6EDF5; margin-top: 6px; }}
.map-hero span {{ display: block; color: {MUTED}; font-size: 1.05rem; margin-top: 2px; }}
.map-foot {{ position: fixed; bottom: 18px; left: calc(var(--map-left, 0px) + (100vw - var(--map-left, 0px)) / 2); transform: translateX(-50%); z-index: 5;
             width: min(760px, 90vw); text-align: center; color: {MUTED}; font-size: 0.8rem; }}
</style>
<div class="map-hero">{icon("brand", 64, ACCENT)}<b>AI-NIDS</b>
<span>AI-Based Network Intrusion Detection System Using Machine Learning</span></div>
<div class="map-foot">Attack bursts from held-out CIC-IDS2017 traffic, replayed at {FLOWS_PER_SECOND} flows per
second and drawn as arcs to the protected network. Arc origins are random: the dataset records no attacker
locations.</div>
""")
attack_map(0, {}, key="attack_map_page", sequence=sequence, fullscreen=True)
