"""Generator of small fake CIC-IDS2017 files, for tests only.

The files copy the real dataset's quirks (padded headers, a duplicated column,
cp1252 en-dashes in labels, inf/NaN values, duplicate and empty rows) so the
whole pipeline can be exercised without the 1 GB download. The flows
themselves are random numbers: results on them say nothing about real traffic.

    python -m tests.synthetic OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nids.data import SYNTHETIC_MARKER

FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min",
]
# Always zero in the real dataset.
CONSTANT = {"Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
            "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"}

# category: (rows, raw labels, destination ports)
CATEGORIES = {
    "Benign": (6000, ["BENIGN"], [80, 443, 53, 123, 8080]),
    "DoS": (1500, ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"], [80]),
    "DDoS": (1000, ["DDoS"], [80]),
    "PortScan": (1000, ["PortScan"], list(range(1, 1024))),
    "BruteForce": (400, ["FTP-Patator", "SSH-Patator"], [21, 22]),
    "WebAttack": (200, ["Web Attack – Brute Force", "Web Attack – XSS",
                        "Web Attack – Sql Injection"], [80]),
    "Botnet": (150, ["Bot"], [8080]),
    "Infiltration": (40, ["Infiltration"], [444]),
}
N_INVALID = 40  # rows given an inf or NaN value


def make_flows(seed: int = 7, scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(3, 2, size=len(FEATURES))
    frames = []
    for category, (rows, labels, ports) in CATEGORIES.items():
        rows = max(int(rows * scale), 12)
        offset = rng.normal(0, 1.2, size=len(FEATURES))
        values = np.exp(rng.normal(base + offset, 1.0, size=(rows, len(FEATURES)))).round(2)
        frame = pd.DataFrame(values, columns=range(len(FEATURES)))
        frame[0] = rng.choice(ports, size=rows)
        for i, name in enumerate(FEATURES):
            if name in CONSTANT:
                frame[i] = 0
        frame["Label"] = rng.choice(labels, size=rows)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    bytes_col, packets_col = FEATURES.index("Flow Bytes/s"), FEATURES.index("Flow Packets/s")
    invalid = rng.choice(len(df), size=N_INVALID, replace=False)
    df.loc[invalid[: N_INVALID // 2], bytes_col] = np.inf
    df.loc[invalid[N_INVALID // 2:], packets_col] = np.nan
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def write_synthetic_dataset(out_dir: Path, seed: int = 7, scale: float = 1.0) -> list[Path]:
    """Write three day-files in the MachineLearningCSV layout and the synthetic marker."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = make_flows(seed, scale)
    finite = np.isfinite(df.drop(columns="Label").to_numpy(float)).all(axis=1)
    df = pd.concat([df, df[finite].iloc[:100]], ignore_index=True)  # exact duplicate flows
    # Real headers carry a leading space, and "Fwd Header Length" appears twice.
    df.columns = [f" {FEATURES[c]}" if isinstance(c, int) else f" {c}" for c in df.columns]

    paths = []
    for i, part in enumerate(np.array_split(np.arange(len(df)), 3)):
        path = out_dir / f"Day{i + 1}-WorkingHours.pcap_ISCX.csv"
        df.iloc[part].to_csv(path, index=False, encoding="cp1252")
        if i == 2:  # trailing empty rows, as in the real Thursday file
            with open(path, "a", encoding="cp1252") as f:
                f.write(("," * (len(df.columns) - 1) + "\n") * 5)
        paths.append(path)
    (out_dir / SYNTHETIC_MARKER).write_text(
        "Generated by tests/synthetic.py. These are NOT real network flows.\n", encoding="utf-8"
    )
    return paths


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for written in write_synthetic_dataset(Path(sys.argv[1])):
        print(written)
