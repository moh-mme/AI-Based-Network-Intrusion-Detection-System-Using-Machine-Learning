"""Loading and cleaning of the CIC-IDS2017 flow CSV files."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

LABEL_COL = "Label"           # original label, e.g. "DoS Hulk"
CATEGORY_COL = "Category"     # grouped attack category, e.g. "DoS"
ATTACK_COL = "is_attack"      # 0 = benign, 1 = attack
DAY_COL = "Day"               # capture day, from the file name, e.g. "Wednesday"
META_COLS = (LABEL_COL, CATEGORY_COL, ATTACK_COL, DAY_COL)  # everything that is not a feature
BENIGN = "Benign"
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# A raw-data directory containing this file holds generated test data, not the
# real dataset. The flag travels through to the trained models and the dashboard.
SYNTHETIC_MARKER = "SYNTHETIC_DATA.txt"


def normalize_label(label) -> str:
    """Matching key for a raw label: 'Web Attack \\x96 XSS' -> 'web attack xss'."""
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def display_label(label) -> str:
    """Readable label with mis-decoded dashes repaired: 'Web Attack - XSS'."""
    label = re.sub(r"[^\x20-\x7e]+", "-", str(label))
    return re.sub(r"\s+", " ", label).strip()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def capture_day(path: Path) -> str:
    """'Wednesday-workingHours.pcap_ISCX.csv' -> 'Wednesday'; else the name's first part."""
    name = path.name.lower()
    return next((d for d in WEEKDAYS if d.lower() in name), path.name.split("-")[0])


def chronological(path: Path) -> tuple:
    """Sort key putting the week's files in capture order (Monday first)."""
    day = capture_day(path)
    name = path.name.lower()
    return (WEEKDAYS.index(day) if day in WEEKDAYS else len(WEEKDAYS), "afternoon" in name, name)


def cap_per_class(
    labels: np.ndarray,
    cap: int | None | Callable[[str], int | None],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sorted positions into `labels` keeping at most `cap` random rows per class.

    `cap` is a number, None (keep everything) or a function of the class name.
    """
    keep = []
    for cls in pd.unique(labels):
        positions = np.flatnonzero(labels == cls)
        limit = cap(cls) if callable(cap) else cap
        if limit is not None and len(positions) > limit:
            positions = rng.choice(positions, size=limit, replace=False)
        keep.append(positions)
    return np.sort(np.concatenate(keep))


def find_csv_files(raw_dir: Path) -> list[Path]:
    """The raw CSVs in capture order, so de-duplication keeps a flow's first sighting."""
    files = sorted(Path(raw_dir).rglob("*.csv"), key=chronological)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Download the CIC-IDS2017 "
            "'MachineLearningCSV' files and extract them there (see README.md)."
        )
    return files


def read_flow_csv(path: Path) -> pd.DataFrame:
    # The Thursday web-attack file stores its labels' en-dashes as cp1252 bytes,
    # which are invalid UTF-8. latin-1 decodes every byte, and display_label()
    # repairs the resulting characters.
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()  # the original headers have leading spaces
    return df


def map_categories(labels: pd.Series, category_map: dict) -> pd.Series:
    lookup = {normalize_label(k): v for k, v in category_map.items()}
    mapping = {label: lookup.get(normalize_label(label)) for label in labels.unique()}
    unmapped = sorted(label for label, category in mapping.items() if category is None)
    if unmapped:
        raise ValueError(
            f"Labels without a category: {unmapped}. "
            "Add them to data.category_map in config.yaml."
        )
    return labels.map(mapping)


def clean_frame(
    df: pd.DataFrame, drop_columns: list[str], category_map: dict
) -> tuple[pd.DataFrame, dict]:
    """Numeric float32 features plus Label/Category; rows with NaN/inf removed."""
    if LABEL_COL not in df.columns:
        raise ValueError(f"No '{LABEL_COL}' column. Found: {list(df.columns)[:10]}...")
    rows_read = len(df)
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    df = df.dropna(subset=[LABEL_COL])  # some files end with thousands of empty rows
    rows_empty = rows_read - len(df)

    labels = df.pop(LABEL_COL)
    labels = labels.map({raw: display_label(raw) for raw in labels.unique()})
    features = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = features.notna().all(axis=1).to_numpy()

    out = features.loc[valid].astype(np.float32).reset_index(drop=True)
    out[LABEL_COL] = labels.to_numpy()[valid]
    out[CATEGORY_COL] = map_categories(out[LABEL_COL], category_map)
    stats = {"rows_read": rows_read, "rows_empty": rows_empty, "rows_invalid": int((~valid).sum())}
    return out, stats


def build_dataset(cfg: Config) -> dict:
    """Clean every raw CSV into one parquet dataset and write its profile."""
    paths = cfg.paths
    files = find_csv_files(paths.raw_dir)
    drop_columns = cfg.data.get("drop_columns") or []
    category_map = cfg.data["category_map"]

    frames, file_stats, expected = [], [], None
    for path in files:
        log.info("Reading %s", path.name)
        frame, stats = clean_frame(read_flow_csv(path), drop_columns, category_map)
        if expected is None:
            expected = frame.columns
        elif set(frame.columns) != set(expected):
            raise ValueError(
                f"{path.name} has different columns than {files[0].name}. Do not mix "
                "the 'MachineLearningCSV' and 'GeneratedLabelledFlows' variants."
            )
        frame = frame[expected]
        frame[DAY_COL] = capture_day(path)
        log.info("  %d rows kept, %d with NaN/inf removed", len(frame), stats["rows_invalid"])
        frames.append(frame)
        file_stats.append({"file": path.name, "rows_kept": len(frame), **stats})

    df = pd.concat(frames, ignore_index=True)
    del frames
    rows_clean = len(df)
    if cfg.data.get("drop_duplicates", True):
        log.info("Removing duplicate flows from %d rows", rows_clean)
        # A flow repeated on a later day is still a duplicate: compare everything but the day.
        df = df.drop_duplicates(subset=[c for c in df.columns if c != DAY_COL], ignore_index=True)
    for column in (LABEL_COL, CATEGORY_COL, DAY_COL):
        df[column] = df[column].astype("category")
    df[ATTACK_COL] = (df[CATEGORY_COL] != BENIGN).astype(np.int8)

    features = feature_columns(df)
    stats = df[features].describe().T
    synthetic = (paths.raw_dir / SYNTHETIC_MARKER).exists()
    profile = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": "synthetic" if synthetic else "CIC-IDS2017",
        "files": file_stats,
        "rows_read": sum(s["rows_read"] for s in file_stats),
        "rows_empty": sum(s["rows_empty"] for s in file_stats),
        "rows_invalid": sum(s["rows_invalid"] for s in file_stats),
        "rows_clean": rows_clean,
        "duplicates_removed": rows_clean - len(df),
        "rows_final": len(df),
        "attack_ratio": float(df[ATTACK_COL].mean()),
        "features": features,
        "constant_features": stats.index[stats["min"] == stats["max"]].tolist(),
        "feature_stats": stats[["mean", "std", "min", "max"]].astype(float).to_dict(orient="index"),
        "label_counts": _counts(df[LABEL_COL]),
        "category_counts": _counts(df[CATEGORY_COL]),
        "day_category_counts": {
            str(day): _counts(group[CATEGORY_COL])
            for day, group in df.groupby(DAY_COL, observed=True, sort=False)
        },
        "label_to_category": dict(
            df[[LABEL_COL, CATEGORY_COL]].drop_duplicates().astype(str).itertuples(index=False)
        ),
    }

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing %s", paths.dataset)
    df.to_parquet(paths.dataset, index=False)

    per_category = cfg.data.get("explorer_sample_per_category", 5000)
    sample = cap_per_class(df[CATEGORY_COL].to_numpy(), per_category, np.random.default_rng(0))
    df.iloc[sample].to_parquet(paths.explorer_sample, index=False)

    paths.profile.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().items() if v > 0}


def load_dataset(cfg: Config) -> pd.DataFrame:
    if not cfg.paths.dataset.exists():
        raise FileNotFoundError(
            f"{cfg.paths.dataset} not found. Run `python -m nids.prepare` first."
        )
    return pd.read_parquet(cfg.paths.dataset)


def load_profile(cfg: Config) -> dict | None:
    if not cfg.paths.profile.exists():
        return None
    return json.loads(cfg.paths.profile.read_text(encoding="utf-8"))
