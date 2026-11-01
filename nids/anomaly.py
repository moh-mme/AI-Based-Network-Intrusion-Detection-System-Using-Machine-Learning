"""Anomaly detectors trained on benign traffic only, and conformal calibration.

A supervised classifier can only recognise the attack types it was trained on.
An anomaly detector learns what normal traffic looks like and flags whatever
lies far from it, so it can catch attacks that nobody labelled. Conformal
p-values put any detector's scores on one scale with a known false-alarm rate.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from .models import preprocessing

DISPLAY_NAMES = {"isolation_forest": "Isolation Forest", "autoencoder": "Autoencoder"}


class IsolationForestDetector:
    """Benign flows need many random splits to isolate; outliers need few."""

    def __init__(self, random_state: int, **params):
        self.prep = Pipeline(preprocessing(scale=True))
        self.model = IsolationForest(n_jobs=-1, random_state=random_state, **params)

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(self.prep.fit_transform(X))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher means more anomalous."""
        return -self.model.score_samples(self.prep.transform(X))


class AutoencoderDetector:
    """A network trained to reproduce benign flows through a narrow bottleneck.

    It learns to rebuild normal traffic only, so attacks come back with a large
    reconstruction error.
    """

    def __init__(self, random_state: int, **params):
        params["hidden_layer_sizes"] = tuple(params.get("hidden_layer_sizes", (32, 8, 32)))
        self.prep = Pipeline(preprocessing(scale=True))
        self.model = MLPRegressor(early_stopping=True, random_state=random_state, **params)

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        Z = self.prep.fit_transform(X)
        self.model.fit(Z, Z)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher means more anomalous (mean squared reconstruction error)."""
        Z = self.prep.transform(X)
        return ((self.model.predict(Z) - Z) ** 2).mean(axis=1)


_DETECTORS = {"isolation_forest": IsolationForestDetector, "autoencoder": AutoencoderDetector}


def build_detector(name: str, params: dict | None, random_state: int):
    if name not in _DETECTORS:
        raise ValueError(f"Unknown anomaly detector '{name}'. Available: {', '.join(_DETECTORS)}")
    return _DETECTORS[name](random_state, **dict(params or {}))


def conformal_pvalues(calibration_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Share of benign calibration flows scoring at least as high as each flow.

    Scores must grow with suspicion. Flagging flows with p <= alpha raises alarms
    on at most a share alpha of benign traffic that resembles the calibration
    set (the conformal guarantee), whatever detector produced the scores.
    """
    calibration = np.sort(np.asarray(calibration_scores))
    at_least = len(calibration) - np.searchsorted(calibration, scores, side="left")
    return (1 + at_least) / (len(calibration) + 1)
