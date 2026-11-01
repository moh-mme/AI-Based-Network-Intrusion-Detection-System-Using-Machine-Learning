"""Model zoo: every model is a scikit-learn Pipeline of preprocessing + classifier."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "knn": "K-Nearest Neighbors",
    "xgboost": "XGBoost",
    "mlp": "Neural Network (MLP)",
}

# Distance- and gradient-based models need comparable feature scales; tree
# models split on thresholds and are unaffected by scaling.
SCALE_SENSITIVE = {"logistic_regression", "knn", "mlp"}


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Compress the heavy-tailed flow statistics (byte rates span ~10 orders of magnitude)."""
    return np.sign(x) * np.log1p(np.abs(x))


def _estimator(name: str, params: dict, random_state: int):
    if name == "logistic_regression":
        return LogisticRegression(random_state=random_state, **params)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state, **params)
    if name == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=random_state, **params)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=-1, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier  # heavy import, only needed when training it

        return XGBClassifier(tree_method="hist", n_jobs=-1, random_state=random_state, **params)
    if name == "mlp":
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        return MLPClassifier(early_stopping=True, random_state=random_state, **params)
    raise ValueError(f"Unknown model '{name}'. Available: {', '.join(DISPLAY_NAMES)}")


def preprocessing(scale: bool) -> list[tuple]:
    """Pipeline steps shared by every model: drop constant features, optionally log-scale."""
    steps = [("drop_constant", VarianceThreshold(threshold=0.0))]
    if scale:
        steps += [("log", FunctionTransformer(signed_log1p)), ("scale", StandardScaler())]
    return steps


def build_pipeline(name: str, params: dict | None, random_state: int) -> tuple[Pipeline, int | None]:
    """The model's pipeline and its optional training-row cap (`max_train_rows`)."""
    params = dict(params or {})
    max_train_rows = params.pop("max_train_rows", None)
    steps = preprocessing(scale=name in SCALE_SENSITIVE)
    steps.append(("clf", _estimator(name, params, random_state)))
    return Pipeline(steps), max_train_rows


def predict_proba(pipeline: Pipeline, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Class probabilities with one column per class, even for classes the model never saw."""
    proba = pipeline.predict_proba(X)
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((len(X), n_classes), dtype=proba.dtype)
    full[:, pipeline.classes_] = proba
    return full


def feature_importance(pipeline: Pipeline, features: list[str]) -> list[dict] | None:
    """(feature, importance) pairs, highest first, for tree-based models."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    kept = np.asarray(features)[pipeline.named_steps["drop_constant"].get_support()]
    order = np.argsort(clf.feature_importances_)[::-1]
    return [
        {"feature": str(kept[i]), "importance": float(clf.feature_importances_[i])}
        for i in order
    ]
