from pathlib import Path

import pytest
import yaml

from nids.config import DEFAULT_CONFIG, load_config
from nids.data import build_dataset
from nids.train import run_training
from tests.synthetic import write_synthetic_dataset

# Small settings so the whole pipeline runs in seconds on the synthetic data.
FAST_MODELS = {
    "logistic_regression": {"max_iter": 300},
    "decision_tree": {},
    "random_forest": {"n_estimators": 20},
    "knn": {"n_neighbors": 5, "max_train_rows": 2000},
    "xgboost": {"n_estimators": 30, "max_depth": 4},
    "mlp": {"hidden_layer_sizes": [32], "max_iter": 30},
}


@pytest.fixture(scope="session")
def pipeline_config(tmp_path_factory) -> Path:
    """A config whose paths point at a temporary project with synthetic raw data."""
    root = tmp_path_factory.mktemp("project")
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["models"] = FAST_MODELS
    raw["training"]["models"] = list(FAST_MODELS)
    raw["live_monitor"]["stream_rows"] = 1000
    raw["research"].update({
        "temporal": {"train_days": ["Day1", "Day2"], "test_days": ["Day3"], "recent_calibration_benign": 300},
        "calibration_benign": 500,
        "anomaly_train_benign": 2000,
        "max_test_benign": 1500,
        "anomaly_detectors": {"isolation_forest": {"n_estimators": 50},
                              "autoencoder": {"hidden_layer_sizes": [16, 4, 16], "max_iter": 20}},
        "shap_flows_per_category": 100,
    })
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    write_synthetic_dataset(load_config(path).paths.raw_dir)
    return path


@pytest.fixture(scope="session")
def profile(pipeline_config) -> dict:
    return build_dataset(load_config(pipeline_config))


@pytest.fixture(scope="session")
def trained(pipeline_config, profile) -> dict:
    return run_training(load_config(pipeline_config))
