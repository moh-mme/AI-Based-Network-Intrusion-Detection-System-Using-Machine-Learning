"""End-to-end run of prepare -> train -> predict on the synthetic dataset."""

import json

import pandas as pd
import pytest

from nids.config import TASKS, load_config
from nids.data import find_csv_files, read_flow_csv
from nids.predict import IntrusionDetector
from nids.predict import main as predict_main
from tests.synthetic import CATEGORIES, N_INVALID


def test_prepare_cleans_and_profiles(profile):
    assert profile["data_source"] == "synthetic"
    assert profile["rows_invalid"] == N_INVALID
    assert profile["rows_empty"] == 5
    assert profile["duplicates_removed"] >= 100
    assert set(profile["category_counts"]) == set(CATEGORIES)
    assert "Fwd Header Length.1" not in profile["features"]
    assert len(profile["features"]) == 77
    assert "Bwd PSH Flags" in profile["constant_features"]
    assert "Web Attack - XSS" in profile["label_counts"]


def test_training_writes_models_and_reports(pipeline_config, trained):
    cfg = load_config(pipeline_config)
    assert set(trained) == set(TASKS)
    for task in TASKS:
        metadata = json.loads((cfg.paths.task_models(task) / "metadata.json").read_text())
        report = json.loads((cfg.paths.task_reports(task) / "metrics.json").read_text())
        figures = cfg.paths.task_reports(task) / "figures"

        assert metadata["data_source"] == "synthetic"
        assert metadata["best_model"] in metadata["models"]
        assert len(metadata["models"]) == 6
        for name in metadata["models"]:
            assert (cfg.paths.task_models(task) / f"{name}.joblib").exists()
            assert (figures / f"confusion_matrix_{name}.png").exists()
        assert (figures / "model_comparison.png").exists()
        assert (figures / "roc_curves.png").exists()
        assert (cfg.paths.task_reports(task) / "results.md").exists()
        # The synthetic categories are well separated; anything less means a broken pipeline.
        assert report["models"][metadata["best_model"]]["metrics"]["accuracy"] > 0.95

    assert len(pd.read_parquet(cfg.paths.demo_stream)) == 1000


@pytest.mark.parametrize("task", TASKS)
def test_detector_classifies_raw_csv(pipeline_config, trained, task):
    cfg = load_config(pipeline_config)
    flows = read_flow_csv(find_csv_files(cfg.paths.raw_dir)[0])  # padded headers, inf values
    detector = IntrusionDetector(task, "best", cfg)
    result = detector.predict(flows)

    assert len(result) == len(flows)
    assert set(result["prediction"]) <= set(detector.classes)
    assert result["confidence"].between(0, 1).all()


def test_detector_reports_missing_columns(pipeline_config, trained):
    detector = IntrusionDetector("binary", "best", load_config(pipeline_config))
    with pytest.raises(ValueError, match="missing"):
        detector.predict(pd.DataFrame({"Flow Duration": [1.0]}))


def test_predict_cli_writes_output(pipeline_config, trained, tmp_path, capsys):
    cfg = load_config(pipeline_config)
    output = tmp_path / "predictions.csv"
    predict_main(["--input", str(find_csv_files(cfg.paths.raw_dir)[2]), "--output", str(output),
                  "--config", str(pipeline_config)])

    written = pd.read_csv(output)
    assert {"prediction", "confidence", "is_attack"} <= set(written.columns)
    assert "Accuracy against the file's labels" in capsys.readouterr().out
