"""Research experiments on the synthetic dataset, plus their statistical building blocks."""

from pathlib import Path

import numpy as np

from nids.anomaly import conformal_pvalues
from nids.config import load_config
from nids.data import capture_day, chronological
from nids.experiments import EXPERIMENTS, run_experiments
from tests.synthetic import CATEGORIES


def test_conformal_pvalues_keep_the_false_alarm_budget():
    rng = np.random.default_rng(0)
    p = conformal_pvalues(rng.normal(size=5000), rng.normal(size=50000))
    assert abs((p <= 0.05).mean() - 0.05) < 0.01
    # (1 + calibration scores >= 10) / (3 + 1)
    assert conformal_pvalues(np.array([1.0, 2.0, 3.0]), np.array([10.0]))[0] == 0.25


def test_files_are_read_in_capture_order():
    names = ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Monday-WorkingHours.pcap_ISCX.csv",
             "Friday-WorkingHours-Morning.pcap_ISCX.csv"]
    ordered = [p.name for p in sorted(map(Path, names), key=chronological)]
    assert ordered == [names[1], names[2], names[0]]
    assert capture_day(Path("Wednesday-workingHours.pcap_ISCX.csv")) == "Wednesday"


def test_prepare_records_the_capture_day(profile):
    assert set(profile["day_category_counts"]) == {"Day1", "Day2", "Day3"}


def test_experiments_end_to_end(pipeline_config, profile):
    cfg = load_config(pipeline_config)
    results = run_experiments(cfg)

    for name in EXPERIMENTS:
        assert name in results
    loao = results["loao"]["result"]["rows"]
    assert {r["held_out"] for r in loao} == set(CATEGORIES) - {"Benign"}
    calibrated = [r for r in loao if r["detector"] != "Supervised, threshold 0.5"]
    # Alarm budget is 1%; the test sets hold ~1,500 benign flows.
    assert all(r["fpr"] <= 0.05 for r in calibrated)
    assert {r["features"] for r in loao} == set(cfg.research["feature_sets"])

    temporal = results["temporal"]["result"]["rows"]
    assert {r["calibration"] for r in temporal} == {None, "training data", "recent benign"}

    shap = results["shap"]["result"]["all"]["by_category"]
    for entry in shap.values():
        assert set(entry["shortcut_shares"]) <= set(results["settings"]["shortcut_features"])
        assert 0 <= sum(entry["shortcut_shares"].values()) <= 1
    out = cfg.paths.research_reports
    assert (out / "summary.md").exists()
    for figure in ("headline.png", "loao_heatmap.png", "overconfidence.png", "shap_shortcut_share.png"):
        assert (out / "figures" / figure).exists()
