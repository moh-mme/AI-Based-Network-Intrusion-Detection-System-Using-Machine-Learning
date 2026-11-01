"""Step 3: detect intrusions in new flow records (CICFlowMeter / CIC-IDS2017 format).

    python -m nids.predict --input flows.csv [--output predictions.csv]
                           [--task multiclass] [--model best]
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .config import TASKS, Config, load_config
from .data import BENIGN, LABEL_COL, map_categories, read_flow_csv
from .models import predict_proba


def load_metadata(cfg: Config, task: str) -> dict | None:
    path = cfg.paths.task_models(task) / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class IntrusionDetector:
    """A trained model plus everything needed to apply it to raw flow records."""

    def __init__(self, task: str = "multiclass", model: str = "best", cfg: Config | None = None):
        cfg = cfg or load_config()
        metadata = load_metadata(cfg, task)
        if metadata is None:
            raise FileNotFoundError(
                f"No trained {task} models in {cfg.paths.task_models(task)}. "
                "Run `python -m nids.train` first."
            )
        name = metadata["best_model"] if model == "best" else model
        if name not in metadata["models"]:
            raise ValueError(f"Model '{name}' was not trained. Available: {', '.join(metadata['models'])}")

        self.task = task
        self.model_name = name
        self.display_name = metadata["models"][name]
        self.metadata = metadata
        self.features: list[str] = metadata["features"]
        self.classes = np.asarray(metadata["classes"])
        self.pipeline = joblib.load(cfg.paths.task_models(task) / f"{name}.joblib")

    def to_matrix(self, flows: pd.DataFrame) -> np.ndarray:
        """Select the model's features; unparseable or infinite values get the training median."""
        flows = flows.rename(columns=lambda c: str(c).strip())
        missing = [f for f in self.features if f not in flows.columns]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} of the {len(self.features)} feature columns "
                f"the model needs, e.g. {missing[:5]}. Flows must be in CICFlowMeter "
                "(CIC-IDS2017) format."
            )
        X = flows[self.features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return X.fillna(self.metadata["fill_values"]).to_numpy(np.float32)

    def predict(self, flows: pd.DataFrame) -> pd.DataFrame:
        proba = predict_proba(self.pipeline, self.to_matrix(flows), len(self.classes))
        best = proba.argmax(axis=1)
        prediction = self.classes[best]
        return pd.DataFrame(
            {
                "prediction": prediction,
                "confidence": proba[np.arange(len(best)), best],
                "is_attack": prediction != BENIGN,
            },
            index=flows.index,
        )


def true_classes(labels: pd.Series, task: str, category_map: dict) -> pd.Series:
    """Ground truth in the model's class names, from a raw Label column."""
    categories = map_categories(labels.astype(str), category_map)
    if task == "binary":
        return categories.where(categories == BENIGN, "Attack")
    return categories


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect intrusions in a CSV of network flows.")
    parser.add_argument("--input", required=True, help="CSV of flows in CICFlowMeter format")
    parser.add_argument("--output", help="where to write the predictions (default: <input>_predictions.csv)")
    parser.add_argument("--task", choices=TASKS, default="multiclass")
    parser.add_argument("--model", default="best", help="model name, or 'best' (default)")
    parser.add_argument("--config", help="path to a config file (default: config.yaml)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    detector = IntrusionDetector(args.task, args.model, cfg)
    flows = read_flow_csv(args.input).dropna(how="all").reset_index(drop=True)
    result = detector.predict(flows)

    output = args.output or str(args.input).rsplit(".", 1)[0] + "_predictions.csv"
    pd.concat([flows, result], axis=1).to_csv(output, index=False)

    print(f"Model: {detector.display_name} ({args.task})")
    print(f"Flows analysed: {len(result):,} | flagged as attacks: {int(result['is_attack'].sum()):,}")
    for cls, count in result["prediction"].value_counts().items():
        print(f"  {cls:<14}{count:>10,}")
    if LABEL_COL in flows.columns:
        try:
            truth = true_classes(flows[LABEL_COL], args.task, cfg.data["category_map"])
            print(f"Accuracy against the file's labels: {(truth == result['prediction']).mean():.4f}")
        except ValueError as error:
            print(f"Labels not compared: {error}")
    print(f"Predictions written to {output}")


if __name__ == "__main__":
    main()
