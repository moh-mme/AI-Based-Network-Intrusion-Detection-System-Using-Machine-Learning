"""Step 2: train and evaluate every configured model.

    python -m nids.train [--config CONFIG] [--tasks binary multiclass]
                         [--models random_forest xgboost ...] [--quick]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import setup_logging
from .config import TASKS, Config, load_config
from .data import BENIGN, CATEGORY_COL, cap_per_class, feature_columns, load_dataset, load_profile
from .evaluate import evaluate as evaluate_model
from .evaluate import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curves,
    results_table,
)
from .models import DISPLAY_NAMES, build_pipeline, feature_importance, predict_proba

log = logging.getLogger(__name__)

BINARY_CLASSES = [BENIGN, "Attack"]
# --quick: small samples so a full run finishes in a few minutes.
QUICK_OVERRIDES = {"max_train_per_class": 20000, "max_benign_train": 50000, "max_test_rows": 100000}


def class_order(categories) -> list[str]:
    """Benign first, then the attack categories alphabetically."""
    return [BENIGN] + sorted(c for c in set(categories) if c != BENIGN)


def encode_targets(task: str, categories: np.ndarray, multiclass_classes: list[str]) -> np.ndarray:
    if task == "binary":
        return (categories != BENIGN).astype(np.int64)
    return pd.Categorical(categories, categories=multiclass_classes).codes.astype(np.int64)


def stratified_subset(positions: np.ndarray, y: np.ndarray, size: int | None, seed: int) -> np.ndarray:
    """At most `size` of `positions`, keeping the class proportions of `y`."""
    if size is None or len(positions) <= size:
        return positions
    try:
        subset, _ = train_test_split(positions, train_size=size, stratify=y, random_state=seed)
    except ValueError:  # a class too small to stratify
        subset = np.random.default_rng(seed).choice(positions, size=size, replace=False)
    return np.sort(subset)


def split_indices(categories: np.ndarray, tcfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test split; only the training side is undersampled."""
    seed = tcfg["random_state"]
    train_idx, test_idx = train_test_split(
        np.arange(len(categories)), test_size=tcfg.get("test_size", 0.2),
        stratify=categories, random_state=seed,
    )
    cap = tcfg.get("max_train_per_class")
    benign_cap = tcfg.get("max_benign_train", cap)
    keep = cap_per_class(
        categories[train_idx], lambda c: benign_cap if c == BENIGN else cap, np.random.default_rng(seed)
    )
    train_idx = np.sort(train_idx[keep])
    test_idx = np.sort(test_idx)
    test_idx = stratified_subset(test_idx, categories[test_idx], tcfg.get("max_test_rows"), seed)
    return train_idx, test_idx


def save_demo_stream(test_df: pd.DataFrame, path, n_rows: int, seed: int) -> None:
    """Held-out flows for the live monitor, re-ordered so attacks arrive in bursts."""
    rng = np.random.default_rng(seed)
    categories = test_df[CATEGORY_COL].astype(str).to_numpy()
    attack_types = [c for c in np.unique(categories) if c != BENIGN]

    bursts = []
    per_type = int(n_rows * 0.3) // max(len(attack_types), 1)
    for attack in attack_types:
        positions = rng.permutation(np.flatnonzero(categories == attack))[:per_type]
        while len(positions):
            size = int(rng.integers(20, 150))
            bursts.append(positions[:size])
            positions = positions[size:]
    rng.shuffle(bursts)

    benign = rng.permutation(np.flatnonzero(categories == BENIGN))
    benign = benign[: max(n_rows - sum(map(len, bursts)), 0)]
    cuts = np.sort(rng.integers(0, len(benign) + 1, size=len(bursts)))
    order, previous = [], 0
    for cut, burst in zip(cuts, bursts):
        order += [benign[previous:cut], burst]
        previous = cut
    order.append(benign[previous:])
    test_df.iloc[np.concatenate(order)].reset_index(drop=True).to_parquet(path, index=False)


def run_training(
    cfg: Config, tasks: list[str] | None = None, models: list[str] | None = None, quick: bool = False
) -> dict[str, dict]:
    tcfg = {"random_state": 42, **cfg.training}
    if quick:
        tcfg.update(QUICK_OVERRIDES)
    tasks = tasks or tcfg.get("tasks") or list(TASKS)
    models = models or tcfg.get("models") or list(DISPLAY_NAMES)
    for name in models:
        if name not in DISPLAY_NAMES:
            raise ValueError(f"Unknown model '{name}'. Available: {', '.join(DISPLAY_NAMES)}")

    profile = load_profile(cfg) or {}
    log.info("Loading %s", cfg.paths.dataset)
    df = load_dataset(cfg)
    features = feature_columns(df)
    categories = df[CATEGORY_COL].astype(str).to_numpy()

    counts = pd.Series(categories).value_counts()
    rare = counts[counts < tcfg.get("min_class_samples", 10)]
    if len(rare):
        log.warning("Dropping categories with too few rows to split: %s", rare.to_dict())
        keep = ~np.isin(categories, rare.index)
        df, categories = df[keep].reset_index(drop=True), categories[keep]

    train_idx, test_idx = split_indices(categories, tcfg)
    columns = df.columns.get_indexer(features)
    X_train = df.iloc[train_idx, columns].to_numpy(np.float32)
    X_test = df.iloc[test_idx, columns].to_numpy(np.float32)
    cat_train, cat_test = categories[train_idx], categories[test_idx]
    log.info("Train: %d flows | Test: %d flows | Features: %d", len(X_train), len(X_test), len(features))

    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    stream_rows = cfg.live_monitor.get("stream_rows", 20000)
    save_demo_stream(df.iloc[test_idx], cfg.paths.demo_stream, stream_rows, tcfg["random_state"])
    del df

    context = {
        "features": features,
        # Used at inference time to fill missing/invalid values in uploaded flows.
        "fill_values": dict(zip(features, np.median(X_train, axis=0).astype(float))),
        "data_source": profile.get("data_source", "unknown"),
    }
    multiclass_classes = class_order(categories)
    summaries = {}
    for task in tasks:
        classes = BINARY_CLASSES if task == "binary" else multiclass_classes
        summaries[task] = train_task(
            task, cfg, tcfg, models, classes, context,
            X_train, encode_targets(task, cat_train, multiclass_classes),
            X_test, encode_targets(task, cat_test, multiclass_classes),
        )
    return summaries


def train_task(task, cfg, tcfg, models, classes, context, X_train, y_train, X_test, y_test) -> dict:
    model_dir = cfg.paths.task_models(task)
    report_dir = cfg.paths.task_reports(task)
    figure_dir = report_dir / "figures"
    for directory in (model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    seed = tcfg["random_state"]
    features = context["features"]

    results = {}
    for name in models:
        display = DISPLAY_NAMES[name]
        pipeline, max_rows = build_pipeline(name, cfg.models.get(name), seed)
        rows = stratified_subset(np.arange(len(y_train)), y_train, max_rows, seed)
        X_fit, y_fit = (X_train, y_train) if len(rows) == len(y_train) else (X_train[rows], y_train[rows])

        log.info("[%s] Training %s on %d flows", task, display, len(y_fit))
        start = time.perf_counter()
        pipeline.fit(X_fit, y_fit)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        proba = predict_proba(pipeline, X_test, len(classes))
        predict_seconds = time.perf_counter() - start

        result = evaluate_model(y_test, proba, classes)
        importance = feature_importance(pipeline, features)
        result.update(
            display_name=display,
            n_train=len(y_fit),
            fit_seconds=fit_seconds,
            predict_ms_per_1k=predict_seconds * 1e6 / len(y_test),
            feature_importance=importance[:30] if importance else None,
        )
        results[name] = result
        joblib.dump(pipeline, model_dir / f"{name}.joblib", compress=3)

        m = result["metrics"]
        log.info("[%s] %s: accuracy %.4f | F1 macro %.4f | detection %.4f | false alarms %.4f | %.0fs",
                 task, display, m["accuracy"], m["f1_macro"], m["detection_rate"],
                 m["false_alarm_rate"], fit_seconds)

        plot_confusion_matrix(result["confusion_matrix"], classes, f"{display} ({task})",
                              figure_dir / f"confusion_matrix_{name}.png")
        if importance:
            plot_feature_importance(importance, f"{display}: top features ({task})",
                                    figure_dir / f"feature_importance_{name}.png")

    selection = tcfg.get("selection_metric", "f1_macro")
    best = max(results, key=lambda n: results[n]["metrics"][selection])
    created_at = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "task": task,
        "created_at": created_at,
        "data_source": context["data_source"],
        "classes": classes,
        "features": features,
        "fill_values": context["fill_values"],
        "models": {name: DISPLAY_NAMES[name] for name in results},
        "best_model": best,
        "selection_metric": selection,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def class_counts(y):
        return {cls: int((y == i).sum()) for i, cls in enumerate(classes)}

    report = {
        "task": task,
        "created_at": created_at,
        "data_source": context["data_source"],
        "classes": classes,
        "best_model": best,
        "selection_metric": selection,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_class_counts": class_counts(y_train),
        "test_class_counts": class_counts(y_test),
        "models": results,
    }
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    pd.DataFrame(
        {name: {**r["metrics"], "fit_seconds": r["fit_seconds"], "predict_ms_per_1k": r["predict_ms_per_1k"]}
         for name, r in results.items()}
    ).T.rename_axis("model").to_csv(report_dir / "metrics.csv")
    (report_dir / "results.md").write_text(
        f"# {task.capitalize()} classification results\n\n"
        f"Data: {context['data_source']} | train flows: {len(y_train):,} | test flows: {len(y_test):,} | "
        f"best model ({selection}): {DISPLAY_NAMES[best]}\n\n" + results_table(results, DISPLAY_NAMES),
        encoding="utf-8",
    )
    plot_model_comparison(results, DISPLAY_NAMES, figure_dir / "model_comparison.png")
    plot_roc_curves(results, DISPLAY_NAMES, figure_dir / "roc_curves.png")
    log.info("[%s] Best model by %s: %s", task, selection, DISPLAY_NAMES[best])
    return {"best_model": best, "metrics": results[best]["metrics"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the intrusion detection models.")
    parser.add_argument("--config", help="path to a config file (default: config.yaml)")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, help="default: training.tasks in the config")
    parser.add_argument("--models", nargs="+", choices=list(DISPLAY_NAMES),
                        help="default: training.models in the config")
    parser.add_argument("--quick", action="store_true", help="train on small samples for a fast trial run")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    summaries = run_training(cfg, args.tasks, args.models, args.quick)

    print()
    for task, summary in summaries.items():
        m = summary["metrics"]
        print(f"{task:<11} best: {DISPLAY_NAMES[summary['best_model']]:<22} "
              f"F1 macro {m['f1_macro']:.4f} | accuracy {m['accuracy']:.4f} | "
              f"detection {m['detection_rate']:.4f} | false alarms {m['false_alarm_rate']:.4f}")
    print(f"\nModels: {cfg.paths.models_dir}\nReports and figures: {cfg.paths.reports_dir}")


if __name__ == "__main__":
    main()
