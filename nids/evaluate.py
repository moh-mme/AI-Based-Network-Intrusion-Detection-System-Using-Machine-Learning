"""Evaluation metrics and the figures saved for the project report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1 (macro)",
    "f1_weighted": "F1 (weighted)",
    "mcc": "MCC",
    "detection_rate": "Detection rate",
    "false_alarm_rate": "False alarm rate",
    "roc_auc": "ROC AUC",
}

# Evaluated on a grid that is dense near FPR = 0, where an IDS operates.
_ROC_GRID = np.unique(np.concatenate([[0.0], np.logspace(-5, 0, 120), np.linspace(0, 1, 101)]))


def evaluate(y_true: np.ndarray, proba: np.ndarray, classes: list[str], benign_index: int = 0) -> dict:
    """Metrics, confusion matrix, per-class report and ROC curve for one model."""
    y_pred = proba.argmax(axis=1)
    labels = np.arange(len(classes))
    metrics = {"accuracy": accuracy_score(y_true, y_pred), "mcc": matthews_corrcoef(y_true, y_pred)}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        )
        metrics.update({f"precision_{average}": p, f"recall_{average}": r, f"f1_{average}": f})

    # Attack-vs-benign view, meaningful for both tasks: a DoS flow classified as
    # PortScan is still a detected attack.
    true_attack = y_true != benign_index
    pred_attack = y_pred != benign_index
    metrics["detection_rate"] = _ratio((true_attack & pred_attack).sum(), true_attack.sum())
    metrics["false_alarm_rate"] = _ratio((~true_attack & pred_attack).sum(), (~true_attack).sum())

    roc = None
    if 0 < true_attack.sum() < len(true_attack):
        attack_score = 1.0 - proba[:, benign_index]
        metrics["roc_auc"] = roc_auc_score(true_attack, attack_score)
        fpr, tpr, _ = roc_curve(true_attack, attack_score)
        roc = {"fpr": _ROC_GRID.tolist(), "tpr": np.interp(_ROC_GRID, fpr, tpr).tolist()}

    return {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0
        ),
        "roc": roc,
    }


def _ratio(numerator, denominator) -> float:
    return float(numerator / denominator) if denominator else 0.0


def results_table(results: dict[str, dict], display_names: dict[str, str]) -> str:
    """Markdown comparison table, ready to paste into the report."""
    columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
               "detection_rate", "false_alarm_rate", "roc_auc"]
    header = ["Model", *(METRIC_LABELS[c] for c in columns), "Train time (s)", "Predict (ms / 1k flows)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, result in results.items():
        m = result["metrics"]
        cells = [display_names.get(name, name)]
        cells += [f"{m[c]:.4f}" if c in m else "-" for c in columns]
        cells += [f"{result['fit_seconds']:.1f}", f"{result['predict_ms_per_1k']:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_model_comparison(results: dict[str, dict], display_names: dict[str, str], path: Path) -> None:
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    names = list(results)
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(names)), 4.5), layout="constrained")
    for i, metric in enumerate(metrics):
        values = [results[n]["metrics"][metric] for n in names]
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, values, width, label=METRIC_LABELS[metric])
    ax.set_xticks(x, [display_names.get(n, n) for n in names], rotation=15, ha="right")
    lowest = min(results[n]["metrics"][m] for n in names for m in metrics)
    ax.set_ylim(max(0.0, lowest - 0.05), 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the held-out test set")
    ax.legend(ncols=len(metrics), loc="lower center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], classes: list[str], title: str, path: Path) -> None:
    cm = np.asarray(cm, dtype=float)
    row_share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(4.5, 0.75 * len(classes) + 2)
    fig, ax = plt.subplots(figsize=(size + 1, size), layout="constrained")
    image = ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] == 0:
                continue
            color = "white" if row_share[i, j] > 0.5 else "black"
            ax.text(j, i, f"{row_share[i, j]:.1%}\n({int(cm[i, j]):,})", ha="center", va="center",
                    fontsize=7, color=color)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Share of actual class")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(results: dict[str, dict], display_names: dict[str, str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    for name, result in results.items():
        if result["roc"]:
            auc = result["metrics"]["roc_auc"]
            ax.plot(result["roc"]["fpr"], result["roc"]["tpr"],
                    label=f"{display_names.get(name, name)} (AUC {auc:.4f})")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False alarm rate (log scale)")
    ax.set_ylabel("Detection rate")
    ax.set_title("ROC curves: attack vs benign")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance: list[dict], title: str, path: Path, top: int = 20) -> None:
    items = importance[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(items) + 1.2), layout="constrained")
    ax.barh([i["feature"] for i in items], [i["importance"] for i in items], color="#2a7ab9")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)
