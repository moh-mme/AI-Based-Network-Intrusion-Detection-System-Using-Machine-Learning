"""Turn reports/research/results.json into summary.md and figures for the thesis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .models import DISPLAY_NAMES as MODEL_NAMES

SUPERVISED = "Supervised, threshold 0.5"
TRAINING_DATA, RECENT_BENIGN = "training data", "recent benign"  # see nids.experiments


def pct(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}%}"


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def primary(results: dict) -> tuple[str, str]:
    """The model and feature set the headline numbers are reported for."""
    settings = results["settings"]
    model = "xgboost" if "xgboost" in settings["models"] else settings["models"][0]
    features = settings.get("primary_feature_set")
    if features not in settings["feature_sets"]:
        features = next(iter(settings["feature_sets"]))
    return model, features


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def selected(row: dict, model: str, features: str, calibration: str = TRAINING_DATA) -> bool:
    """Rows for one model and feature set; threshold-0.5 rows have no calibration set."""
    return (row["features"] == features and row["model"] in (model, "none")
            and row.get("calibration") in (None, calibration))


def zero_day_summary(rows: list[dict], model: str, features: str,
                     calibration: str = TRAINING_DATA) -> dict[str, dict]:
    """Per detector: unseen-attack detection and false alarms, averaged over scenarios."""
    out: dict[str, dict] = {}
    for row in rows:
        if not selected(row, model, features, calibration):
            continue
        entry = out.setdefault(row["detector"], {"dr": [], "fpr": [], "auc": []})
        entry["dr"].append(row["dr_unseen"])
        entry["fpr"].append(row["fpr"])
        entry["auc"].append(row.get("auc_unseen"))
    return {name: {k: mean(v) for k, v in entry.items()} for name, entry in out.items()}


def best_hybrid(summary: dict[str, dict]) -> tuple[str, dict] | None:
    hybrids = {k: v for k, v in summary.items() if k.startswith("Hybrid") and v["dr"] is not None}
    return max(hybrids.items(), key=lambda kv: kv[1]["dr"]) if hybrids else None


def headline(results: dict) -> list[tuple[str, float | None, float | None]]:
    """(setting, attack detection rate, false alarm rate) from the reported setup to the realistic ones."""
    model, features = primary(results)
    out = []
    labels = results["settings"]["feature_sets"]
    for row in results.get("random", {}).get("result", []):
        if row["task"] == "binary" and row["model"] == model and row["features"] in ("all", features):
            out.append((f"Random split, {labels[row['features']].lower()}", row["detection_rate"],
                        row["false_alarm_rate"]))
    for key, label in (("loao", "Unseen attack family"), ("temporal", "Later days (temporal)")):
        rows = results.get(key, {}).get("result", {}).get("rows", [])
        if not rows:
            continue
        summary = zero_day_summary(rows, model, features)
        if SUPERVISED in summary:
            out.append((f"{label}, supervised", summary[SUPERVISED]["dr"], summary[SUPERVISED]["fpr"]))
        for calibration in (TRAINING_DATA, RECENT_BENIGN):
            best = best_hybrid(zero_day_summary(rows, model, features, calibration))
            if best and any(r.get("calibration") == calibration for r in rows):
                suffix = ", recalibrated on recent benign" if calibration == RECENT_BENIGN else ""
                out.append((f"{label}, {best[0].replace('Hybrid: supervised + ', 'hybrid with ')}{suffix}",
                            best[1]["dr"], best[1]["fpr"]))
    return out


# --- Figures -------------------------------------------------------------------------------

def plot_headline(items: list[tuple], path: Path) -> None:
    labels = [label for label, _, _ in items][::-1]
    rates = [dr or 0 for _, dr, _ in items][::-1]
    alarms = [fpr for _, _, fpr in items][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(items) + 1.3), layout="constrained")
    colors = ["#2B7A9B" if "hybrid" in label else "#8A99AD" if "Random" in label else "#E5484D"
              for label in labels]
    bars = ax.barh(labels, rates, color=colors)
    for bar, value, fpr in zip(bars, rates, alarms):
        ax.text(min(value, 0.98) + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value:.1%}  (false alarms {pct(fpr, 2)})", va="center", fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Share of attack flows detected")
    ax.set_title("Detection rate: reported setup vs realistic evaluations")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loao_heatmap(matrix: np.ndarray, families: list[str], detectors: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.6 * len(detectors) + 2, 0.55 * len(families) + 1.8), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(families)):
        for j in range(len(detectors)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(detectors)), [d.replace("Hybrid: supervised + ", "Hybrid + ") for d in detectors],
                  rotation=25, ha="right")
    ax.set_yticks(range(len(families)), families)
    ax.set_ylabel("Attack family left out of training")
    ax.set_title("Detection of an attack family never seen in training")
    fig.colorbar(image, ax=ax, label="Share detected")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overconfidence(counts: list[int], bins: list[float], confident: float, path: Path) -> None:
    counts = np.asarray(counts, dtype=float)
    share = counts / max(counts.sum(), 1)
    edges = np.asarray(bins)
    edges[0] = 1e-7  # draw the [0, 1e-6) bin on the log axis
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    colors = ["#E5484D" if right <= confident else "#8A99AD" if right <= 0.5 else "#2B7A9B"
              for right in edges[1:]]
    ax.bar(edges[:-1], share, width=np.diff(edges), align="edge", color=colors, edgecolor="white")
    ax.set_xscale("log")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.text(0.5, ax.get_ylim()[1] * 0.92, " threshold 0.5", fontsize=8)
    ax.set_xlabel("Model's P(attack) for attacks from an unseen family (log scale)")
    ax.set_ylabel("Share of flows")
    ax.set_title(f"Red: missed with over {1 - confident:.0%} confidence that the flow is benign")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shortcut_share(by_category: dict, path: Path) -> None:
    categories = list(by_category)
    features = list(dict.fromkeys(f for c in categories for f in by_category[c]["shortcut_shares"]))
    fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
    bottom = np.zeros(len(categories))
    for feature, color in zip(features, ["#2B7A9B", "#F2C14E", "#F08A4B", "#8A99AD"]):
        values = np.array([by_category[c]["shortcut_shares"].get(feature, 0.0) for c in categories])
        ax.bar(categories, values, bottom=bottom, color=color, label=feature)
        bottom += values
    for x, total in enumerate(bottom):
        ax.text(x, total + 0.01, f"{total:.0%}", ha="center", fontsize=8)
    ax.set_ylim(0, max(bottom.max(), 0.1) * 1.25)
    ax.set_ylabel("Share of the decision (TreeSHAP)")
    ax.set_title("How much the model's decision rests on candidate shortcut features")
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Report -----------------------------------------------------------------------------------

def loao_matrix(rows: list[dict], families: list[str], model: str, features: str):
    detectors = []
    rows = [row for row in rows if selected(row, model, features)]
    for row in rows:
        if row["detector"] not in detectors:
            detectors.append(row["detector"])
    order = [SUPERVISED, "Supervised, calibrated"]
    detectors = [d for d in order if d in detectors] + [d for d in detectors if d not in order]
    matrix = np.full((len(families), len(detectors)), np.nan)
    fpr = np.full((len(families), len(detectors)), np.nan)
    for row in rows:
        if row["held_out"] in families:
            i, j = families.index(row["held_out"]), detectors.index(row["detector"])
            matrix[i, j] = np.nan if row["dr_unseen"] is None else row["dr_unseen"]
            fpr[i, j] = np.nan if row["fpr"] is None else row["fpr"]
    return matrix, fpr, detectors


def write_report(results: dict, out_dir: Path) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    settings = results["settings"]
    model, features = primary(results)
    model_name = MODEL_NAMES.get(model, model)
    alpha = settings["target_fpr"]
    lines = [
        "# Research results: realistic evaluation and zero-day detection\n",
        f"Data: {settings['data_source']}. Headline numbers use **{model_name}** "
        f"({settings['feature_sets'][features].lower()}). Alarm budget for calibrated detectors: "
        f"{alpha:.1%} of benign flows. Feature sets compared: "
        f"{'; '.join(settings['feature_sets'].values())}.\n",
    ]
    quick = [name for name in ("random", "temporal", "loao", "shap") if results.get(name, {}).get("quick")]
    if quick:
        lines.append(f"> Quick run (small samples) for: {', '.join(quick)}. Rerun without --quick for final numbers.\n")

    items = headline(results)
    if items:
        plot_headline(items, figures / "headline.png")
        lines += ["## 1. Reported vs realistic detection\n",
                  table(["Setting", "Attacks detected", "False alarms"],
                        [[label, pct(dr), pct(fpr, 2)] for label, dr, fpr in items]),
                  "Unseen-family figures are averaged over families, each counting equally.\n"]

    if "random" in results:
        rows = results["random"]["result"]
        lines += ["## 2. E1: random split (the usual setup)\n",
                  table(["Task", "Features", "Model", "F1 macro", "Detection", "False alarms"],
                        [[r["task"], settings["feature_sets"][r["features"]], MODEL_NAMES.get(r["model"], r["model"]),
                          f"{r['f1_macro']:.4f}", pct(r["detection_rate"], 2), pct(r["false_alarm_rate"], 2)]
                         for r in rows])]

    if results.get("temporal", {}).get("result", {}).get("rows"):
        temporal = results["temporal"]["result"]
        lines += [f"## 3. E2: train on {', '.join(temporal['train_days'])}; test on "
                  f"{', '.join(temporal['test_days'])}\n",
                  f"Attack types in the test days never seen in training: {', '.join(temporal['unseen'])}. "
                  f"Thresholds are calibrated on benign flows from the {TRAINING_DATA} or on "
                  f"{RECENT_BENIGN} flows from the test days.\n",
                  table(["Features", "Model", "Detector", "Calibration", "Unseen detected", "False alarms", "AUC"],
                        [[settings["feature_sets"][r["features"]], MODEL_NAMES.get(r["model"], r["model"]),
                          r["detector"], r.get("calibration") or "—", pct(r["dr_unseen"]), pct(r["fpr"], 2),
                          "—" if r.get("auc_unseen") is None else f"{r['auc_unseen']:.3f}"]
                         for r in temporal["rows"]])]

    if results.get("loao", {}).get("result", {}).get("rows"):
        loao = results["loao"]["result"]
        families = loao["families"]
        matrix, fpr, detectors = loao_matrix(loao["rows"], families, model, features)
        plot_loao_heatmap(matrix, families, detectors, figures / "loao_heatmap.png")
        header = ["Family left out"] + detectors
        body = [[family] + [pct(None if np.isnan(v) else v) for v in matrix[i]] for i, family in enumerate(families)]
        body.append(["**Mean**"] + [pct(float(np.nanmean(col))) for col in matrix.T])
        body.append(["False alarms (mean)"] + [pct(float(np.nanmean(col)), 2) for col in fpr.T])
        lines += ["## 4. E3: leave one attack family out (zero-day protocol)\n",
                  f"{model_name}, {settings['feature_sets'][features].lower()}. Share of the left-out "
                  "family's flows that raised an alarm.\n", table(header, body)]
        misses = [r for r in loao["rows"] if selected(r, model, features) and r["detector"] == SUPERVISED]
        if misses:
            lines += [f"Share of unseen-family flows the model calls benign with over "
                      f"{1 - settings['confident_threshold']:.0%} confidence:\n",
                      table(["Family left out", "Confidently missed"],
                            [[r["held_out"], pct(r.get("confident_miss"))] for r in misses])]
        counts = loao["unseen_scores"].get(features, {}).get(model)
        if counts:
            plot_overconfidence(counts, settings["score_bins"], settings["confident_threshold"],
                                figures / "overconfidence.png")

    if "shap" in results:
        shap = results["shap"]["result"]
        if "all" in shap:
            plot_shortcut_share(shap["all"]["by_category"], figures / "shap_shortcut_share.png")
        shortcuts = settings["shortcut_features"]
        strictest = list(settings["feature_sets"])[-1]  # the set that removes the most
        rows = []
        for category, entry in shap.get("all", {}).get("by_category", {}).items():
            without = shap.get(strictest, {}).get("by_category", {}).get(category, {})
            rows.append([category, *(pct(entry["shortcut_shares"].get(f)) for f in shortcuts),
                         ", ".join(f["feature"] for f in entry["top_features"][:3]),
                         ", ".join(f["feature"] for f in without.get("top_features", [])[:3]) or "—"])
        lines += ["## 5. E4: reliance on candidate shortcut features (TreeSHAP, XGBoost, binary)\n",
                  "Mean share of each flow's attribution, model trained with all features.\n",
                  table(["Category", *shortcuts, "Top features (all)",
                         f"Top features ({settings['feature_sets'][strictest].lower()})"], rows)]

    lines.append("\nFigures: " + ", ".join(sorted(p.name for p in figures.glob("*.png"))) + "\n")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
