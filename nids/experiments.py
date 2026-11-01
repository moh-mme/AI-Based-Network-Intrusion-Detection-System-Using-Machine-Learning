"""Research experiments: how much of the reported 99.9% survives a realistic
evaluation, and can a hybrid detector catch attacks it was never trained on?

    python -m nids.experiments [--config CONFIG] [--quick] [--only random temporal loao shap]

E1 random    The usual random 80/20 split, for each configured feature set.
E2 temporal  Train on the first days of the week and test on the last: every
             attack type in the test set is new to the model.
E3 loao      Leave one attack family out of training, then test on it (the
             zero-day protocol): supervised models, anomaly detectors, hybrids.
E4 shap      How much each attack type's detection leans on shortcut features.

Results go to reports/research/ (results.json, summary.md, figures/).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score

from . import setup_logging
from .anomaly import DISPLAY_NAMES as DETECTOR_NAMES
from .anomaly import build_detector, conformal_pvalues
from .config import Config, load_config
from .data import BENIGN, CATEGORY_COL, DAY_COL, cap_per_class, feature_columns, load_dataset, load_profile
from .evaluate import evaluate as evaluate_model
from .models import build_pipeline, predict_proba
from .research_report import write_report
from .train import BINARY_CLASSES, QUICK_OVERRIDES, class_order, encode_targets, split_indices

log = logging.getLogger(__name__)

EXPERIMENTS = ("random", "temporal", "loao", "shap")
QUICK_RESEARCH = {"calibration_benign": 10000, "anomaly_train_benign": 30000,
                  "max_test_benign": 50000, "shap_flows_per_category": 300}
# Where thresholds are calibrated: benign flows held out of the training data, or
# recent benign flows from the test period (the temporal experiment only).
TRAINING_DATA, RECENT_BENIGN = "training data", "recent benign"
# P(attack) below this means the model is at least 99% sure an attack flow is benign.
CONFIDENT = 0.01
# Log-spaced bins for the histogram of P(attack) on attacks the model never saw.
SCORE_BINS = np.concatenate([[0.0], np.logspace(-6, 0, 31)])


@dataclass
class Flows:
    X: np.ndarray
    features: np.ndarray
    category: np.ndarray
    day: np.ndarray

    @property
    def is_attack(self) -> np.ndarray:
        return self.category != BENIGN


def load_flows(cfg: Config) -> Flows:
    df = load_dataset(cfg)
    if DAY_COL not in df.columns:
        raise ValueError("The prepared dataset has no capture-day column. Rerun `python -m nids.prepare`.")
    features = feature_columns(df)
    return Flows(df[features].to_numpy(np.float32), np.asarray(features),
                 df[CATEGORY_COL].astype(str).to_numpy(), df[DAY_COL].astype(str).to_numpy())


def feature_sets(features: np.ndarray, specs: dict) -> dict[str, np.ndarray]:
    """Column mask for each configured feature set (name -> {label, drop})."""
    return {name: ~np.isin(features, spec.get("drop") or []) for name, spec in specs.items()}


def shortcut_features(specs: dict) -> list[str]:
    """Every feature some set removes, in configuration order."""
    return list(dict.fromkeys(f for spec in specs.values() for f in spec.get("drop") or []))


def rate(flags: np.ndarray) -> float | None:
    return float(flags.mean()) if flags.size else None


def family_rates(flagged: np.ndarray, categories: np.ndarray, families=None) -> dict[str, float]:
    """Share of each attack family's flows that raised an alarm."""
    families = families or [c for c in class_order(categories) if c != BENIGN]
    return {c: rate(flagged[categories == c]) for c in sorted(families) if (categories == c).any()}


def fit_supervised(name: str, cfg: Config, X: np.ndarray, y: np.ndarray, seed: int):
    pipeline, _ = build_pipeline(name, cfg.models.get(name), seed)
    return pipeline.fit(X, y)


def rows_of(X: np.ndarray, idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return X[np.ix_(idx, np.flatnonzero(mask))]


# --- E1: the usual random split --------------------------------------------------------

def run_random_split(flows: Flows, sets: dict, cfg: Config, tcfg: dict, rcfg: dict) -> list[dict]:
    seed = tcfg["random_state"]
    train_idx, test_idx = split_indices(flows.category, tcfg)
    classes = class_order(flows.category)
    rows = []
    for task in ("binary", "multiclass"):
        names = BINARY_CLASSES if task == "binary" else classes
        y = encode_targets(task, flows.category, classes)
        for set_name, mask in sets.items():
            X_train, X_test = rows_of(flows.X, train_idx, mask), rows_of(flows.X, test_idx, mask)
            for model in rcfg["models"]:
                start = time.perf_counter()
                pipeline = fit_supervised(model, cfg, X_train, y[train_idx], seed)
                proba = predict_proba(pipeline, X_test, len(names))
                m = evaluate_model(y[test_idx], proba, names)["metrics"]
                rows.append({
                    "task": task, "features": set_name, "model": model,
                    **{k: m[k] for k in ("accuracy", "f1_macro", "detection_rate", "false_alarm_rate")},
                    "by_family": family_rates(proba.argmax(axis=1) != 0, flows.category[test_idx]),
                })
                log.info("[E1 random] %s %s %s: F1 macro %.4f, detection %.4f, false alarms %.4f (%.0fs)",
                         task, set_name, model, m["f1_macro"], m["detection_rate"],
                         m["false_alarm_rate"], time.perf_counter() - start)
    return rows


# --- Shared zero-day machinery ---------------------------------------------------------

@dataclass
class BenignSplit:
    calibration: np.ndarray    # only used to set thresholds
    anomaly_train: np.ndarray  # fits the anomaly detectors
    supervised: np.ndarray     # benign part of the supervised training set


def split_benign(benign_idx: np.ndarray, rcfg: dict, seed: int, max_supervised: int | None) -> BenignSplit:
    shuffled = np.random.default_rng(seed).permutation(benign_idx)
    n_calibration = min(rcfg["calibration_benign"], len(shuffled) // 4)
    calibration, rest = shuffled[:n_calibration], shuffled[n_calibration:]
    return BenignSplit(np.sort(calibration), np.sort(rest[: rcfg["anomaly_train_benign"]]),
                       np.sort(rest[:max_supervised]))


def fit_anomaly_detectors(flows: Flows, split: BenignSplit, mask: np.ndarray, rcfg: dict, seed: int) -> dict:
    X_train = rows_of(flows.X, split.anomaly_train, mask)
    detectors = {}
    for name, params in rcfg["anomaly_detectors"].items():
        start = time.perf_counter()
        detectors[name] = build_detector(name, params, seed).fit(X_train)
        log.info("Fitted %s on %d benign flows (%.0fs)", DETECTOR_NAMES[name], len(X_train),
                 time.perf_counter() - start)
    return detectors


def anomaly_scores(detectors: dict, flows: Flows, idx: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Scores for the flows at `idx`, as full-length arrays (NaN elsewhere) for cheap reuse."""
    X = rows_of(flows.X, idx, mask)
    scores = {}
    for name, detector in detectors.items():
        full = np.full(len(flows.category), np.nan)
        full[idx] = detector.score(X)
        scores[name] = full
    return scores


def capped_benign(idx: np.ndarray, categories: np.ndarray, limit: int, seed: int) -> np.ndarray:
    benign = idx[categories[idx] == BENIGN]
    if len(benign) > limit:
        benign = np.random.default_rng(seed).choice(benign, size=limit, replace=False)
    return np.sort(benign)


def zero_day_scenario(scenario: str, flows: Flows, mask: np.ndarray, supervised_train: np.ndarray,
                      calibrations: dict[str, np.ndarray], test: np.ndarray, unseen: set[str],
                      scores: dict, cfg: Config, rcfg: dict, seed: int) -> tuple[list[dict], dict]:
    """Supervised models, anomaly detectors and hybrids on one train/test arrangement.

    Every calibrated detector flags a flow when its conformal p-value is at most the
    alarm budget alpha. The hybrid flags a flow when either part is suspicious at
    alpha/2, so by the union bound it still stays within alpha. Thresholds are set
    once per calibration set (name -> benign flow indices).
    """
    alpha = rcfg["target_fpr"]
    categories = flows.category[test]
    benign = categories == BENIGN
    is_unseen = np.isin(categories, sorted(unseen))
    known = ~benign & ~is_unseen
    rows, unseen_scores = [], {}

    def record(model: str, detector: str, calibration: str | None, flagged: np.ndarray,
               suspicion: np.ndarray, **extra):
        row = {"scenario": scenario, "model": model, "detector": detector, "calibration": calibration,
               "fpr": rate(flagged[benign]), "dr_unseen": rate(flagged[is_unseen]),
               "dr_known": rate(flagged[known]), "by_family": family_rates(flagged, categories, unseen), **extra}
        if is_unseen.any() and benign.any():
            both = benign | is_unseen
            row["auc_unseen"] = float(roc_auc_score(is_unseen[both], suspicion[both]))
        rows.append(row)

    p_anomaly = {(name, where): conformal_pvalues(s[idx], s[test])
                 for name, s in scores.items() for where, idx in calibrations.items()}
    for (name, where), p in p_anomaly.items():
        record("none", DETECTOR_NAMES[name], where, p <= alpha, -p)

    y_train = flows.is_attack[supervised_train].astype(np.int64)
    X_train = rows_of(flows.X, supervised_train, mask)
    for model in rcfg["models"]:
        start = time.perf_counter()
        pipeline = fit_supervised(model, cfg, X_train, y_train, seed)
        s_test = predict_proba(pipeline, rows_of(flows.X, test, mask), 2)[:, 1]
        confident_miss = rate(s_test[is_unseen] < CONFIDENT)
        record(model, "Supervised, threshold 0.5", None, s_test >= 0.5, s_test, confident_miss=confident_miss)
        for where, idx in calibrations.items():
            s_calibration = predict_proba(pipeline, rows_of(flows.X, idx, mask), 2)[:, 1]
            p_supervised = conformal_pvalues(s_calibration, s_test)
            record(model, "Supervised, calibrated", where, p_supervised <= alpha, -p_supervised)
            for name in scores:
                combined = np.minimum(p_supervised, p_anomaly[(name, where)])
                record(model, f"Hybrid: supervised + {DETECTOR_NAMES[name]}", where,
                       combined <= alpha / 2, -combined)
        unseen_scores[model] = s_test[is_unseen]
        log.info("[%s] %s: unseen attacks detected %.3f at threshold 0.5, %.1f%% of them with "
                 "P(attack) < %.2f (%.0fs)", scenario, model, rate(s_test[is_unseen] >= 0.5) or 0,
                 100 * (confident_miss or 0), CONFIDENT, time.perf_counter() - start)
    return rows, unseen_scores


def histogram(values: list[np.ndarray]) -> list[int]:
    values = np.concatenate(values) if values else np.array([])
    return np.histogram(np.clip(values, 0, 1), bins=SCORE_BINS)[0].tolist()


# --- E2: train on early days, test on later days ------------------------------------------

def run_temporal(flows: Flows, sets: dict, cfg: Config, tcfg: dict, rcfg: dict) -> dict:
    seed = tcfg["random_state"]
    days = np.char.lower(flows.day.astype(str))
    train_days = [d.lower() for d in rcfg["temporal"]["train_days"]]
    test_days = [d.lower() for d in rcfg["temporal"]["test_days"]]
    train_pool, test_pool = np.flatnonzero(np.isin(days, train_days)), np.flatnonzero(np.isin(days, test_days))
    if not len(train_pool) or not len(test_pool):
        log.warning("[E2 temporal] No flows on the configured days %s / %s; skipped.", train_days, test_days)
        return {"rows": [], "unseen_scores": {}}

    split = split_benign(train_pool[flows.category[train_pool] == BENIGN], rcfg, seed,
                         tcfg.get("max_benign_train"))
    attacks = train_pool[flows.is_attack[train_pool]]
    attacks = attacks[cap_per_class(flows.category[attacks], tcfg.get("max_train_per_class"),
                                    np.random.default_rng(seed))]
    supervised_train = np.sort(np.concatenate([split.supervised, attacks]))
    # Test-day benign flows: one part for testing, a separate part as "recent" calibration data.
    test_benign_pool = np.random.default_rng(seed).permutation(
        test_pool[flows.category[test_pool] == BENIGN])
    n_recent = min(rcfg["temporal"].get("recent_calibration_benign", 0), len(test_benign_pool) // 4)
    recent, test_benign = np.sort(test_benign_pool[:n_recent]), test_benign_pool[n_recent:]
    test = np.sort(np.concatenate([test_benign[: rcfg["max_test_benign"]], test_pool[flows.is_attack[test_pool]]]))
    calibrations = {TRAINING_DATA: split.calibration}
    if n_recent:
        calibrations[RECENT_BENIGN] = recent
    unseen = set(flows.category[test][flows.is_attack[test]]) - set(flows.category[attacks])
    log.info("[E2 temporal] Train %s: %d flows; test %s: %d flows; unseen attack types: %s",
             train_days, len(supervised_train), test_days, len(test), sorted(unseen))

    rows, unseen_scores = [], {}
    for set_name, mask in sets.items():
        detectors = fit_anomaly_detectors(flows, split, mask, rcfg, seed)
        scores = anomaly_scores(detectors, flows, np.concatenate([*calibrations.values(), test]), mask)
        scenario_rows, model_scores = zero_day_scenario("temporal", flows, mask, supervised_train,
                                                        calibrations, test, unseen, scores, cfg, rcfg, seed)
        rows += [{**row, "features": set_name} for row in scenario_rows]
        unseen_scores[set_name] = {model: histogram([s]) for model, s in model_scores.items()}
    return {"rows": rows, "unseen_scores": unseen_scores, "unseen": sorted(unseen),
            "train_days": rcfg["temporal"]["train_days"], "test_days": rcfg["temporal"]["test_days"]}


# --- E3: leave one attack family out ---------------------------------------------------

def run_leave_one_out(flows: Flows, sets: dict, cfg: Config, tcfg: dict, rcfg: dict) -> dict:
    seed = tcfg["random_state"]
    categories = flows.category
    train_idx, test_idx = split_indices(categories, tcfg)
    split = split_benign(train_idx[categories[train_idx] == BENIGN], rcfg, seed, None)
    test_benign = capped_benign(test_idx, categories, rcfg["max_test_benign"], seed)
    train_attacks = train_idx[flows.is_attack[train_idx]]
    test_attacks = test_idx[flows.is_attack[test_idx]]
    families = [c for c in class_order(categories) if c != BENIGN]

    rows, unseen_scores = [], {}
    for set_name, mask in sets.items():
        detectors = fit_anomaly_detectors(flows, split, mask, rcfg, seed)
        scored = np.concatenate([split.calibration, test_benign, np.flatnonzero(flows.is_attack)])
        scores = anomaly_scores(detectors, flows, scored, mask)
        pooled: dict[str, list] = {}
        for family in families:
            supervised_train = np.sort(np.concatenate(
                [split.supervised, train_attacks[categories[train_attacks] != family]]))
            # The held-out family is tested on all its flows: none were used for training.
            test = np.sort(np.concatenate([test_benign, test_attacks[categories[test_attacks] != family],
                                           np.flatnonzero(categories == family)]))
            scenario_rows, model_scores = zero_day_scenario(
                f"leave out {family}", flows, mask, supervised_train, {TRAINING_DATA: split.calibration},
                test, {family}, scores, cfg, rcfg, seed)
            rows += [{**row, "features": set_name, "held_out": family} for row in scenario_rows]
            for model, s in model_scores.items():
                pooled.setdefault(model, []).append(s)
        unseen_scores[set_name] = {model: histogram(s) for model, s in pooled.items()}
    return {"rows": rows, "unseen_scores": unseen_scores, "families": families}


# --- E4: TreeSHAP, how much detection leans on shortcuts --------------------------------

def run_shap(flows: Flows, sets: dict, cfg: Config, tcfg: dict, rcfg: dict) -> dict:
    from xgboost import DMatrix

    seed = tcfg["random_state"]
    train_idx, test_idx = split_indices(flows.category, tcfg)
    sample = test_idx[cap_per_class(flows.category[test_idx], rcfg["shap_flows_per_category"],
                                    np.random.default_rng(seed))]
    categories = flows.category[sample]
    shortcuts = shortcut_features(rcfg["feature_sets"])
    y = flows.is_attack.astype(np.int64)
    results = {}
    for set_name, mask in sets.items():
        pipeline = fit_supervised("xgboost", cfg, rows_of(flows.X, train_idx, mask), y[train_idx], seed)
        kept = flows.features[mask][pipeline.named_steps["drop_constant"].get_support()]
        Z = pipeline[:-1].transform(rows_of(flows.X, sample, mask))
        booster = pipeline.named_steps["clf"].get_booster()
        contributions = np.abs(booster.predict(DMatrix(Z), pred_contribs=True)[:, :-1])
        # Each flow's attribution as shares of 1, so every flow counts equally.
        shares = contributions / np.maximum(contributions.sum(axis=1, keepdims=True), 1e-12)
        by_category = {}
        for category in class_order(categories):
            mean_share = shares[categories == category].mean(axis=0)
            top = np.argsort(mean_share)[::-1][:5]
            by_category[category] = {
                "shortcut_shares": {f: float(mean_share[kept == f].sum()) for f in shortcuts if (kept == f).any()},
                "top_features": [{"feature": str(kept[i]), "share": float(mean_share[i])} for i in top],
            }
        overall = contributions.mean(axis=0)
        top = np.argsort(overall)[::-1][:15]
        results[set_name] = {
            "by_category": by_category,
            "global": [{"feature": str(kept[i]), "mean_abs_shap": float(overall[i])} for i in top],
        }
        log.info("[E4 shap] %s: shortcut shares by category %s", set_name,
                 {c: {f: round(s, 3) for f, s in v["shortcut_shares"].items()} for c, v in by_category.items()})
    return results


# --- Orchestration ------------------------------------------------------------------------

def run_experiments(cfg: Config, only: list[str] | None = None, quick: bool = False) -> dict:
    tcfg = {"random_state": 42, **cfg.training}
    rcfg = dict(cfg.research)
    if quick:
        tcfg.update(QUICK_OVERRIDES)
        rcfg.update(QUICK_RESEARCH)
    rcfg.setdefault("models", ["xgboost"])
    only = list(only or EXPERIMENTS)

    flows = load_flows(cfg)
    sets = feature_sets(flows.features, rcfg["feature_sets"])
    runners = {"random": run_random_split, "temporal": run_temporal,
               "loao": run_leave_one_out, "shap": run_shap}
    stamp = {"created_at": datetime.now().isoformat(timespec="seconds"), "quick": quick}

    out_dir = cfg.paths.research_reports
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    # Earlier results are kept, so a single experiment can be rerun on its own.
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    results["settings"] = {
        "data_source": (load_profile(cfg) or {}).get("data_source", "unknown"),
        "target_fpr": rcfg["target_fpr"],
        "shortcut_features": shortcut_features(rcfg["feature_sets"]),
        "models": rcfg["models"],
        "feature_sets": {name: spec["label"] for name, spec in rcfg["feature_sets"].items()},
        "primary_feature_set": rcfg.get("primary_feature_set", next(iter(sets))),
        "anomaly_detectors": {name: DETECTOR_NAMES[name] for name in rcfg["anomaly_detectors"]},
        "confident_threshold": CONFIDENT,
        "score_bins": SCORE_BINS.tolist(),
    }
    for name in only:
        start = time.perf_counter()
        outcome = runners[name](flows, sets, cfg, tcfg, rcfg)
        results[name] = {**stamp, "seconds": round(time.perf_counter() - start), "result": outcome}
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")  # save as we go
        log.info("Finished %s in %.0fs", name, time.perf_counter() - start)

    write_report(results, out_dir)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the research experiments.")
    parser.add_argument("--config", help="path to a config file (default: config.yaml)")
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, help="default: all experiments")
    parser.add_argument("--quick", action="store_true", help="small samples for a fast trial run")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    run_experiments(cfg, args.only, args.quick)
    print(f"\nResults: {cfg.paths.research_reports / 'summary.md'}")


if __name__ == "__main__":
    main()
