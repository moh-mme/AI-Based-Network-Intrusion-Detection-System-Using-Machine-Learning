import numpy as np
import streamlit as st

import bento
from common import (
    CATEGORY_COLORS,
    compact,
    get_config,
    get_demo_stream,
    get_metrics,
    get_profile,
    read_json,
    synthetic_warning,
    trained_tasks,
)
from nids.data import BENIGN, CATEGORY_COL
from nids.research_report import headline

WINDOW = 400  # flows per point of the replay chart

profile = get_profile()
tasks = trained_tasks()
report = get_metrics("multiclass") if "multiclass" in tasks else (get_metrics(tasks[0]) if tasks else None)
stream = get_demo_stream()
research = read_json(get_config().paths.research_reports / "results.json")
synthetic_warning()

intro = (f"Six machine-learning models trained on {compact(profile['rows_final'])} CIC-IDS2017 network flows, "
         "a live replay of traffic they never saw, and an honest look at how they cope with new attacks."
         if profile else "Machine-learning models that label network flows as benign or as an attack type.")
html = ['<p class="eyebrow">INTRUSION DETECTION</p>',
        "<h1>AI-Based Network Intrusion Detection System</h1>", f'<p class="lead">{intro}</p>',
        '<div class="grid">']

# --- Replay card: attack share over the held-out stream -----------------------------
if stream is not None:
    categories = stream[CATEGORY_COL].astype(str).to_numpy()
    is_attack = (categories != BENIGN).astype(float)
    windows = is_attack[: len(is_attack) // WINDOW * WINDOW].reshape(-1, WINDOW).mean(axis=1)
    counts = {c: int((categories == c).sum()) for c in np.unique(categories) if c != BENIGN}
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
    bursts = int(((windows[1:] > 0.25) & (windows[:-1] <= 0.25)).sum() + (windows[0] > 0.25))
    body = (f'<div class="big">{is_attack.mean():.0%}<small>of flows are attacks</small>'
            f'<span class="stats">peak <b>{windows.max():.0%}</b>bursts <b>{bursts}</b></span></div>'
            + bento.area_chart(windows)
            + f'<div class="axis"><span>{len(categories):,} flows ago</span><span>now</span></div>'
            + bento.legend([(CATEGORY_COLORS.get(c, bento.MUTED), c, f"{n:,}") for c, n in top]))
    html.append(bento.card(body, "Traffic replay: share of attack flows", bento.live()))
else:
    html.append(bento.card('<p class="note-text">Train the models to create the replay stream.</p>', "Traffic replay"))

# --- Terminal card: the pipeline as it ran -------------------------------------------
lines = [("cmd", "python -m nids.prepare")]
if profile:
    lines += [("note", f"{profile['rows_read']:,} rows read, {profile['duplicates_removed']:,} duplicates removed"),
              ("ok", f"{profile['rows_final']:,} clean flows, {len(profile['features'])} features")]
lines.append(("cmd", "python -m nids.train"))
if report:
    best = report["models"][report["best_model"]]
    lines += [("note", f"Training {len(report['models'])} models on {report['n_train']:,} flows…"),
              ("ok", f"Best: {best['display_name']}, F1 macro {best['metrics']['f1_macro']:.4f}")]
items = headline(research) if research else []
if items:
    lines.append(("cmd", "python -m nids.experiments"))
    unseen = [dr for label, dr, _ in items if label.startswith("Unseen")]
    if len(unseen) >= 2:
        lines.append(("ok", f"Unseen attack families: {unseen[0]:.1%} → {max(unseen[1:]):.1%} detected"))
dots = '<span class="dots"><i style="background:#E5484D"></i><i style="background:#E5C454"></i><i style="background:#2BD4A4"></i></span>'
html.append(bento.card(bento.terminal(lines), "pipeline.sh", dots))

# --- Right column: score rings and classification speed ------------------------------
right = []
if report:
    m = best["metrics"]
    rings = (bento.ring(m["accuracy"], "Accuracy", f"{m['accuracy'] * 100:.1f}")
             + bento.ring(m["detection_rate"], "Detection", f"{m['detection_rate'] * 100:.1f}")
             + bento.ring(m["f1_macro"], "F1 macro", f"{m['f1_macro'] * 100:.1f}")
             + bento.ring(1 - m["false_alarm_rate"], "No false alarm", f"{(1 - m['false_alarm_rate']) * 100:.1f}"))
    right.append(bento.card(f'<div class="rings">{rings}</div>', f"Best model: {best['display_name']}", extra="glow"))
    speeds = {r["display_name"]: 1e6 / r["predict_ms_per_1k"] for r in report["models"].values()}
    order = sorted(speeds, key=speeds.get)
    right.append(bento.card(
        f'<div class="big">{compact(round(speeds[best["display_name"]]))}<small>flows/s</small></div>'
        + bento.bars([np.log10(speeds[n] / speeds[order[0]]) + 0.3 for n in order], [f"{n}: {speeds[n]:,.0f} flows/s" for n in order],
                     highlight=order.index(best["display_name"]))
        + '<p class="note-text">Bars per model, log scale; highlighted: the best model.</p>',
        "Classification speed", f'<span class="note-text" style="margin:0">{len(order)} models</span>'))
html.append(f'<div class="stack">{"".join(right)}</div>')

# --- Wide card: per-class results and the reality check ------------------------------
if report:
    rows = [(CATEGORY_COLORS.get(c, bento.MUTED), c, f"{best['report'][c]['recall']:.1%}")
            for c in report["classes"] if c != BENIGN]
    left = bento.card(bento.legend(rows), "Detection by attack type (random split)")
    reality = ""
    if items:
        lookup = {label: dr for label, dr, _ in items}
        random_dr = next((dr for label, dr in lookup.items() if label.startswith("Random")), None)
        unseen = [dr for label, dr in lookup.items() if label.startswith("Unseen")]
        later = [dr for label, dr in lookup.items() if label.startswith("Later") and "supervised" in label]
        cells = [(random_dr, "attacks seen in training"), (unseen[0] if unseen else None, "unseen attack family"),
                 (later[0] if later else None, "later days of traffic")]
        reality = bento.card(
            '<div class="reality">' + "".join(f"<div><b>{v:.1%}</b><span>{t}</span></div>"
                                               for v, t in cells if v is not None) + "</div>"
            + '<p class="note-text">Same model, same data. Details on the Research Findings page.</p>',
            "Beyond 99.9%: detection rate by evaluation")
    html.append(f'<div class="wide">{left}{reality}</div>')

html.append("</div>")
bento.render("".join(html))

complete = bool(profile and tasks)
with st.expander("Reproduce these results" if complete else "Getting started", expanded=not complete):
    st.markdown("Put the CIC-IDS2017 MachineLearningCSV files in `data/raw/`, then run:")
    for command in ("python -m nids.prepare", "python -m nids.train", "python -m nids.experiments",
                    "streamlit run dashboard/app.py"):
        st.code(command, language="bash")
