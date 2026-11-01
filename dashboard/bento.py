"""Bento-grid cards for the overview page: an area chart, a terminal, score rings
and bar sparklines, drawn as plain HTML and SVG from the project's real numbers.

Rendered through a components-v2 element because st.html strips inline SVG.
"""

from __future__ import annotations

from html import escape

import numpy as np
import streamlit as st

LAVENDER, GREEN, TEAL, GOLD, MUTED = "#8E92F7", "#2BD4A4", "#2CC3D5", "#E5C454", "#7D8799"

CSS = f"""
:host {{ display: block; }}
.bento {{ font-family: Barlow, system-ui, sans-serif; color: #E6EDF5; }}
.eyebrow {{ color: {GOLD}; font-size: 0.8rem; letter-spacing: 0.14em; margin: 0 0 0.6rem; }}
.bento h1 {{ font-family: 'Barlow Semi Condensed', Barlow, sans-serif; font-size: 2.3rem; font-weight: 700;
            margin: 0 0 0.5rem; line-height: 1.1; }}
.lead {{ color: #94A0B4; font-size: 1.02rem; margin: 0 0 1.6rem; max-width: 70ch; }}
.grid {{ display: grid; gap: 16px; grid-template-columns: 1.75fr 1.15fr 1.1fr; }}
.stack {{ display: grid; gap: 16px; }}
.card {{ background: #0E1218; border: 1px solid #1F2531; border-radius: 16px; padding: 22px 24px;
         min-width: 0; }}
.card.glow {{ background: radial-gradient(120% 90% at 50% 0%, #0F2A26 0%, #0E1218 70%);
              border-color: #1D3B35; }}
.head {{ display: flex; justify-content: space-between; align-items: center; font-size: 0.98rem; }}
.live {{ color: {GREEN}; font-size: 0.85rem; display: flex; align-items: center; gap: 6px; }}
.live::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: {GREEN};
                 box-shadow: 0 0 0 0 {GREEN}; animation: ping 1.8s infinite; }}
@keyframes ping {{ 0% {{ box-shadow: 0 0 0 0 rgba(43,212,164,.6); }} 100% {{ box-shadow: 0 0 0 7px rgba(43,212,164,0); }} }}
.big {{ font-family: 'Barlow Semi Condensed', Barlow, sans-serif; font-size: 3rem; font-weight: 600;
        line-height: 1; margin: 18px 0 8px; font-variant-numeric: tabular-nums; }}
.big {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 8px; }}
.big small {{ font-size: 1rem; color: {MUTED}; font-weight: 400; white-space: nowrap; }}
.big .stats {{ font-family: Barlow, sans-serif; font-size: 0.85rem; color: {MUTED}; margin-left: 10px; white-space: nowrap; }}
.big .stats b {{ color: #E6EDF5; font-weight: 500; margin-right: 12px; }}
svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
.axis {{ display: flex; justify-content: space-between; color: #5F6878; font-size: 0.78rem; margin-top: 6px; }}
.legend {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px 32px; margin-top: 18px; }}
.legend div {{ display: flex; align-items: center; gap: 10px; font-size: 0.95rem; min-width: 0; }}
.legend i {{ width: 8px; height: 8px; border-radius: 50%; flex: none; }}
.legend span {{ margin-left: auto; white-space: nowrap; font-variant-numeric: tabular-nums; font-size: 0.85rem; }}
.dots {{ display: flex; gap: 7px; }} .dots i {{ width: 10px; height: 10px; border-radius: 50%; }}
.term {{ background: #07090D; border: 1px solid #1A1F28; border-radius: 12px; margin-top: 16px;
         padding: 16px 18px; font-family: Consolas, 'Cascadia Mono', monospace; font-size: 0.8rem;
         line-height: 1.6; min-height: 300px; display: flex; flex-direction: column; justify-content: flex-end; }}
.term p {{ margin: 0 0 8px; display: flex; gap: 10px; opacity: 0; animation: show .4s forwards; }}
.term .c {{ color: #C792EA; }} .term .ok {{ color: {GREEN}; }} .term .note {{ color: {MUTED}; }}
@keyframes show {{ to {{ opacity: 1; }} }}
.rings {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 22px; text-align: center; }}
.rings small {{ color: {MUTED}; font-size: 0.78rem; display: block; margin-top: 6px; }}
.wide {{ grid-column: 1 / -1; display: grid; grid-template-columns: 1.2fr 1fr; gap: 32px; }}
.reality {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 18px; }}
.reality div {{ border-left: 2px solid #232A36; padding-left: 14px; }}
.reality b {{ display: block; font-family: 'Barlow Semi Condensed', Barlow, sans-serif; font-size: 2rem; font-weight: 600; }}
.reality span {{ color: {MUTED}; font-size: 0.85rem; }}
.note-text {{ color: {MUTED}; font-size: 0.82rem; margin-top: 14px; }}
@media (max-width: 900px) {{ .grid, .wide {{ grid-template-columns: 1fr; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; opacity: 1 !important; }} }}
"""

JS = """
export default function ({ data, parentElement }) {
  let root = parentElement.querySelector(".bento");
  if (!root) { root = document.createElement("div"); root.className = "bento"; parentElement.appendChild(root); }
  if (root.dataset.html !== data) { root.innerHTML = data; root.dataset.html = data; }
}
"""

_component = st.components.v2.component("overview_bento", js=JS, css=CSS)


def render(html: str, key: str = "overview_bento") -> None:
    _component(key=key, data=html, height="content")


def card(body: str, title: str = "", right: str = "", extra: str = "") -> str:
    head = f'<div class="head"><span>{escape(title)}</span>{right}</div>' if title else ""
    return f'<div class="card {extra}">{head}{body}</div>'


def live(label: str = "Live") -> str:
    return f'<span class="live">{escape(label)}</span>'


def area_chart(values: np.ndarray, color: str = LAVENDER, width: int = 520, height: int = 150) -> str:
    """Smooth filled line with a gradient fade and a marker on the latest value."""
    values = np.asarray(values, dtype=float)
    top = max(values.max(), 1e-9) * 1.1
    xs = np.linspace(0, width, len(values))
    ys = height - values / top * height
    points = list(zip(xs, ys))
    path = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
    for (x0, y0), (x1, y1) in zip(points, points[1:]):  # midpoint smoothing
        path += f" Q {x0:.1f} {y0:.1f} {(x0 + x1) / 2:.1f} {(y0 + y1) / 2:.1f}"
    path += f" T {points[-1][0]:.1f} {points[-1][1]:.1f}"
    grid = "".join(f'<line x1="0" x2="{width}" y1="{y}" y2="{y}" stroke="#1C222C" stroke-dasharray="2 4"/>'
                   for y in (height * 0.05, height * 0.4))
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Attack share over the replay">'
            f'<defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity="0.32"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>{grid}'
            f'<path d="{path} L {width} {height} L 0 {height} Z" fill="url(#fill)"/>'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<circle cx="{points[-1][0]:.1f}" cy="{points[-1][1]:.1f}" r="5" fill="{color}" '
            f'stroke="#0E1218" stroke-width="2"/></svg>')


def ring(value: float, label: str, text: str, color: str = GREEN) -> str:
    """Circular score gauge, value in [0, 1]."""
    r, c = 22, 2 * np.pi * 22
    return (f'<div><svg viewBox="0 0 56 56" style="width:58px;margin:auto">'
            f'<circle cx="28" cy="28" r="{r}" fill="none" stroke="#163A33" stroke-width="5"/>'
            f'<circle cx="28" cy="28" r="{r}" fill="none" stroke="{color}" stroke-width="5" '
            f'stroke-linecap="round" stroke-dasharray="{c * value:.1f} {c:.1f}" transform="rotate(-90 28 28)"/>'
            f'<text x="28" y="32.5" text-anchor="middle" fill="#E6EDF5" font-size="12.5" font-weight="600" '
            f'font-family="Barlow, sans-serif">{escape(text)}</text></svg><small>{escape(label)}</small></div>')


def bars(values: list[float], labels: list[str], highlight: int = -1, color: str = TEAL,
         width: int = 230, height: int = 56) -> str:
    """Rounded bar sparkline with one highlighted bar (the last by default)."""
    top = max(values) or 1
    highlight %= len(values)
    step = width / len(values)
    out = []
    for i, (value, label) in enumerate(zip(values, labels)):
        h = max(value / top * height, 4)
        last = i == highlight
        out.append(f'<rect x="{i * step + step * 0.18:.1f}" y="{height - h:.1f}" width="{step * 0.64:.1f}" '
                   f'height="{h:.1f}" rx="{step * 0.3:.1f}" fill="{color}" opacity="{1 if last else 0.38}">'
                   f'<title>{escape(label)}</title></rect>')
    return f'<svg viewBox="0 0 {width} {height}" style="margin-top:20px">{"".join(out)}</svg>'


def terminal(lines: list[tuple[str, str]]) -> str:
    """Lines of (kind, text): kind is 'cmd', 'ok' or 'note'. They appear one after another."""
    marks = {"cmd": ("$", "c"), "ok": ("✓", "ok"), "note": ("›", "note")}
    rows = []
    for i, (kind, text) in enumerate(lines):
        mark, cls = marks[kind]
        rows.append(f'<p style="animation-delay:{0.35 * i:.2f}s"><span class="{cls}">{mark}</span>'
                    f'<span{" class=note" if kind == "note" else ""}>{escape(text)}</span></p>')
    return f'<div class="term">{"".join(rows)}</div>'


def legend(rows: list[tuple[str, str, str]]) -> str:
    """Rows of (dot colour, label, value)."""
    return '<div class="legend">' + "".join(
        f'<div><i style="background:{c}"></i>{escape(label)}<span style="color:{c}">{escape(value)}</span></div>'
        for c, label, value in rows) + "</div>"
