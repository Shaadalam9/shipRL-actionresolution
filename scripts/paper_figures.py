#!/usr/bin/env python3
"""Draw the paper's figures for the paper and the README.

Fig. 1 (calm water summary) is drawn from Table II.

The figure must match Table II of the paper exactly, so the values below are
copied from that table rather than produced by a fresh evaluation. Each panel
is written twice:

- ``paper/plots/calm_summary_<metric>.pdf``, sized for half the page width;
- ``docs/figures/calm_summary_<metric>.{png,html}`` for the README, with
  plotly.js embedded so the HTML renders through htmlpreview.

Fig. 2 (model-scale square path experiments) is a set of measured tracks over
aerial imagery, kept as ``docs/figures/experiment_square_<controller>.png``.
For each, an HTML page with plotly.js embedded is written beside it, so the
README can link every figure to a zoomable version.

Run from anywhere::

    uv run scripts/paper_figures.py
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plotly.graph_objects as go  # noqa: E402

from asv_rl.evaluation.plotting import grouped_bar_figure, save_figure  # noqa: E402

PAPER_DIR = REPO_ROOT / "paper" / "plots"
DOCS_DIR = REPO_ROOT / "docs" / "figures"

PATHS = ["(10,10)", "(-10,10)", "(-10,-10)", "(10,-10)",
         "Cardioid", "Astroid", "Sine", "Star", "Circle"]

#: Table II: cross-track RMSE (ship lengths L) per manoeuvre.
RMSE = {
    "dqn_3_action": [0.75, 2.22, 2.06, 0.73, 0.35, 1.23, 0.89, 1.14, 1.08],
    "dqn_5_action": [0.91, 2.48, 2.29, 0.90, 0.47, 1.20, 0.97, 1.44, 0.61],
    "ppo":          [0.84, 2.44, 2.34, 0.82, 0.31, 1.21, 0.91, 1.55, 0.82],
}

#: Table II: mean absolute realised rudder angle E_delta (rad) per manoeuvre.
RUDDER = {
    "dqn_3_action": [0.19, 0.23, 0.40, 0.31, 0.27, 0.33, 0.26, 0.30, 0.37],
    "dqn_5_action": [0.16, 0.22, 0.27, 0.21, 0.22, 0.28, 0.25, 0.26, 0.34],
    "ppo":          [0.12, 0.19, 0.18, 0.12, 0.12, 0.21, 0.13, 0.14, 0.27],
}

PANELS = (
    ("calm_summary_rmse", RMSE, "Cross-track RMSE (L)"),
    ("calm_summary_rudder", RUDDER, "Mean |rudder angle| (rad)"),
)


def paper_panel(values, y_title):
    """Compact panel for half the page width: no value labels, larger text."""
    fig = grouped_bar_figure(PATHS, values, y_title=y_title)
    # Nine groups of three bars: value labels would overlap, and Table II has them.
    fig.update_traces(text=None)
    fig.update_layout(
        font=dict(size=20),
        xaxis=dict(tickfont=dict(size=18)),
        yaxis=dict(tickfont=dict(size=18), title=dict(font=dict(size=20))),
        legend=dict(font=dict(size=18)),
        margin=dict(l=80, r=10, t=50, b=50),
    )
    return fig


#: Fig. 2: measured square path tracks of the model-scale experiments.
EXPERIMENT_IMAGES = ("experiment_square_dqn_3", "experiment_square_dqn_5", "experiment_square_ppo")


def image_figure(png: Path) -> go.Figure:
    """Show a PNG as a zoomable Plotly figure with no axes."""
    encoded = base64.b64encode(png.read_bytes()).decode("ascii")
    fig = go.Figure(go.Image(source=f"data:image/png;base64,{encoded}", hoverinfo="skip"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


def main() -> int:
    for stem, values, y_title in PANELS:
        save_figure(paper_panel(values, y_title), PAPER_DIR / stem, formats=("pdf",), size=(800, 420))
        save_figure(
            grouped_bar_figure(PATHS, values, y_title=y_title),
            DOCS_DIR / stem, formats=("png", "html"), embed_plotlyjs=True,
        )
        print(f"{stem}: {PAPER_DIR / stem}.pdf, {DOCS_DIR / stem}.{{png,html}}")
    for stem in EXPERIMENT_IMAGES:
        save_figure(image_figure(DOCS_DIR / f"{stem}.png"), DOCS_DIR / stem, formats=("html",),
                    embed_plotlyjs=True)
        print(f"{stem}: {DOCS_DIR / stem}.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
