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
The remaining figures from the study are kept as images under
``docs/figures/gallery/<topic>/``. For every one of these images an HTML page
of the same name is written beside it: a zoomable, pannable view of the image,
so each README figure links to an interactive version. The page loads the
image beside it and falls back to the copy on GitHub, which is what the
htmlpreview viewer needs.

Run from anywhere::

    uv run scripts/paper_figures.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asv_rl.evaluation.plotting import grouped_bar_figure, save_figure  # noqa: E402

PAPER_DIR = REPO_ROOT / "paper" / "plots"
DOCS_DIR = REPO_ROOT / "docs" / "figures"
GALLERY_DIR = DOCS_DIR / "gallery"

#: Raw file URL of docs/figures on GitHub, used when the page is not opened locally.
RAW_DOCS_URL = "https://raw.githubusercontent.com/Shaadalam9/shipRL-actionresolution/main/docs/figures"

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

VIEWER_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #fcfcfb; font-family: Helvetica, Arial, sans-serif; }}
  #bar {{ padding: 8px 12px; font-size: 14px; color: #52514e; border-bottom: 1px solid #e6e5e1; }}
  #stage {{ position: absolute; top: 37px; bottom: 0; left: 0; right: 0; overflow: hidden; cursor: grab; }}
  #stage.dragging {{ cursor: grabbing; }}
  #img {{ position: absolute; transform-origin: 0 0; user-select: none; -webkit-user-drag: none; }}
</style>
</head>
<body>
<div id="bar">{title} &mdash; scroll to zoom, drag to pan, double-click to reset</div>
<div id="stage"><img id="img" alt="{title}" src="{local}"
  onerror="if (!this.dataset.f) {{ this.dataset.f = 1; this.src = '{remote}'; }}"></div>
<script>
(function () {{
  var stage = document.getElementById("stage"), img = document.getElementById("img");
  var s = 1, x = 0, y = 0, drag = null;
  function apply() {{ img.style.transform = "translate(" + x + "px," + y + "px) scale(" + s + ")"; }}
  function fit() {{
    var w = img.naturalWidth, h = img.naturalHeight;
    if (!w || !h) return;
    s = Math.min(stage.clientWidth / w, stage.clientHeight / h);
    x = (stage.clientWidth - w * s) / 2; y = (stage.clientHeight - h * s) / 2; apply();
  }}
  img.addEventListener("load", fit); window.addEventListener("resize", fit);
  stage.addEventListener("wheel", function (e) {{
    e.preventDefault();
    var r = stage.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    var k = Math.exp(-e.deltaY * 0.0015), ns = Math.min(Math.max(s * k, 0.05), 40);
    x = px - (px - x) * ns / s; y = py - (py - y) * ns / s; s = ns; apply();
  }}, {{ passive: false }});
  stage.addEventListener("pointerdown", function (e) {{
    drag = {{ px: e.clientX, py: e.clientY, x: x, y: y }};
    stage.classList.add("dragging"); stage.setPointerCapture(e.pointerId);
  }});
  stage.addEventListener("pointermove", function (e) {{
    if (!drag) return; x = drag.x + e.clientX - drag.px; y = drag.y + e.clientY - drag.py; apply();
  }});
  stage.addEventListener("pointerup", function () {{ drag = null; stage.classList.remove("dragging"); }});
  stage.addEventListener("dblclick", fit);
  if (img.complete) fit();
}})();
</script>
</body>
</html>
"""


def write_image_viewer(image: Path) -> Path:
    """Write ``<image>.html``: a zoomable, pannable view of ``image``."""
    rel = image.relative_to(DOCS_DIR).as_posix()
    title = image.stem.replace("_", " ")
    page = VIEWER_TEMPLATE.format(
        title=html.escape(title),
        local=html.escape(image.name),
        remote=html.escape(f"{RAW_DOCS_URL}/{rel}"),
    )
    target = image.with_suffix(".html")
    target.write_text(page, encoding="utf-8")
    return target


def main() -> int:
    for stem, values, y_title in PANELS:
        save_figure(paper_panel(values, y_title), PAPER_DIR / stem, formats=("pdf",), size=(800, 420))
        save_figure(
            grouped_bar_figure(PATHS, values, y_title=y_title),
            DOCS_DIR / stem, formats=("png", "html"), embed_plotlyjs=True,
        )
        print(f"{stem}: {PAPER_DIR / stem}.pdf, {DOCS_DIR / stem}.{{png,html}}")
    images = [DOCS_DIR / f"{stem}.png" for stem in EXPERIMENT_IMAGES]
    images += sorted(p for p in GALLERY_DIR.rglob("*") if p.suffix in (".png", ".jpg"))
    for image in images:
        write_image_viewer(image)
    print(f"image viewers: {len(images)} pages beside the images in {DOCS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
