"""Plotly figures for evaluation and training results.

Every figure is written in three formats by :func:`save_figure`: PNG for
documents, PDF for publication, and HTML for an interactive version that can be
panned, zoomed, and hovered in a browser. PNG and PDF export go through
``kaleido``.

The font family, font size, Plotly template, and the number of decimals shown
on every value are read from the JSON ``config`` file in the repo root, so the
whole figure set restyles from one place.

Figures carry no title and no subtitle. They are destined for papers and
reports where the caption names the figure, so a baked-in title would duplicate
that caption and eat vertical space. Axis labels and the legend do the
explanatory work; the conditions a result was produced under are recorded in
the accompanying ``results.json`` and in the caller's console output.

The categorical colours below were validated for colour-vision deficiency
across all pairs (worst CVD ΔE 9.2, worst normal-vision ΔE 16.3 in OKLab×100),
so the four controllers stay distinguishable when overlaid on one axis. Aqua
sits below 3:1 contrast on the light surface, so identity never rests on colour
alone: every figure carries a legend, bars are labelled with their value, and
the analysis writes a summary table alongside the charts. End-of-line labels
were tried and removed -- on paths where the controllers converge they pile up
into an unreadable blob, which is worse than the legend alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import plotly.graph_objects as go

from asv_rl import settings

#: Output formats written for every figure.
FIGURE_FORMATS = ("png", "html", "pdf")

#: Fixed colour per controller. Assigned by entity, never by rank, so a figure
#: that drops a controller does not repaint the others.
ALGORITHM_COLORS = {
    "dqn_3_action": "#2a78d6",   # blue
    "dqn_5_action": "#eb6834",   # orange
    "ddpg": "#1baf7a",           # aqua
    "ppo": "#4a3aa7",            # violet
}

#: Human-readable names used in legends and axis labels.
ALGORITHM_LABELS = {
    "dqn_3_action": "DQN (3 actions)",
    "dqn_5_action": "DQN (5 actions)",
    "ddpg": "DDPG",
    "ppo": "PPO",
}

#: Typography and Plotly template come from the JSON ``config`` file in the
#: repo root, so every figure in the project changes together.
FONT_FAMILY = settings.get("font_family")
FONT_SIZE = int(settings.get("font_size"))
PLOTLY_TEMPLATE = settings.get("plotly_template")

#: Decimals shown on every number a figure draws: axis ticks, bar value labels,
#: and hover readouts. Set ``decimal_places`` in the config file to change it.
DECIMALS = int(settings.get("decimal_places"))

#: Python format spec and its Plotly (d3) equivalent for that precision.
VALUE_FORMAT = f".{DECIMALS}f"
TICK_FORMAT = f".{DECIMALS}f"

#: Secondary text (legend, tick labels, bar value labels) sits two points below
#: the body size so the axis labels stay the dominant text. Change the offset
#: here; change the base size with ``font_size`` in the config file.
SMALL_FONT_SIZE = max(FONT_SIZE - 8, 6)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
REFERENCE = "#9a9992"

_FALLBACK_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e34948")


def color_for(key: str, index: int = 0) -> str:
    """Colour for a controller, falling back to the categorical order."""
    return ALGORITHM_COLORS.get(key, _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)])


def label_for(key: str) -> str:
    return ALGORITHM_LABELS.get(key, key)


def base_layout(
    x_title: str,
    y_title: str,
    x_tickformat: str | None = None,
    y_tickformat: str | None = TICK_FORMAT,
    **kwargs,
) -> dict:
    """Shared layout: recessive grid and axes, legend above the plot.

    No title is set. The top margin leaves room for the legend band only.

    Tick labels on the value axis carry ``DECIMALS`` decimals. The x axis is
    left on Plotly's automatic format by default, because it usually counts
    steps and ``0.00, 100.00`` reads worse than ``0, 100``; figures whose x
    axis is a measured quantity pass ``x_tickformat=TICK_FORMAT``.
    """
    layout = dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE, color=TEXT_SECONDARY),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=FONT_SIZE)),
            gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
            ticks="outside", tickcolor=GRID, ticklen=4,
            tickfont=dict(size=SMALL_FONT_SIZE), tickformat=x_tickformat,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=FONT_SIZE)),
            gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
            ticks="outside", tickcolor=GRID, ticklen=4,
            tickfont=dict(size=SMALL_FONT_SIZE), tickformat=y_tickformat,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=SMALL_FONT_SIZE),
        ),
        margin=dict(l=80, r=40, t=60, b=60),
        hovermode="closest",
    )
    layout.update(kwargs)
    return layout


def save_figure(
    fig: go.Figure,
    stem: str | Path,
    formats: Iterable[str] = FIGURE_FORMATS,
    embed_plotlyjs: bool = False,
    size: tuple[int, int] = (1000, 700),
) -> list[Path]:
    """Write ``fig`` as ``<stem>.<ext>`` for each requested format.

    By default HTML references plotly.js from a CDN, which keeps each file a
    few tens of kilobytes instead of several megabytes. Pass
    ``embed_plotlyjs=True`` for figures linked from the README: the
    htmlpreview viewer renders only HTML with plotly.js embedded.
    ``size`` is the (width, height) in pixels of the PNG and PDF export.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    written = []
    for fmt in formats:
        target = stem.with_suffix(f".{fmt}")
        if fmt == "html":
            fig.write_html(
                str(target),
                include_plotlyjs=True if embed_plotlyjs else "cdn",
                full_html=True,
                config={"displaylogo": False, "responsive": True},
            )
        elif fmt in ("png", "pdf"):
            try:
                fig.write_image(str(target), width=size[0], height=size[1], scale=2 if fmt == "png" else 1)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Could not write {target}. Static export needs the 'kaleido' package "
                    f"alongside plotly (uv add kaleido). Original error: {exc}"
                ) from exc
        else:
            raise ValueError(f"unsupported figure format {fmt!r}; expected one of {FIGURE_FORMATS}")
        written.append(target)
    return written


def track_figure(
    reference_x: Sequence[float],
    reference_y: Sequence[float],
    waypoints_x: Sequence[float],
    waypoints_y: Sequence[float],
    tracks: Mapping[str, tuple[Sequence[float], Sequence[float]]],
) -> go.Figure:
    """Paths followed by one or more controllers against the reference path.

    Axes are equal-scale and the y axis is reversed, matching the north-east-down
    convention the dynamics are written in.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(reference_x), y=list(reference_y), mode="lines", name="Reference path",
        line=dict(color=REFERENCE, width=2, dash="dash"),
        hovertemplate=f"Reference<br>X/L %{{x:{VALUE_FORMAT}}}<br>Y/L %{{y:{VALUE_FORMAT}}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=list(waypoints_x), y=list(waypoints_y), mode="markers", name="Waypoints",
        marker=dict(color=TEXT_SECONDARY, size=8, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate=(
            f"Waypoint %{{pointNumber}}<br>X/L %{{x:{VALUE_FORMAT}}}"
            f"<br>Y/L %{{y:{VALUE_FORMAT}}}<extra></extra>"
        ),
    ))

    for index, (key, (xs, ys)) in enumerate(tracks.items()):
        color = color_for(key, index)
        name = label_for(key)
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys), mode="lines", name=name,
            line=dict(color=color, width=2),
            hovertemplate=(
                f"{name}<br>X/L %{{x:{VALUE_FORMAT}}}<br>Y/L %{{y:{VALUE_FORMAT}}}<extra></extra>"
            ),
        ))

    fig.update_layout(**base_layout("X / L", "Y / L", x_tickformat=TICK_FORMAT))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig


def series_figure(
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    x_title: str,
    y_title: str,
    zero_line: bool = False,
) -> go.Figure:
    """One line per controller: cross-track error, rudder angle, returns."""
    fig = go.Figure()
    for index, (key, (xs, ys)) in enumerate(series.items()):
        name = label_for(key)
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys), mode="lines", name=name,
            line=dict(color=color_for(key, index), width=2),
            hovertemplate=f"{name}<br>%{{x}}<br>%{{y:{VALUE_FORMAT}}}<extra></extra>",
        ))

    fig.update_layout(**base_layout(x_title, y_title, hovermode="x unified"))
    if zero_line:
        fig.add_hline(y=0, line=dict(color=TEXT_SECONDARY, width=1, dash="dot"))
    return fig


def grouped_bar_figure(
    categories: Sequence[str],
    values: Mapping[str, Sequence[float]],
    y_title: str,
) -> go.Figure:
    """One bar group per reference path, one bar per controller."""
    fig = go.Figure()
    for index, (key, series) in enumerate(values.items()):
        name = label_for(key)
        fig.add_trace(go.Bar(
            x=list(categories), y=list(series), name=name,
            marker=dict(
                color=color_for(key, index),
                line=dict(color=SURFACE, width=2),  # 2px surface gap between bars
                cornerradius=4,
            ),
            hovertemplate=f"{name}<br>%{{x}}<br>%{{y:{VALUE_FORMAT}}}<extra></extra>",
            text=[f"{v:{VALUE_FORMAT}}" if v is not None else "" for v in series],
            textposition="outside",
            textfont=dict(size=SMALL_FONT_SIZE, color=TEXT_SECONDARY),
            # Plotly shrinks outside labels to the width of their own bar by
            # default, so "0.0297" would render smaller than "0.112" on the
            # same chart. Both settings below switch that off: every value
            # label is drawn at SMALL_FONT_SIZE regardless of its length.
            constraintext="none",
            cliponaxis=False,
        ))

    fig.update_layout(**base_layout(
        "", y_title, barmode="group", bargap=0.4, bargroupgap=0.08,
        uniformtext=dict(mode="show", minsize=SMALL_FONT_SIZE),
    ))
    return fig


def table_figure(headers: Sequence[str], columns: Sequence[Sequence]) -> go.Figure:
    """The table view that accompanies the charts, so no result is colour-only."""
    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#f0efec", align="left",
            font=dict(color=TEXT_PRIMARY, size=FONT_SIZE), height=FONT_SIZE + 12,
        ),
        cells=dict(
            values=[list(c) for c in columns],
            fill_color=SURFACE, align="left",
            font=dict(color=TEXT_SECONDARY, size=SMALL_FONT_SIZE), height=FONT_SIZE + 8,
            line=dict(color=GRID, width=1),
        ),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=SURFACE,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def training_curve_figure(values: Sequence[float], y_title: str, x_title: str = "Episode") -> go.Figure:
    """A single training curve, used for per-run return and loss plots."""
    fig = go.Figure(go.Scatter(
        x=list(range(len(values))), y=list(values), mode="lines",
        line=dict(color=_FALLBACK_COLORS[0], width=2),
        hovertemplate=f"{x_title} %{{x}}<br>%{{y:{VALUE_FORMAT}}}<extra></extra>",
        showlegend=False,
    ))
    # A single series needs no legend box, and no title: the axis label says
    # what is plotted.
    fig.update_layout(**base_layout(x_title, y_title, showlegend=False))
    return fig
