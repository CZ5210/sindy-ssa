"""Statistical / box-plot visualisations. / 统计箱线图。"""

from __future__ import annotations

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from scipy.stats import gaussian_kde

from .style import FONT_SIZE, LEGEND_FONT_SIZE, METHOD_COLORS, save_figure

_COLORS = ["lightcoral", "skyblue", "palegreen", "peachpuff", "plum", "lightyellow", "lightgray"]


def _log_y_integer_powers(ax) -> None:
    """Switch ``ax`` to a log y-axis labelled only at integer powers of ten.

    Matplotlib otherwise falls back to minor-tick labels (``6×10^0``,
    ``2×10^1`` …) whenever a panel spans less than one decade, which is noisy for
    the batch error panels.  Forcing the major locator to the powers of ten and
    muting the minor labels keeps the annotations to ``10^n`` only.  The axis
    range itself is left automatic so the data fill the panel; when a panel spans
    less than one decade it may therefore show a single ``10^n`` label.
    """
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_violins(data: pd.DataFrame, values, group: str = "method",
                 save_path=None, title=None, ylabels=None, show=False):
    """Violin plots of one or more ``values`` columns, grouped by ``group``.

    One subplot per column, so e.g. ``["epe", "epe_planar"]`` shows the 3D vs 2D
    end-position-error distributions side by side.
    """
    groups = [g for g in dict.fromkeys(data[group].astype(str))
              if data.loc[data[group].astype(str) == g, values[0]].notna().any()]
    n = len(values)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), squeeze=False)
    axes = axes[0]
    for ax, value in zip(axes, values):
        samples = [
            pd.to_numeric(data.loc[data[group].astype(str) == g, value], errors="coerce").dropna().values
            for g in groups
        ]
        parts = ax.violinplot(samples, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], itertools.cycle(_COLORS)):
            body.set_facecolor(color)
            body.set_edgecolor("k")
            body.set_alpha(0.7)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=15, ha="right")
        ylabel = (ylabels or {}).get(value, value)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_metric_violins(data: pd.DataFrame, group: str = "method",
                        save_path=None, title=None, show=False):
    """Merged violin comparison of both error metrics, 3D vs 2D per method.

    Single two-row figure matching the style of :func:`plot_error_metrics`: the
    top row is the end-position error :math:`\\mathcal{EPE}` and the bottom row
    the thrust error integral :math:`\\mathcal{EI}`.  Within each row every method
    gets two violins sharing the method's colour---the 3D variant (solid) and the
    planar 2D variant (hatched)---with a shared 3D/2D legend.
    """
    # (panel y-label, [(column, hatch), ...]) per row.
    panels = [
        (r"$\mathcal{EPE}$ (km)", [("epe", ""), ("epe_planar", "//")]),
        (r"$\mathcal{EI}$ (m/s)", [("ei_total", ""), ("ei_planar", "//")]),
    ]
    methods = [
        m
        for m in dict.fromkeys(data[group].astype(str))
        if pd.to_numeric(data.loc[data[group].astype(str) == m, "epe"], errors="coerce").notna().any()
    ]
    n = len(methods)
    x = np.arange(n)
    width = 0.38
    # Same main tone as the RTN / thrust / error-metric panels for each method.
    colors = [METHOD_COLORS.get(m, _COLORS[i % len(_COLORS)]) for i, m in enumerate(methods)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, (ylabel, variants) in zip(axes, panels):
        for j, (col, hatch) in enumerate(variants):
            for i, method in enumerate(methods):
                samples = pd.to_numeric(
                    data.loc[data[group].astype(str) == method, col], errors="coerce"
                ).dropna().values
                if samples.size == 0:
                    continue
                pos = x[i] + (j - 0.5) * width
                parts = ax.violinplot(
                    [samples], positions=[pos], widths=width * 0.95,
                    showmedians=True, showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(colors[i])
                    body.set_edgecolor("k")
                    body.set_alpha(0.7)
                    if hatch:
                        body.set_hatch(hatch)
                if "cmedians" in parts:
                    parts["cmedians"].set_color("black")
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE + 4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.tick_params(axis="x", labelsize=FONT_SIZE)
        ax.tick_params(axis="y", labelsize=FONT_SIZE + 4)
        ax.grid(True, axis="y", alpha=0.3)

    proxies = [
        Patch(facecolor="0.8", edgecolor="k", hatch="", label="3D"),
        Patch(facecolor="0.8", edgecolor="k", hatch="//", label="2D"),
    ]
    fig.legend(
        handles=proxies,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        borderaxespad=0.0,
        fontsize=LEGEND_FONT_SIZE + 4,
    )

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_metric_overview(case_data, noise_cases, ideal_case=None,
                         save_path=None, title=None, show=False, log_scale=False):
    """Cross-noise overview of both metrics, 3D vs 2D per method.

    Two rows (EPE top, EI bottom) by up to four columns: the deterministic
    ``ideal_case`` (rendered as bars, since it has a single value) followed by the
    Monte-Carlo noise tiers (rendered as violins).  Every panel shows the 3D
    (solid) and 2D (hatched) variants per method, coloured like
    :func:`plot_error_metrics`.  Each panel has its own linear y-axis so the
    widely different magnitudes remain readable.  No figure title is drawn.
    """
    rows = [
        (r"$\mathcal{EPE}$ (km)", "epe", "epe_planar"),
        (r"$\mathcal{EI}$ (m/s)", "ei_total", "ei_planar"),
    ]
    noise_cases = [c for c in noise_cases if c in case_data]
    has_ideal = bool(ideal_case) and ideal_case in case_data
    columns = ([ideal_case] if has_ideal else []) + noise_cases
    if not columns:
        return save_path

    ref = case_data[noise_cases[0]] if noise_cases else case_data[ideal_case]
    methods = list(dict.fromkeys(ref["method"].astype(str)))
    n = len(methods)
    x = np.arange(n)
    width = 0.38
    colors = [METHOD_COLORS.get(m, _COLORS[i % len(_COLORS)]) for i, m in enumerate(methods)]
    ideal_df = case_data.get(ideal_case) if has_ideal else None

    fig, axes = plt.subplots(2, len(columns), figsize=(4.4 * len(columns), 9), squeeze=False)
    for r, (ylabel, col3d, col2d) in enumerate(rows):
        for j, case in enumerate(columns):
            ax = axes[r][j]
            if has_ideal and case == ideal_case:
                # Deterministic ideal: single value per method -> grouped bars.
                for k, (comp, hatch) in enumerate(((col3d, ""), (col2d, "//"))):
                    heights = []
                    for method in methods:
                        value = pd.to_numeric(
                            ideal_df.loc[ideal_df["method"].astype(str) == method, comp],
                            errors="coerce",
                        ).dropna()
                        heights.append(float(value.iloc[0]) if not value.empty else 0.0)
                    offset = (k - 0.5) * width
                    ax.bar(x + offset, heights, width, color=colors,
                           edgecolor="k", alpha=0.7, hatch=hatch)
            else:
                df = case_data[case]
                for k, (comp, hatch) in enumerate(((col3d, ""), (col2d, "//"))):
                    for i, method in enumerate(methods):
                        samples = pd.to_numeric(
                            df.loc[df["method"].astype(str) == method, comp], errors="coerce"
                        ).dropna().values
                        if samples.size == 0:
                            continue
                        pos = x[i] + (k - 0.5) * width
                        parts = ax.violinplot(
                            [samples], positions=[pos], widths=width * 0.95,
                            showmedians=True, showextrema=False,
                        )
                        for body in parts["bodies"]:
                            body.set_facecolor(colors[i])
                            body.set_edgecolor("k")
                            body.set_alpha(0.7)
                            if hatch:
                                body.set_hatch(hatch)
                        if "cmedians" in parts:
                            parts["cmedians"].set_color("black")

            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=FONT_SIZE - 2)
            ax.set_title(case, fontsize=FONT_SIZE + 4)
            ax.grid(True, axis="y", alpha=0.3)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=FONT_SIZE + 4)
            ax.tick_params(axis="y", labelsize=FONT_SIZE)
            if log_scale:
                _log_y_integer_powers(ax)

    method_proxies = [
        Patch(facecolor=colors[i], edgecolor="k", alpha=0.7, label=methods[i])
        for i in range(n)
    ]
    style_proxies = [
        Patch(facecolor="0.8", edgecolor="k", hatch="", label="3D"),
        Patch(facecolor="0.8", edgecolor="k", hatch="//", label="2D"),
        Patch(facecolor="white", edgecolor="k", hatch="", label="ideal"),
    ]
    fig.legend(
        handles=method_proxies + style_proxies,
        loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.0,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


_TIER_LABELS = {"high": "High", "low": "Low", "reduced": "Reduced"}
_HALF_WIDTH = 0.42


def _draw_box(ax, samples, pos: float, color, width: float) -> None:
    """Draw a boxplot (1.5x IQR whiskers, fliers shown) centred at ``pos``."""
    parts = ax.boxplot(
        [samples], positions=[pos], widths=width, patch_artist=True,
        showfliers=True, manage_ticks=False, zorder=3,
    )
    for patch in parts["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(0.7)
    for median in parts["medians"]:
        median.set_color("k")
    for flier in parts["fliers"]:
        flier.set(
            marker="o", markersize=3, markerfacecolor="none",
            markeredgecolor=color, alpha=0.6, linestyle="none",
        )


def _draw_half_violin(ax, samples, pos: float, color, width: float) -> None:
    """Draw only the left half of a KDE violin anchored at ``pos``."""
    values = np.asarray(samples, dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if values.size < 2 or hi - lo <= 0.0:
        # Degenerate sample (single value / all equal): a thin horizontal bar.
        ax.plot([pos - width, pos], [lo, lo], color=color, lw=2.0, zorder=2)
        return
    pad = 0.05 * (hi - lo)
    y = np.linspace(lo - pad, hi + pad, 200)
    density = gaussian_kde(values)(y)
    density = density / density.max()
    ax.fill_betweenx(
        y, pos - width * density, pos,
        facecolor=color, edgecolor="k", linewidth=0.8, alpha=0.7, zorder=2,
    )


def _draw_half(ax, samples, pos: float, color) -> None:
    """Draw the split glyph at ``pos``: violin density left, box+fliers right."""
    _draw_half_violin(ax, samples, pos, color, _HALF_WIDTH)
    _draw_box(ax, samples, pos + _HALF_WIDTH / 2.0, color, _HALF_WIDTH)


def plot_metric_tiers(case_data, noise_cases, ideal_case=None,
                      save_path=None, log_scale=True, show=False, png=True):
    """Tier-separated batch overview: a 4xN grid of metric panels.

    Rows are the four metric variants---:math:`\\mathcal{EPE}_{3D}`,
    :math:`\\mathcal{EPE}_{2D}`, :math:`\\mathcal{EI}_{3D}` and
    :math:`\\mathcal{EI}_{2D}`---and columns the noise tiers (typically High /
    Low / Reduced).  Every panel therefore gets its own y-axis, which keeps the
    widely different tier magnitudes readable.  Within a panel the methods sit
    side by side and are told apart by their shared colours, each drawn as a
    split glyph---the left half is the KDE violin and the right half a boxplot
    with the outlying runs shown as fliers.  The method legend is drawn once on
    the right.  ``log_scale`` selects a logarithmic y-axis and ``png`` also
    writes a raster preview next to the PDF.  The deterministic ideal case is
    intentionally not drawn (it lies far below the noisy tiers).
    """
    panels = [
        (r"$\mathcal{EPE}_{\mathrm{3D}}$ (km)", "epe"),
        (r"$\mathcal{EPE}_{\mathrm{2D}}$ (km)", "epe_planar"),
        (r"$\mathcal{EI}_{\mathrm{3D}}$ (m/s)", "ei_total"),
        (r"$\mathcal{EI}_{\mathrm{2D}}$ (m/s)", "ei_planar"),
    ]
    noise_cases = [c for c in noise_cases if c in case_data]
    if not noise_cases:
        return save_path

    methods = list(dict.fromkeys(case_data[noise_cases[0]]["method"].astype(str)))
    n = len(methods)
    n_tiers = len(noise_cases)
    colors = [METHOD_COLORS.get(m, _COLORS[i % len(_COLORS)]) for i, m in enumerate(methods)]
    x = np.arange(n)

    fig, axes = plt.subplots(
        len(panels), n_tiers, figsize=(4.4 * n_tiers, 12.0),
        squeeze=False, sharex=False, sharey=False,
    )
    for r, (title, col) in enumerate(panels):
        for c, case in enumerate(noise_cases):
            ax = axes[r][c]
            df = case_data[case]
            for i, method in enumerate(methods):
                samples = pd.to_numeric(
                    df.loc[df["method"].astype(str) == method, col], errors="coerce"
                ).dropna().values
                if samples.size == 0:
                    continue
                _draw_half(ax, samples, x[i], colors[i])

            ax.set_xticks(x)
            ax.set_xticklabels([])
            ax.set_xlim(-0.7, n - 0.3)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="y", labelsize=FONT_SIZE - 2)
            if log_scale:
                _log_y_integer_powers(ax)
            # Metric label down the left edge, tier label across the top row.
            if c == 0:
                ax.set_ylabel(title, fontsize=FONT_SIZE + 2)
            if r == 0:
                ax.set_title(_TIER_LABELS.get(case, case), fontsize=FONT_SIZE + 4)

    method_proxies = [
        Patch(facecolor=colors[i], edgecolor="k", alpha=0.7, label=methods[i])
        for i in range(n)
    ]
    fig.legend(
        handles=method_proxies,
        loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.0,
        fontsize=LEGEND_FONT_SIZE + 2,
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_figure(fig, save_path, png=png)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


_WINDOW_MARKERS = ["o", "s", "^", "D", "v", "P"]


def plot_metric_vs_window(case_data, windows, tiers, ideal_data=None, save_path=None,
                          methods=None, title=None, show=False, png=True, log_scale=True):
    """Convergence with the observed fraction: a 2xN metric grid.

    Rows are the planar 2D metrics (:math:`\\mathcal{EPE}_{2D}` and
    :math:`\\mathcal{EI}_{2D}`) and columns the noise tiers.  ``case_data`` maps
    ``window -> tier -> long evaluation DataFrame``; for every (window, tier,
    method) the marker is the median over the Monte-Carlo runs and the shaded
    band the interquartile range, joined across the windows given in
    ``windows`` (e.g. 25, 50, 75).  When ``ideal_data`` (``window -> DataFrame``)
    is supplied an extra deterministic ``Ideal`` column is appended, drawn as
    single markers with no band.  Methods share the global colour palette;
    ``log_scale`` selects a logarithmic y-axis.
    """
    rows = [
        (r"$\mathcal{EPE}_{\mathrm{2D}}$ (km)", "epe_planar"),
        (r"$\mathcal{EI}_{\mathrm{2D}}$ (m/s)", "ei_planar"),
    ]
    windows = list(windows)
    tiers = [t for t in tiers if any(t in case_data.get(w, {}) for w in windows)]
    columns = [(t, False) for t in tiers]
    if ideal_data:
        columns.append(("ideal", True))
    if not windows or not columns:
        return save_path

    ref = None
    for w in windows:
        for name, is_ideal in columns:
            df = ideal_data.get(w) if is_ideal else case_data.get(w, {}).get(name)
            if df is not None:
                ref = df
                break
        if ref is not None:
            break
    if ref is None:
        return save_path

    ordered = list(methods) if methods else list(dict.fromkeys(ref["method"].astype(str)))
    colors = [METHOD_COLORS.get(m, _COLORS[i % len(_COLORS)]) for i, m in enumerate(ordered)]
    markers = [_WINDOW_MARKERS[i % len(_WINDOW_MARKERS)] for i in range(len(ordered))]

    fig, axes = plt.subplots(
        len(rows), len(columns), figsize=(4.4 * len(columns), 8.6),
        squeeze=False, sharex=False, sharey=False,
    )
    for r, (ylabel, col) in enumerate(rows):
        for c, (name, is_ideal) in enumerate(columns):
            ax = axes[r][c]
            for i, method in enumerate(ordered):
                xs, meds, q1s, q3s = [], [], [], []
                for w in windows:
                    df = ideal_data.get(w) if is_ideal else case_data.get(w, {}).get(name)
                    if df is None:
                        continue
                    values = pd.to_numeric(
                        df.loc[df["method"].astype(str) == method, col], errors="coerce"
                    ).dropna().values
                    if values.size == 0:
                        continue
                    xs.append(w)
                    meds.append(float(np.median(values)))
                    q1s.append(float(np.percentile(values, 25)))
                    q3s.append(float(np.percentile(values, 75)))
                if not xs:
                    continue
                ax.plot(xs, meds, marker=markers[i], color=colors[i],
                        linewidth=1.8, markersize=7, label=method, zorder=3)
                if not is_ideal:
                    ax.fill_between(xs, q1s, q3s, color=colors[i], alpha=0.2, linewidth=0)

            ax.set_xticks(windows)
            ax.set_xticklabels([str(w) for w in windows], fontsize=FONT_SIZE - 2)
            ax.set_xlabel("Observed fraction (%)", fontsize=FONT_SIZE + 2)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=FONT_SIZE + 2)
            if r == 0:
                label = "Ideal" if is_ideal else _TIER_LABELS.get(name, name)
                ax.set_title(label, fontsize=FONT_SIZE + 4)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="y", labelsize=FONT_SIZE - 2)
            if log_scale:
                _log_y_integer_powers(ax)

    proxies = [
        plt.Line2D([0], [0], color=colors[i], marker=markers[i], linewidth=1.8,
                   markersize=7, label=ordered[i])
        for i in range(len(ordered))
    ]
    fig.legend(
        handles=proxies, loc="center left", bbox_to_anchor=(1.0, 0.5),
        borderaxespad=0.0, fontsize=LEGEND_FONT_SIZE + 2,
    )
    if title:
        fig.suptitle(title, fontsize=FONT_SIZE + 4)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_figure(fig, save_path, png=png)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def summarise(data: pd.DataFrame, value: str, methods) -> pd.DataFrame:
    """Median / interquartile summary per method (reviewer-friendly table)."""
    rows = []
    for m in methods:
        values = data.loc[data["method"] == m, value].values
        rows.append(
            {
                "method": m,
                "median": float(np.median(values)),
                "q1": float(np.percentile(values, 25)),
                "q3": float(np.percentile(values, 75)),
                "mean": float(np.mean(values)),
            }
        )
    return pd.DataFrame(rows)


_HEATMAP_TIERS = ("high", "low", "reduced")
_HEATMAP_TIER_LABELS = {"high": "High", "low": "Low", "reduced": "Reduced"}
# Only the coloured data cells carry white text; every other cell (method
# names, headers, group labels) stays white with black text.  The two colour
# maps are truncated so that even their lightest end is a mid tone (no bright
# white), which keeps the white value text readable.
_HEATMAP_LABEL_COLOR = "black"
_HEATMAP_TEXT_COLOR = "white"
_HEATMAP_EDGE_COLOR = "#8c8c8c"
_HEATMAP_MEDIAN_RANGE = (0.45, 1.0)
_HEATMAP_IQR_RANGE = (0.45, 1.0)


def _summary_fmt(value) -> str:
    """Format a summary statistic with a fixed three decimal places.

    Falls back to ``--`` for missing values (e.g. the interquartile range of the
    single-run ideal case).
    """
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value):
        return "--"
    return f"{value:.3f}"


def _metric_values(group, methods, column):
    """Finite values of one metric column over the methods of a row group."""
    values = []
    for method in methods:
        raw = group["data"].get(method, {}).get(column[0], {}).get(column[1])
        if raw is not None and np.isfinite(float(raw)):
            values.append(float(raw))
    return values


def plot_summary_heatmap(row_groups, save_path=None,
                         median_cmap="OrRd", iqr_cmap="Blues",
                         median_range=None, iqr_range=None,
                         fontsize=8.5, method_col_width=1.9, data_col_width=1.0,
                         row_height=1.0, show=False, png=True):
    """Coloured "Median + IQR" summary table (heatmap-style figure).

    Renders a table into a matplotlib figure so it can replace the hand-written
    result tables.  The columns are grouped by statistic: all median columns
    (the three noise tiers plus the deterministic ideal reference) are packed on
    the left under a single ``Median`` header, and the three interquartile-range
    columns on the right under ``IQR``.  Median cells are tinted with a warm
    colour map and IQR cells with a cool one, so the reader can tell an "error
    level" from a "scatter level" at a glance.

    ``row_groups`` is a list of dicts, one per metric block::

        {
            "label": r"$\\mathcal{EPE}_{3D}$ (km)",
            "methods": ["MEEFD-SG", ...],
            "data": {
                method: {
                    "high":    {"median": 16.0, "iqr": 3.5},
                    "low":     {"median": 3.50, "iqr": None},
                    "reduced": {"median": 0.397, "iqr": 0.2},
                    "ideal":   {"median": 0.0261},   # optional, no IQR
                },
                ...
            },
        }

    Colour scaling is applied once per metric row group and per statistic: all
    median cells of a block (the three noise tiers *and* the ideal reference)
    share one warm scale, and all IQR cells of the block share one cool scale.
    Values are mapped through :math:`\\log_{10}` so that a larger value is
    darker.  The smallest value in each column is also bolded (matching the old
    tables); colour and bold therefore encode complementary information, and the
    numbers remain legible without colour.  The colour maps are truncated to
    ``median_range`` / ``iqr_range`` (fractions of the full map) so their lightest
    end is a mid tone rather than white, which keeps the white value text
    readable on every cell; all non-data cells stay white with black text.  Two
    legend colour bars for the Median (warm) and IQR (cool) scales are drawn
    below the table.
    """
    if median_range is None:
        median_range = _HEATMAP_MEDIAN_RANGE
    if iqr_range is None:
        iqr_range = _HEATMAP_IQR_RANGE
    tiers = _HEATMAP_TIERS
    has_ideal = any(
        "ideal" in mdata
        for group in row_groups
        for mdata in group["data"].values()
    )

    # Median block on the left (tiers + ideal), IQR block on the right.
    median_cols = [(tier, "median") for tier in tiers]
    if has_ideal:
        median_cols.append(("ideal", "median"))
    iqr_cols = [(tier, "iqr") for tier in tiers]
    data_cols = median_cols + iqr_cols

    def _truncate(cmap, lo, hi):
        base = colormaps[cmap]
        return LinearSegmentedColormap.from_list(
            f"{cmap}_{lo:.2f}_{hi:.2f}", base(np.linspace(lo, hi, 256))
        )

    median_cmap_obj = _truncate(median_cmap, *median_range)
    iqr_cmap_obj = _truncate(iqr_cmap, *iqr_range)

    groups = []
    for group in row_groups:
        methods = list(group.get("methods") or group["data"].keys())

        per_column = {
            column: _metric_values(group, methods, column) for column in data_cols
        }
        # One scale per metric and per statistic (median / IQR), so a cell's
        # shade is comparable across the tiers and the ideal reference inside the
        # same row group rather than being renormalised in every column.
        scales = {}
        for stat, columns in (("median", median_cols), ("iqr", iqr_cols)):
            values = [v for column in columns for v in per_column[column]]
            scales[stat] = (min(values), max(values)) if values else (None, None)

        cells = {}
        for method in methods:
            cells[method] = {}
            for column in data_cols:
                tier, stat = column
                raw = group["data"].get(method, {}).get(tier, {}).get(stat)
                lo, hi = scales[stat]
                if raw is None or not np.isfinite(float(raw)) or lo is None:
                    cells[method][column] = ("--", "white", False, False)
                    continue
                value = float(raw)
                if hi > lo:
                    pos = (np.log10(max(value, 1e-12)) - np.log10(max(lo, 1e-12))) / (
                        np.log10(max(hi, 1e-12)) - np.log10(max(lo, 1e-12))
                    )
                else:
                    pos = 0.5
                cmap = median_cmap_obj if stat == "median" else iqr_cmap_obj
                facecolor = cmap(float(np.clip(pos, 0.0, 1.0)))
                column_values = per_column[column]
                best = bool(column_values) and value == min(column_values)
                cells[method][column] = (_summary_fmt(value), facecolor, True, best)
        groups.append({"label": group["label"], "methods": methods, "cells": cells})

    n_body = sum(1 + len(group["methods"]) for group in groups)
    n_rows = 2 + n_body

    x_edges = [0.0, method_col_width]
    for _ in data_cols:
        x_edges.append(x_edges[-1] + data_col_width)
    total_width = x_edges[-1]

    fig, ax = plt.subplots(figsize=(total_width * 0.95, max(n_rows * row_height * 0.28, 1.8)))
    ax.set_xlim(0.0, total_width)
    ax.set_ylim(n_rows, 0.0)
    ax.axis("off")

    def cell(x0, y0, width, height, facecolor, edgecolor=_HEATMAP_EDGE_COLOR,
             linewidth=0.7):
        ax.add_patch(
            Rectangle((x0, y0), width, height, facecolor=facecolor,
                      edgecolor=edgecolor, linewidth=linewidth)
        )

    # Two-row header: the statistic name ("Median" / "IQR") spans its block of
    # columns, then each sub-column is labelled by tier (or "Ideal").  The method
    # column spans both header rows.  Header / method / label cells stay white
    # with black text; only the coloured data cells use white text.
    cell(0.0, 0.0, method_col_width, 2.0, "white")
    ax.text(method_col_width / 2.0, 1.0, "Method", ha="center", va="center",
            fontsize=fontsize + 1, fontweight="bold", color=_HEATMAP_LABEL_COLOR)

    n_median = len(median_cols)
    for label, j0, j1 in (("Median", 0, n_median), ("IQR", n_median, len(data_cols))):
        x0, x1 = x_edges[1 + j0], x_edges[1 + j1]
        cell(x0, 0.0, x1 - x0, 1.0, "white")
        ax.text((x0 + x1) / 2.0, 0.5, label, ha="center", va="center",
                fontsize=fontsize + 1, fontweight="bold", color=_HEATMAP_LABEL_COLOR)

    for j, column in enumerate(data_cols):
        tier = column[0]
        x0 = x_edges[1 + j]
        cell(x0, 1.0, data_col_width, 1.0, "white")
        label = "Ideal" if tier == "ideal" else _HEATMAP_TIER_LABELS[tier]
        ax.text(x0 + data_col_width / 2.0, 1.5, label, ha="center", va="center",
                fontsize=fontsize, color=_HEATMAP_LABEL_COLOR)

    row = 2
    for group in groups:
        cell(0.0, row, total_width, 1.0, "white")
        ax.text(0.08, row + 0.5, group["label"], ha="left", va="center",
                fontsize=fontsize + 1, fontweight="bold", color=_HEATMAP_LABEL_COLOR)
        row += 1
        for method in group["methods"]:
            cell(0.0, row, method_col_width, 1.0, "white")
            ax.text(method_col_width / 2.0, row + 0.5, method, ha="center",
                    va="center", fontsize=fontsize, color=_HEATMAP_LABEL_COLOR)
            for j, column in enumerate(data_cols):
                text, facecolor, on_color, bold = group["cells"][method][column]
                x0 = x_edges[1 + j]
                cell(x0, row, data_col_width, 1.0, facecolor)
                ax.text(x0 + data_col_width / 2.0, row + 0.5, text, ha="center",
                        va="center", fontsize=fontsize,
                        color=_HEATMAP_TEXT_COLOR if on_color else _HEATMAP_LABEL_COLOR,
                        fontweight="bold" if bold else "normal")
            row += 1

    # Divider between the Median and IQR blocks.
    sep_x = x_edges[1 + n_median]
    ax.plot([sep_x, sep_x], [0.0, n_rows], color=_HEATMAP_EDGE_COLOR, linewidth=1.4,
            zorder=5)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.075)

    # Two relative legend colour bars: scaling is per metric/statistic, so the
    # bars only convey the "darker = larger" direction, not absolute values.
    cax_median = fig.add_axes([0.12, 0.025, 0.34, 0.016])
    cax_iqr = fig.add_axes([0.54, 0.025, 0.34, 0.016])
    cb_median = fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=median_cmap_obj),
        cax=cax_median, orientation="horizontal", ticks=[0.0, 1.0],
    )
    cb_median.ax.set_xticklabels(["smaller", "larger"], fontsize=fontsize)
    cb_median.set_label("Median", fontsize=fontsize)
    cb_iqr = fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=iqr_cmap_obj),
        cax=cax_iqr, orientation="horizontal", ticks=[0.0, 1.0],
    )
    cb_iqr.ax.set_xticklabels(["smaller", "larger"], fontsize=fontsize)
    cb_iqr.set_label("IQR", fontsize=fontsize)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        save_figure(fig, save_path, png=png)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_error_metrics(df: pd.DataFrame, save_path=None, title=None, show=False):
    """Grouped bar chart of the error metrics, one row per metric family.

    The top row is the end-position error :math:`\\mathcal{EPE}` and the bottom
    row the thrust error integral :math:`\\mathcal{EI}`.  Within each row every
    method gets two bars: the 3D variant (solid) and the 2D planar variant
    (hatched).
    """
    # (panel y-label, [(column, legend label, hatch), ...]) per row.
    panels = [
        (
            r"$\mathcal{EPE}$ (km)",
            [
                ("epe", r"$\mathcal{EPE}_{\mathrm{3D}}$", ""),
                ("epe_planar", r"$\mathcal{EPE}_{\mathrm{2D}}$", "//"),
            ],
        ),
        (
            r"$\mathcal{EI}$ (m/s)",
            [
                ("ei_total", r"$\mathcal{EI}_{\mathrm{3D}}$", ""),
                ("ei_planar", r"$\mathcal{EI}_{\mathrm{2D}}$", "//"),
            ],
        ),
    ]
    methods = list(dict.fromkeys(df["method"].astype(str)))
    n = len(methods)
    x = np.arange(n)
    width = 0.38
    # Same main tone as the RTN / thrust curves for each method.
    colors = [METHOD_COLORS.get(m, _COLORS[i % len(_COLORS)]) for i, m in enumerate(methods)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, (ylabel, variants) in zip(axes, panels):
        for j, (col, _label, hatch) in enumerate(variants):
            heights = [float(df.loc[df["method"].astype(str) == m, col].iloc[0]) for m in methods]
            offset = (j - 0.5) * width
            ax.bar(
                x + offset, heights, width,
                color=colors, edgecolor="k", alpha=0.7, hatch=hatch,
            )
            for xi, h in zip(x + offset, heights):
                if h > 0:
                    ax.text(xi, h, f"{h:.2f}", ha="center", va="bottom", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE + 4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.tick_params(axis="x", labelsize=FONT_SIZE)
        ax.tick_params(axis="y", labelsize=FONT_SIZE + 4)
        ax.grid(True, axis="y", alpha=0.3)
        # Leave headroom so the value label on the tallest bar stays inside.
        top = max(max(float(df.loc[df["method"].astype(str) == m, col].iloc[0])
                      for m in methods) for col, _label, _hatch in variants)
        ax.set_ylim(0.0, top * 1.12 if top > 0 else 1.0)

    proxies = [
        Patch(facecolor="0.8", edgecolor="k", hatch="", label="3D"),
        Patch(facecolor="0.8", edgecolor="k", hatch="//", label="2D"),
    ]
    fig.legend(
        handles=proxies,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        borderaxespad=0.0,
        fontsize=LEGEND_FONT_SIZE + 4,
    )

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path

def _plot_one_selection(ax, path: str) -> int:
    """Draw the SSR path and selected point on ``ax``; return retained terms."""
    import json

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    selected = data["selected_max_iter"]
    path_entries = data["path"]
    iters = [p["max_iter"] for p in path_entries]
    rss = np.array([p["rss"] for p in path_entries], dtype=float)
    n_active = np.array([p["n_active"] for p in path_entries], dtype=float)
    sel_idx = iters.index(selected)

    ax.plot(n_active, rss, "o-", color="steelblue", markersize=3, linewidth=1, zorder=2)
    order = np.argsort(n_active)
    envelope = np.minimum.accumulate(rss[order])
    ax.step(n_active[order], envelope, where="post", color="darkorange",
            linewidth=1.4, zorder=3)
    ax.plot(n_active[sel_idx], rss[sel_idx], "ro", markersize=9, zorder=4)
    ax.set_xlabel(r"$n_a$", fontsize=FONT_SIZE + 1)
    ax.tick_params(axis="both", labelsize=FONT_SIZE - 1)
    ax.grid(True, alpha=0.3)
    return int(n_active[sel_idx])


def plot_selection_path(path, save_path=None, title=None, show=False):
    """Plot the SSR deletion path (RSS vs retained coefficients) with the selected point.

    ``path`` can be:
    - a single ``selection.json`` file path (legacy);
    - a list of ``(path, label)`` tuples to render multiple methods in one figure.

    When multiple paths are given they are arranged in a single figure with one
    subplot per method, a shared legend outside the axes (``RSS(n_a)`` /
    ``\\overline{RSS}(n_a)``), and per-subplot titles showing the method name
    and the number of retained terms.
    """
    if isinstance(path, (str, os.PathLike)):
        paths = [(path, title)]
    else:
        paths = list(path)

    if not paths:
        return save_path

    n = len(paths)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (sel_path, label) in zip(axes, paths):
        n_retained = _plot_one_selection(ax, sel_path)
        title_text = f"{label}, {n_retained} terms" if label else f"{n_retained} terms"
        ax.set_title(title_text, fontsize=FONT_SIZE + 1)

    for ax in axes[n:]:
        ax.axis("off")

    line_proxy = plt.Line2D([0], [0], color="steelblue", marker="o", markersize=3,
                            linewidth=1, linestyle="-")
    env_proxy = plt.Line2D([0], [0], color="darkorange", linewidth=1.4)
    fig.legend([line_proxy, env_proxy],
               [r"$\mathrm{RSS}(n_a)$",
                r"$\overline{\mathrm{RSS}}(n_a)$"],
               loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=LEGEND_FONT_SIZE + 1)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path
