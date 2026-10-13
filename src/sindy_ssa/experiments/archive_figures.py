"""Paper figures rendered from an archived Monte-Carlo batch.

从归档的蒙特卡洛批量结果渲染论文图件。

The archived results live under::

    batch_archive/<window>/<scenario>/<tier>/batch_evaluation.csv

where ``<window>`` is one of ``full``/``window30``/``window70``,
``<scenario>`` is ``evasive``/``stationkeeping`` and ``<tier>`` is
``ideal``/``high``/``low``.  This module reads those long-format evaluation
tables directly (no config or CLI needed) and renders:

* the ``batch_tiers`` figure---a 2xN grid with one row per error metric (EPE
  on top, EI below) and one column per noise tier, each panel on its own
  y-axis.  Every method is drawn as a pair of split glyphs: the left half of a
  glyph is the KDE violin and the right half a boxplot with the outlying runs
  shown as fliers; the solid glyph is the full 3D metric and the hatched glyph
  its planar 2D variant;
* the ``batch_vs_window`` convergence chart---2xN grid (planar EPE/EI rows, noise
  tiers columns) with the observed fraction (30 / 70 / 100%) on the x-axis
  (the 100% point being the fully observed batch), one line per method (median
  over the MC runs) with its IQR shaded;
* the ``batch_heatmap`` summary tables---colour-coded replacements for the
  hand-written result tables, one for the fully observed campaign and one
  stacking the partial windows.

Both the ``plot_batch_tiers.py`` entry script and ``sindy-ssa batch paper``
call :func:`plot_archive_figures`, so the figure set has a single definition.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..plotting import plot_metric_tiers, plot_metric_vs_window, plot_summary_heatmap
from ..plotting.style import METHOD_ORDER
from ..registry import PAPER_WINDOWS, paper_window_prefix
from ..utils import ensure_dir

# window directory -> observed fraction, for the partial-window summary.
WINDOW_FRACTIONS = tuple(
    (window, round(fraction * 100))
    for window, fraction in PAPER_WINDOWS
    if fraction is not None
)
# convergence line chart additionally includes the fully observed (100%) case.
CONVERGENCE_WINDOWS = tuple(
    (window, 100 if fraction is None else round(fraction * 100))
    for window, fraction in PAPER_WINDOWS
)
SCENARIOS = ("evasive", "stationkeeping")
# Column blocks of the merged summary heatmaps: (scenario key, column label).
SCENARIO_BLOCKS = (("stationkeeping", "Station-keeping"), ("evasive", "Evasive"))
# Tier order shown in the figure (left to right within each method).
NOISE_CASES = ("high", "low")
# Deterministic noise-free reference case, drawn as an extra column.
IDEAL_CASE = "ideal"

# (evaluation-column, label) row groups for the full-observation heatmaps.
FULL_METRICS = (
    ("epe", r"$\mathcal{EPE}_{\mathrm{3D}}$ (km)"),
    ("epe_planar", r"$\mathcal{EPE}_{\mathrm{2D}}$ (km)"),
    ("ei_total", r"$\mathcal{EI}_{\mathrm{3D}}$ (m/s)"),
    ("ei_planar", r"$\mathcal{EI}_{\mathrm{2D}}$ (m/s)"),
)
# Partial-window heatmaps only use the planar 2D metrics.
PARTIAL_METRICS = (
    ("epe_planar", r"$\mathcal{EPE}_{\mathrm{2D}}$ (km)"),
    ("ei_planar", r"$\mathcal{EI}_{\mathrm{2D}}$ (m/s)"),
)


def _summary_group(label: str, scenario_data: dict, methods, column: str) -> dict:
    """Assemble one heatmap row group (a metric block) from the tier frames.

    ``scenario_data`` maps every column-block key (see ``SCENARIO_BLOCKS``) to
    that scenario's ``(tier frames, ideal frame)`` pair.  Each method gets the
    median and IQR over its Monte-Carlo runs per tier and the single
    deterministic ideal median (no IQR).
    """
    data = {}
    for block, (tier_data, ideal_df) in scenario_data.items():
        entry = {}
        for method in methods:
            method_entry = {}
            for tier, df in tier_data.items():
                values = pd.to_numeric(
                    df.loc[df["method"].astype(str) == method, column], errors="coerce"
                ).dropna().values
                if values.size == 0:
                    continue
                method_entry[tier] = {
                    "median": float(np.median(values)),
                    "iqr": float(np.percentile(values, 75) - np.percentile(values, 25))
                    if values.size > 1 else None,
                }
            if ideal_df is not None:
                ideal_values = pd.to_numeric(
                    ideal_df.loc[ideal_df["method"].astype(str) == method, column],
                    errors="coerce",
                ).dropna().values
                if ideal_values.size:
                    method_entry[IDEAL_CASE] = {"median": float(ideal_values[0])}
            if method_entry:
                entry[method] = method_entry
        if entry:
            data[block] = entry
    return {"label": label, "methods": list(methods), "data": data}


def _write_heatmap(row_groups, save_path: str, png: bool, saved: list[str]) -> bool:
    """Render one summary heatmap if any row group carries data."""
    row_groups = [g for g in row_groups if g["data"]]
    if not row_groups:
        return False
    plot_summary_heatmap(row_groups, save_path=save_path, png=png,
                         blocks=SCENARIO_BLOCKS)
    saved.append(save_path + ".pdf")
    return True


def _load_case(archive: str, window: str, scenario: str, tier: str) -> pd.DataFrame | None:
    """Return the ``batch_evaluation.csv`` for one case, or ``None`` if absent."""
    path = os.path.join(archive, window, scenario, tier, "batch_evaluation.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


def _scenario_frames(archive: str, window: str, scenario: str) -> tuple[dict, pd.DataFrame | None]:
    """Tier frames plus the deterministic ideal frame of one (window, scenario)."""
    tiers = {}
    for tier in NOISE_CASES:
        df = _load_case(archive, window, scenario, tier)
        if df is not None:
            tiers[tier] = df
    return tiers, _load_case(archive, window, scenario, IDEAL_CASE)


def _present_methods(frames) -> list[str]:
    """Methods found in ``frames``, in the canonical method order."""
    seen = {
        method for df in frames if df is not None
        for method in df["method"].astype(str)
    }
    ordered = [method for method in METHOD_ORDER if method in seen]
    return ordered + sorted(seen.difference(ordered))


def plot_archive_figures(archive: str, out_dir: str | None = None, *,
                         log_scale: bool = False, png: bool = False) -> list[str]:
    """Render every archive figure and return the written paths.

    ``archive`` is the root of the archived results (see the module docstring)
    and ``out_dir`` the figure directory, defaulting to ``<archive>/figs``.
    Figures are written as PDF; ``png`` adds a raster preview next to each PDF.
    Windows/scenarios whose evaluation tables are missing are skipped.
    """
    archive = os.path.abspath(archive)
    out_dir = os.path.abspath(out_dir) if out_dir else os.path.join(archive, "figs")
    ensure_dir(out_dir)
    saved: list[str] = []

    for window, _ in PAPER_WINDOWS:
        for scenario in SCENARIOS:
            case_data = {}
            for tier in NOISE_CASES:
                df = _load_case(archive, window, scenario, tier)
                if df is not None:
                    case_data[tier] = df

            noise_cases = [c for c in NOISE_CASES if c in case_data]
            if len(noise_cases) < 2:
                print(f"skip {window}/{scenario}: only {len(noise_cases)} tier(s) available")
                continue

            save_path = os.path.join(
                out_dir, f"batch_tiers_{paper_window_prefix(window)}{scenario}"
            )
            plot_metric_tiers(
                case_data, noise_cases, save_path=save_path,
                log_scale=log_scale, png=png,
            )
            saved.append(save_path + ".pdf")

    # Convergence with the observed fraction: one line chart per scenario, rows
    # the planar 2D metrics and columns the noise tiers, each line a method's
    # median over the MC runs with its IQR shaded.  The fully observed (100%)
    # case is appended so the fully observed batch serves as the convergence
    # limit; the non-predictive MEEFD baselines only exist there, so they are
    # drawn as an interquartile box instead of a line.
    for scenario in SCENARIOS:
        window_data, ideal_data = {}, {}
        for window, fraction in CONVERGENCE_WINDOWS:
            tiers = {}
            for tier in NOISE_CASES:
                df = _load_case(archive, window, scenario, tier)
                if df is not None:
                    tiers[tier] = df
            if tiers:
                window_data[fraction] = tiers
            ideal = _load_case(archive, window, scenario, IDEAL_CASE)
            if ideal is not None:
                ideal_data[fraction] = ideal
        windows = [f for _, f in CONVERGENCE_WINDOWS if f in window_data]
        if not windows:
            continue
        methods = _present_methods(
            [df for tiers in window_data.values() for df in tiers.values()]
            + list(ideal_data.values())
        )
        save_path = os.path.join(out_dir, f"batch_vs_window_{scenario}")
        plot_metric_vs_window(
            window_data, windows, list(NOISE_CASES), ideal_data=ideal_data,
            methods=methods, save_path=save_path, png=png, log_scale=log_scale,
        )
        saved.append(save_path + ".pdf")

    # Coloured "Median + IQR" summary heatmaps, two in total: a full-observation
    # table (four metric row groups) and a partial-window table (observed
    # fraction x planar 2D metric), each carrying both scenarios as side-by-side
    # column blocks.
    full_blocks = {key: _scenario_frames(archive, "full", key) for key, _ in SCENARIO_BLOCKS}
    full_blocks = {key: frames for key, frames in full_blocks.items() if frames[0]}
    if full_blocks:
        methods = _present_methods(
            [df for tiers, ideal in full_blocks.values()
             for df in (*tiers.values(), ideal)]
        )
        groups = [
            _summary_group(label, full_blocks, methods, column)
            for column, label in FULL_METRICS
        ]
        _write_heatmap(groups, os.path.join(out_dir, "batch_heatmap"), png, saved)

    groups = []
    for window, fraction in WINDOW_FRACTIONS:
        blocks = {key: _scenario_frames(archive, window, key) for key, _ in SCENARIO_BLOCKS}
        blocks = {key: frames for key, frames in blocks.items() if frames[0]}
        if not blocks:
            continue
        methods = _present_methods(
            [df for tiers, ideal in blocks.values() for df in (*tiers.values(), ideal)]
        )
        for column, label in PARTIAL_METRICS:
            groups.append(
                _summary_group(f"{fraction}% observed: {label}", blocks,
                               methods, column)
            )
    _write_heatmap(groups, os.path.join(out_dir, "batch_heatmap_windows"), png, saved)

    return saved
