"""Standalone regeneration of the batch-tier figures from an archived batch run.

The Monte-Carlo results live under::

    batch_archive/<window>/<scenario>/<tier>/batch_evaluation.csv

where ``<window>`` is one of ``full``/``window25``/``window50``/``window75``,
``<scenario>`` is ``evasive``/``stationkeeping`` and ``<tier>`` is
``high``/``low``/``reduced``.  This script reads those long-format evaluation
tables directly (no config or CLI needed) and renders:

* the ``batch_tiers`` figure---a 4xN grid with one row per EPE/EI variant (3D
  and planar 2D) and one column per noise tier (High / Low / Reduced), each
  panel on its own linear y-axis.  Every method is drawn as a split glyph: the
  left half is the KDE violin and the right half a boxplot with the outlying
  runs shown as fliers;
* the ``batch_vs_window`` convergence chart---2xN grid (planar EPE/EI rows, noise
  tiers columns) with the observed fraction (25 / 50 / 75 / 100%) on the x-axis
  (the 100% point being the fully observed batch), one line per method (median
  over the MC runs) with its IQR shaded;
* ``batch_heatmap`` summary tables---colour-coded replacements for the hand-written
  result tables.  Each noise tier contributes a median and an IQR column (the
  deterministic ideal case adds a median-only column); median cells use a warm
  colour map and IQR cells a cool one, each scaled per metric row group.  The
  full-observation table stacks the four metrics as row groups, while the
  partial-window table stacks the observed fraction (25 / 50 / 75%) times the
  planar 2D metric.

Figures are written as PDF only; ``--png`` adds a raster preview.

Output names follow the convention already used in the paper:

* ``batch_tiers_<scenario>.pdf``         (full manoeuvre window)
* ``batch_tiers_<25|50|75><scenario>.pdf`` (partial-window prediction)
* ``batch_vs_window_<scenario>.pdf``       (median/IQR convergence line chart)
* ``batch_heatmap_<scenario>.pdf``         (full-observation median/IQR table)
* ``batch_heatmap_windows_<scenario>.pdf`` (partial-window median/IQR table)

Usage::

    python plot_batch_tiers.py
    python plot_batch_tiers.py --png
    python plot_batch_tiers.py --log
    python plot_batch_tiers.py --archive path/to/batch_archive --out path/to/figs
"""

from __future__ import annotations

import argparse
import os
import sys

# Make ``sindy_ssa`` importable when run as a script without pip-installing it.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd

from sindy_ssa.plotting import plot_metric_tiers, plot_metric_vs_window, plot_summary_heatmap

# window directory -> filename prefix (mirrors the paper figure names).
WINDOWS = (("full", ""), ("window25", "25"), ("window50", "50"), ("window75", "75"))
# partial-window directories -> observed fraction, for the partial-window summary.
WINDOW_FRACTIONS = (("window25", 25), ("window50", 50), ("window75", 75))
# convergence line chart additionally includes the fully observed (100%) case.
CONVERGENCE_WINDOWS = (("window25", 25), ("window50", 50), ("window75", 75), ("full", 100))
SCENARIOS = ("evasive", "stationkeeping")
# Tier order shown in the figure (left to right within each method).
NOISE_CASES = ("high", "low", "reduced")
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


def _summary_group(label: str, tier_data: dict, ideal_df, methods, column: str) -> dict:
    """Assemble one heatmap row group (a metric block) from the tier frames.

    Each method gets the median and IQR over its Monte-Carlo runs per tier and
    the single deterministic ideal median (no IQR).
    """
    data = {}
    for method in methods:
        entry = {}
        for tier, df in tier_data.items():
            values = pd.to_numeric(
                df.loc[df["method"].astype(str) == method, column], errors="coerce"
            ).dropna().values
            if values.size == 0:
                continue
            entry[tier] = {
                "median": float(np.median(values)),
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25))
                if values.size > 1 else None,
            }
        if ideal_df is not None:
            ideal_values = pd.to_numeric(
                ideal_df.loc[ideal_df["method"].astype(str) == method, column], errors="coerce"
            ).dropna().values
            if ideal_values.size:
                entry[IDEAL_CASE] = {"median": float(ideal_values[0])}
        if entry:
            data[method] = entry
    return {"label": label, "methods": list(methods), "data": data}


def _write_heatmap(row_groups, save_path: str, png: bool) -> bool:
    """Render one summary heatmap if any row group carries data."""
    row_groups = [g for g in row_groups if g["data"]]
    if not row_groups:
        return False
    plot_summary_heatmap(row_groups, save_path=save_path, png=png)
    print(f"wrote {save_path}.pdf")
    return True


def _load_case(archive: str, window: str, scenario: str, tier: str) -> pd.DataFrame | None:
    """Return the ``batch_evaluation.csv`` for one case, or ``None`` if absent."""
    path = os.path.join(archive, window, scenario, tier, "batch_evaluation.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--archive",
        default=os.path.join(_HERE, "batch_archive"),
        help="root of the batch archive (default: ./batch_archive)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output directory for the figures (default: <archive>/figs)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="use a logarithmic y-axis on the tier and convergence charts (default: linear)",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="also write a PNG preview next to each PDF (default: PDF only)",
    )
    args = parser.parse_args(argv)

    archive = os.path.abspath(args.archive)
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(archive, "figs")
    os.makedirs(out_dir, exist_ok=True)

    for window, prefix in WINDOWS:
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

            save_path = os.path.join(out_dir, f"batch_tiers_{prefix}{scenario}")
            plot_metric_tiers(
                case_data, noise_cases, save_path=save_path,
                log_scale=args.log, png=args.png,
            )
            print(f"wrote {save_path}.pdf")

    # Convergence with the observed fraction: one line chart per scenario, rows
    # the planar 2D metrics and columns the noise tiers, each line a method's
    # median over the MC runs with its IQR shaded.  The fully observed (100%)
    # case is appended so the fully observed batch serves as the convergence
    # limit.
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
        save_path = os.path.join(out_dir, f"batch_vs_window_{scenario}")
        plot_metric_vs_window(
            window_data, windows, list(NOISE_CASES), ideal_data=ideal_data,
            save_path=save_path, png=args.png, log_scale=args.log,
        )
        print(f"wrote {save_path}.pdf")

    # Coloured "Median + IQR" summary heatmaps, one pair per scenario: a
    # full-observation table (four metric row groups) and a merged partial-window
    # table (observed fraction x planar 2D metric).
    for scenario in SCENARIOS:
        full_tiers = {t: _load_case(archive, "full", scenario, t) for t in NOISE_CASES}
        full_tiers = {t: df for t, df in full_tiers.items() if df is not None}
        if full_tiers:
            ref = next(iter(full_tiers.values()))
            methods = list(dict.fromkeys(ref["method"].astype(str)))
            ideal_df = _load_case(archive, "full", scenario, IDEAL_CASE)
            groups = [
                _summary_group(label, full_tiers, ideal_df, methods, column)
                for column, label in FULL_METRICS
            ]
            save_path = os.path.join(out_dir, f"batch_heatmap_{scenario}")
            _write_heatmap(groups, save_path, args.png)

        groups = []
        for window, fraction in WINDOW_FRACTIONS:
            tier_data = {t: _load_case(archive, window, scenario, t) for t in NOISE_CASES}
            tier_data = {t: df for t, df in tier_data.items() if df is not None}
            if not tier_data:
                continue
            ref = next(iter(tier_data.values()))
            methods = list(dict.fromkeys(ref["method"].astype(str)))
            ideal_df = _load_case(archive, window, scenario, IDEAL_CASE)
            for column, label in PARTIAL_METRICS:
                groups.append(
                    _summary_group(f"{fraction}% observed: {label}", tier_data,
                                   ideal_df, methods, column)
                )
        save_path = os.path.join(out_dir, f"batch_heatmap_windows_{scenario}")
        _write_heatmap(groups, save_path, args.png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
