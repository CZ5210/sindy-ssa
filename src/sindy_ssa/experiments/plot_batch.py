"""Batch-study plotting utilities. / 蒙特卡洛批处理绘图。

These functions are used by ``sindy-ssa batch plot`` and read the long-format
``batch_evaluation.csv`` produced by ``sindy-ssa batch run``.
"""

from __future__ import annotations

import os

import pandas as pd

from ..config import AppConfig
from ..plotting import plot_metric_overview, plot_metric_tiers, plot_metric_violins
from ..utils import ensure_dir

IDEAL_CASE = "ideal"


def _load_case(config: AppConfig, case: str) -> pd.DataFrame | None:
    path = os.path.join(config.paths.data_root, config.scenario.name, "batch", case, "batch_evaluation.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def plot_batch_figures(config: AppConfig) -> list[str]:
    """Render the merged per-case batch violin figure from an existing evaluation file.

    A single two-row figure is produced, matching the typical-results error-bar
    style: the top row is the end-position error :math:`\\mathcal{EPE}` and the
    bottom row the error integral :math:`\\mathcal{EI}`, each with a solid 3D and
    a hatched 2D violin per method.  The evaluation CSV must contain a ``run``
    column with more than one unique value, otherwise an empty list is returned.
    """
    eval_path = config.paths.evaluation_file
    if not os.path.exists(eval_path):
        return []

    df = pd.read_csv(eval_path)
    if "run" not in df.columns or df["run"].nunique() <= 1:
        return []

    figures_dir = config.paths.figures_dir
    ensure_dir(figures_dir)
    scenario = config.scenario.name

    return [
        plot_metric_violins(
            df,
            save_path=os.path.join(figures_dir, "batch_metrics"),
            title=f"{scenario}: reconstruction error (batch)",
        )
    ]


def plot_batch_overview(config: AppConfig, cases) -> list[str]:
    """Render the cross-noise overview: EPE / EI violins for every noise tier.

    One figure per scenario combining all Monte-Carlo cases: rows are the two
    metrics, columns the noise tiers, and the deterministic ``ideal`` case is
    overlaid as a marker.  No figure is produced when fewer than two noise cases
    are available.
    """
    case_data: dict[str, pd.DataFrame] = {}
    for case in cases:
        df = _load_case(config, case)
        if df is not None and not df.empty:
            case_data[case] = df

    noise_cases = [c for c in cases if c != IDEAL_CASE and c in case_data]
    if len(noise_cases) < 2:
        return []

    figures_dir = os.path.join(config.paths.figures_root, config.scenario.name, "batch")
    ensure_dir(figures_dir)
    return [
        plot_metric_overview(
            case_data,
            noise_cases,
            ideal_case=IDEAL_CASE if IDEAL_CASE in case_data else None,
            save_path=os.path.join(figures_dir, "batch_overview"),
            title=f"{config.scenario.name}: error across noise levels (batch)",
            log_scale=config.batch.overview_log,
        )
    ]


def plot_batch_tiers(config: AppConfig, cases) -> list[str]:
    """Render the tier-separated 4xN overview (EPE/EI x 3D/2D).

    Same data as :func:`plot_batch_overview` but a 4xN grid: one row per metric
    variant and one column per noise tier (High / Low / Reduced), so each panel
    keeps its own y-axis.  Within a panel each method is drawn as a split glyph
    (violin density on the left, box+fliers on the right) and identified by its
    shared colour.  The y-axis is linear and only the PDF is written.  Returns an
    empty list when fewer than two noise cases are available.
    """
    case_data: dict[str, pd.DataFrame] = {}
    for case in cases:
        df = _load_case(config, case)
        if df is not None and not df.empty:
            case_data[case] = df

    noise_cases = [c for c in cases if c != IDEAL_CASE and c in case_data]
    if len(noise_cases) < 2:
        return []

    figures_dir = os.path.join(config.paths.figures_root, config.scenario.name, "batch")
    ensure_dir(figures_dir)
    return [
        plot_metric_tiers(
            case_data,
            noise_cases,
            save_path=os.path.join(figures_dir, "batch_tiers"),
            log_scale=False,
            png=False,
        )
    ]
