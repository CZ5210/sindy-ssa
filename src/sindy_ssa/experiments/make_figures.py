"""Generate single-run result figures from existing outputs.

从已有输出生成单次实验的结果图。

Produces: six-method thrust comparison (mm/s^2), SINDy coefficient heatmaps
and the RT-plane re-propagation comparison. 轨迹真值图（ECI/LVLH）不在此处，
统一由动力学绘图入口生成。
"""

from __future__ import annotations

import os

import pandas as pd

from ..config import AppConfig
from ..plotting import (
    plot_error_metrics,
    plot_repropagated_rt,
    plot_selection_path,
    plot_thrust_comparison,
    plot_xi_heatmap,
    scenario_rtn_limits,
)
from ..plotting.style import ACC_LABELS, ELEMENT_LABELS
from ..registry import METHODS
from ..utils import ensure_dir
from .evaluate import evaluate_existing

_XI_DIRS = {"SINDy": "SINDy", "SINDy-TIME": "SINDy-TIME", "SINDy-UMEE": "SINDy-UMEE", "TIME-OLS": "TIME-OLS"}
# Methods without a forward-prediction model (excluded from partial-mode reprop).
_NON_PREDICTIVE = {"MEEFD-SG", "MEEFD-Spline"}
# Fixed acceleration window (mm/s^2) for every R/T/N panel of the thrust figure.
_THRUST_YLIM = (-0.23, 0.23)
# Scenario-specific x-tick spacing (interval_h, start_h); evasive keeps automatic ticks.
_THRUST_XTICKS = {"stationkeeping": (2.0, 16.0)}


def _fit_horizon(config: AppConfig, real_path: str):
    """Fit-horizon time (s) for the partial-window figure, or ``None``."""
    pred = config.sindy.prediction
    if not pred.enabled or not (0.0 < float(pred.fit_fraction) < 1.0):
        return None
    df = pd.read_csv(real_path, usecols=["Time", "Flag"])
    man = df[df["Flag"] == "manoeuvre"]
    if man.empty:
        return None
    t0, t1 = float(man["Time"].iloc[0]), float(man["Time"].iloc[-1])
    return t0 + float(pred.fit_fraction) * (t1 - t0)


def make_result_figures(config: AppConfig) -> list[str]:
    """Render the standard result figures and return the saved paths."""
    data_dir = config.paths.data_dir
    methods_dir = config.paths.methods_dir
    figures_dir = config.paths.figures_dir
    ensure_dir(figures_dir)

    real_path = os.path.join(data_dir, "Data_real.csv")
    saved: list[str] = []

    def _save(path):
        saved.append(path)
        return path

    # 1) thrust comparison (true + four methods), mm/s^2
    thrust_paths = [real_path]
    labels = ["True"]
    for name, filename in METHODS.items():
        candidate = os.path.join(methods_dir, name, filename)
        if os.path.exists(candidate):
            thrust_paths.append(candidate)
            labels.append(name)
    xticks = _THRUST_XTICKS.get(config.scenario.name)
    plot_thrust_comparison(
        thrust_paths, labels,
        vline=_fit_horizon(config, real_path),
        vline_label="fit horizon",
        ylim=_THRUST_YLIM,
        x_tick_interval=xticks[0] if xticks else None,
        x_tick_start=xticks[1] if xticks else None,
        save_path=_save(os.path.join(figures_dir, "thrust_comparison.png")),
    )

    # 2) coefficient heatmaps
    for name, folder in _XI_DIRS.items():
        xi_path = os.path.join(methods_dir, folder, "Xi.csv")
        if os.path.exists(xi_path):
            if name == "SINDy":
                x_labels = ELEMENT_LABELS
            else:
                x_labels = ACC_LABELS
            plot_xi_heatmap(
                xi_path, save_path=_save(os.path.join(figures_dir, f"xi_{name}.png")), x_labels=x_labels
            )

    # 3) RT-plane re-propagation comparison (skip non-predictive in partial mode)
    pred = config.sindy.prediction
    partial = pred.enabled and 0.0 < float(pred.fit_fraction) < 1.0
    reprop = []
    reprop_labels = []
    for name in METHODS:
        if partial and name in _NON_PREDICTIVE:
            continue
        path = os.path.join(methods_dir, name, "Data_reLVLH.csv")
        if not os.path.exists(path):
            continue
        try:
            rel = os.path.relpath(path, data_dir)
        except ValueError:  # different drives on Windows
            rel = path
        reprop.append(rel)
        reprop_labels.append(name)
    if reprop:
        plot_repropagated_rt(
            data_dir, reprop, reprop_labels,
            limits=scenario_rtn_limits(config.scenario.name),
            save_path=_save(os.path.join(figures_dir, "repropagated_rt.png")),
        )

    # 4) error metrics bar chart (single-run evaluation)
    eval_path = config.paths.evaluation_file
    if not os.path.exists(eval_path):
        evaluate_existing(config)
    if os.path.exists(eval_path):
        eval_df = pd.read_csv(eval_path)
        plot_error_metrics(
            eval_df,
            save_path=_save(os.path.join(figures_dir, "error_metrics.png")),
        )

    # 5) L-curve / selection path for all SINDy methods in one figure
    selection_paths = []
    for name, folder in _XI_DIRS.items():
        sel_path = os.path.join(methods_dir, folder, "selection.json")
        if os.path.exists(sel_path):
            selection_paths.append((sel_path, name))
    if selection_paths:
        plot_selection_path(
            selection_paths,
            save_path=_save(os.path.join(figures_dir, "selection_path.png")),
        )

    # 6) identified sparse expressions for this single typical run
    from ..studies.expressions import export_run

    saved.extend(export_run(methods_dir, figures_dir, names=list(_XI_DIRS.values()), partial=partial))

    return saved
