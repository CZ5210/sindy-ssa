"""Truth-dynamics plotting utilities. / 真值动力学绘图。

These functions are used by ``sindy-ssa dynamics plot``.
"""

from __future__ import annotations

import os

from ..config import AppConfig
from ..plotting import (
    plot_lvlh_evasive,
    plot_lvlh_stationkeeping,
    plot_trajectory_eci,
    plot_true_thrust,
)
from ..utils import ensure_dir


def plot_scenario_truth(config: AppConfig) -> list[str]:
    """Plot ECI/LVLH trajectories and true thrust for the current scenario.

    Saves PDF (+ PNG) under ``config.paths.figures_dir`` using filenames without
    extension so that :func:`sindy_ssa.plotting.style.save_figure` can attach the
    appropriate suffixes.
    """
    figures_dir = config.paths.figures_dir
    ensure_dir(figures_dir)

    prefix = f"{config.scenario.name}_"
    real_path = os.path.join(config.paths.data_dir, "Data_real.csv")
    lvlh_path = os.path.join(config.paths.data_dir, "Data_LVLH.csv")
    saved: list[str] = []

    saved.append(
        plot_trajectory_eci(
            real_path,
            save_path=os.path.join(figures_dir, f"{prefix}trajectory_eci"),
        )
    )
    if config.scenario.name == "stationkeeping":
        saved.append(
            plot_lvlh_stationkeeping(
                lvlh_path,
                box_size=config.scenario.box_size,
                shift=config.scenario.box_shift,
                save_path=os.path.join(figures_dir, f"{prefix}trajectory_lvlh"),
            )
        )
    else:
        saved.append(
            plot_lvlh_evasive(
                lvlh_path,
                target=config.scenario.target_position,
                save_path=os.path.join(figures_dir, f"{prefix}trajectory_lvlh"),
            )
        )
    saved.append(
        plot_true_thrust(
            real_path,
            save_path=os.path.join(figures_dir, f"{prefix}thrust_true"),
        )
    )
    return saved
