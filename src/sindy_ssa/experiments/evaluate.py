"""Evaluate existing reconstruction outputs. / 评价已有的重建结果。"""

from __future__ import annotations

import os

import pandas as pd

from ..config import AppConfig
from ..data import DataTable
from ..metrics import compute_thrust_metrics, end_position_error
from ..registry import METHODS
from ..utils import ensure_dir


def evaluate_existing(config: AppConfig) -> pd.DataFrame:
    """Read the saved method outputs and compute EPE / EI without re-running."""
    data_dir = config.paths.data_dir
    methods_dir = config.paths.methods_dir
    truth_thrust = DataTable.from_csv(os.path.join(data_dir, "Data_real.csv"), columns=["Time", "u_r", "u_t", "u_n"])
    truth_lvlh = os.path.join(data_dir, "Data_LVLH.csv")

    rows = []
    for name, filename in METHODS.items():
        thrust_path = os.path.join(methods_dir, name, filename)
        if not os.path.exists(thrust_path):
            continue
        reprop_path = os.path.join(methods_dir, name, "Data_reLVLH.csv")
        if not os.path.exists(reprop_path):
            # Non-predictive method in partial-window mode: no full-arc re-propagation.
            continue
        rec = DataTable.from_csv(thrust_path)
        metrics = compute_thrust_metrics(truth_thrust, rec)
        epe_3d_path = os.path.join(methods_dir, name, "Data_reLVLH_3D.csv")
        if os.path.exists(epe_3d_path):
            epe_3d, _ = end_position_error(truth_lvlh, epe_3d_path, planar=False)
            metrics["epe"] = epe_3d
            epe_planar, _ = end_position_error(truth_lvlh, reprop_path, planar=True)
            metrics["epe_planar"] = epe_planar
        else:
            epe, _ = end_position_error(truth_lvlh, reprop_path, planar=True)
            metrics["epe"] = epe
        metrics["method"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(config.paths.evaluation_file))
    df.to_csv(config.paths.evaluation_file, index=False)
    return df
