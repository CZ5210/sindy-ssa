"""Ballistic-coefficient (Cd) uncertainty study. / 弹道系数不确定性研究。

Drag-coefficient mismatch between the truth satellite and the reconstruction
model biases the estimated thrust by the difference in drag acceleration.  This
study quantifies that bias for both scenarios.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..config import AppConfig, load_scenarios
from ..data import DataTable
from ..dynamics.perturbations import drag_acceleration_rtn
from ..registry import DEFAULT_CONFIG

_MEE_COLS = ["p", "f", "g", "h", "k", "L"]
_R_COLS = ["R_x", "R_y", "R_z"]
_V_COLS = ["V_x", "V_y", "V_z"]
_U_COLS = ["u_r", "u_t", "u_n"]


def _load_truth(path: str):
    table = DataTable.from_csv(path)
    time = np.asarray(table.get("Time").data, dtype=float)
    dt = float(time[1] - time[0]) if len(time) > 1 else 1.0
    mee = np.column_stack([table.get(c).data for c in _MEE_COLS]).astype(float)
    R = np.column_stack([table.get(c).data for c in _R_COLS]).astype(float)
    V = np.column_stack([table.get(c).data for c in _V_COLS]).astype(float)
    mass = np.asarray(table.get("Mass").data, dtype=float)
    u_true = np.column_stack([table.get(c).data for c in _U_COLS]).astype(float)
    return time, dt, mee, R, V, mass, u_true


def estimate_thrust_under_cd_error(
    data_real_path: str,
    cd_nominal: float,
    area_nominal: float,
    cd_factors: Sequence[float],
):
    """For each Cd factor compute the biased thrust estimate and error integral."""
    time, dt, mee, R, V, mass, u_true = _load_truth(data_real_path)
    n = len(time)

    drag_true = np.empty((n, 3), dtype=float)
    for i in range(n):
        drag_true[i] = drag_acceleration_rtn(mee[i], R[i], V[i], cd_nominal, area_nominal, mass[i])

    results = []
    for factor in cd_factors:
        drag_wrong = np.empty((n, 3), dtype=float)
        for i in range(n):
            drag_wrong[i] = drag_acceleration_rtn(
                mee[i], R[i], V[i], cd_nominal * factor, area_nominal, mass[i]
            )

        u_est = u_true + drag_true - drag_wrong
        error = u_est - u_true
        ei_components = dt * np.sum(np.abs(error), axis=0) * 1000.0  # km/s -> m/s
        ei_total = float(np.sum(ei_components))

        results.append({
            "cd_factor": float(factor),
            "cd_used": cd_nominal * factor,
            "u_est": u_est,
            "error": error,
            "ei_r": float(ei_components[0]),
            "ei_t": float(ei_components[1]),
            "ei_n": float(ei_components[2]),
            "ei_total": ei_total,
        })

    return time, u_true, results


def run(config: AppConfig, cd_factors: Sequence[float] = (0.6, 0.8, 1.0, 1.2, 1.4)) -> pd.DataFrame:
    """Run the Cd-uncertainty study and print a summary to the console.

    No figure or data file is written.
    """
    scenario = config.scenario.name
    real_path = os.path.join(config.paths.data_dir, "Data_real.csv")
    cd_nominal = config.satellite.drag_coefficient
    area_nominal = config.satellite.cross_section_area

    _time, _u_true, results = estimate_thrust_under_cd_error(
        real_path, cd_nominal, area_nominal, cd_factors
    )

    summary = pd.DataFrame([
        {
            "scenario": scenario,
            "cd_factor": res["cd_factor"],
            "cd_used": res["cd_used"],
            "ei_r": res["ei_r"],
            "ei_t": res["ei_t"],
            "ei_n": res["ei_n"],
            "ei_total": res["ei_total"],
        }
        for res in results
    ])

    print(f"=== Cd uncertainty study: {scenario} ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print()
    return summary


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cd uncertainty study (console output only)")
    parser.add_argument("--config", default=None, help="single-scenario YAML")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--factors", nargs="+", type=float, default=[0.6, 0.8, 1.0, 1.2, 1.4])
    args = parser.parse_args(argv)

    scenario_cfgs = load_scenarios(args.config or DEFAULT_CONFIG)
    for cfg in scenario_cfgs.values():
        cfg.paths.data_root = args.data_root
        cfg.paths.data_dir = os.path.join(args.data_root, cfg.scenario.name, "truth")
        run(cfg, cd_factors=args.factors)
    return 0
