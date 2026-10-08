"""Performance metrics: end-position error (EPE) and error integral (EI).

评价指标：终端位置误差（EPE）与误差积分（EI）。
"""

from __future__ import annotations

import numpy as np

from ..data import DataTable


def error_integral(true_table: DataTable, rec_table: DataTable):
    """Component-wise absolute thrust error integral [m/s].

    分方向推力绝对误差积分。
    """
    time = true_table.get("Time").data
    dt = float(time[1] - time[0])

    components = {}
    for comp in ("u_r", "u_t", "u_n"):
        diff = np.abs(true_table.get(comp).data - rec_table.get(comp).data)
        components[comp] = dt * float(np.sum(diff)) * 1000.0
    total = components["u_r"] + components["u_t"] + components["u_n"]
    return components["u_r"], components["u_t"], components["u_n"], total


def _align_to_reference(true_table: DataTable, rec_table: DataTable, window=None) -> DataTable:
    """Slice the true table onto the reconstructed time window."""
    start = float(rec_table.get("Time").data[0])
    end = float(rec_table.get("Time").data[-1])
    times = np.asarray(true_table.get("Time").data, dtype=float)
    start_idx = int(np.argmin(np.abs(times - start)))
    end_idx = int(np.argmin(np.abs(times - end))) + 1
    return true_table.filter((start_idx, end_idx))


def compute_thrust_metrics(true_table: DataTable, rec_table: DataTable) -> dict:
    true_aligned = _align_to_reference(true_table, rec_table)
    r, t, n, total = error_integral(true_aligned, rec_table)
    return {"ei_r": r, "ei_t": t, "ei_n": n, "ei_total": total, "ei_planar": r + t}


def end_position_error(true_lvlh_path: str, rec_lvlh_path: str, planar: bool = False):
    """Terminal position error [km] and the true terminal position.

    ``planar=False`` gives the full 3D RTN error (primary metric);
    ``planar=True`` restricts to the RT plane (secondary metric under the
    coplanar-maneuver prior).
    """
    axes = ["R_x", "R_y"] if planar else ["R_x", "R_y", "R_z"]
    true_tbl = DataTable.from_csv(true_lvlh_path, columns=["Time", *axes, "Flag"])
    flag = np.asarray(true_tbl.get("Flag").data)
    true_pos = np.array([
        np.asarray(true_tbl.get(a).data, dtype=float)[flag == "manoeuvre"][-1] for a in axes
    ])

    rec = DataTable.from_csv(rec_lvlh_path, columns=axes)
    rec_pos = np.array([float(rec.get(a).data[-1]) for a in axes])
    return float(np.linalg.norm(rec_pos - true_pos)), true_pos
