"""Shared preprocessing for the reconstruction methods.

各重建方法共用的预处理。
"""

from __future__ import annotations

import numpy as np

from ..data import DataSeries, DataTable
from ..dynamics.elements import MEE_Dynamics, MEEdot2Acc

_MEE = ["p", "f", "g", "h", "k", "L"]
_MEEDOT_NAMES = ["Time", "p_dot", "f_dot", "g_dot", "h_dot", "k_dot", "L_dot"]


def smoothing_window_samples(window_seconds: float, time) -> int:
    """把 SG 时间窗长 T_w 折算成样本窗长 N_w = 2·⌊T_w/(2τ)⌋ + 1（强制奇数）。

    对应论文 ``eq:SGFilter`` 的 ``m_w = ⌊T_w/(2τ)⌋``、``N_w = 2m_w + 1``。
    ``τ`` 由时间序列的相邻间隔估计。
    """
    time = np.asarray(time, dtype=float)
    if time.size < 2:
        return 1
    tau = float(time[1] - time[0])
    if tau <= 0.0:
        return 1
    m = int(np.floor(window_seconds / (2.0 * tau)))
    return max(1, 2 * m + 1)


def manoeuvre_intervals(table: DataTable):
    """Return the list of ``(start, end)`` index intervals flagged as manoeuvres."""
    flag = np.asarray(table.get("Flag").data)
    intervals = []
    in_manoeuvre = False
    start = None
    for i, value in enumerate(flag):
        if value == "manoeuvre" and not in_manoeuvre:
            start = i
            in_manoeuvre = True
        elif value == "normal" and in_manoeuvre:
            intervals.append((start, i))
            in_manoeuvre = False
            start = None
    if in_manoeuvre and start is not None:
        intervals.append((start, len(flag)))
    return intervals


def compute_derivatives(observation: DataTable) -> DataTable:
    """Central-difference MEE derivatives (manuscript Eq. 13)."""
    time = np.asarray(observation.get("Time").data, dtype=float)
    dt = float(time[1] - time[0])
    columns = [np.asarray(observation.get(name).data, dtype=float) for name in _MEE]
    columns[5] = np.unwrap(columns[5])
    derivatives = [np.gradient(col, dt) for col in columns]
    return DataTable.from_arrays(_MEEDOT_NAMES, [time, *derivatives], "MEEdotFD")


def gve_meedot(observation: DataTable) -> DataTable:
    """Natural MEE derivatives driven by the known perturbation (Gauss form)."""
    time = np.asarray(observation.get("Time").data, dtype=float)
    mee = np.column_stack([observation.get(n).data for n in _MEE]).astype(float)
    apn = np.column_stack([observation.get(n).data for n in ("Apn_r", "Apn_t", "Apn_n")]).astype(float)
    meedot = np.array([MEE_Dynamics(row, acc) for row, acc in zip(mee, apn)])
    return DataTable.from_arrays(_MEEDOT_NAMES, [time, *meedot.T], "MEEdotNatural")


def acceleration_from_meedot(observation: DataTable, meedot_table: DataTable, weights=None) -> DataTable:
    """反解摄动方程，由 ``MEEdot`` 得到 RTN 加速度（可加权）。"""
    time = np.asarray(observation.get("Time").data, dtype=float)
    mee = np.column_stack([observation.get(n).data for n in _MEE]).astype(float)
    _, pdot, fdot, gdot, hdot, kdot, Ldot = meedot_table.to_tuple()
    meedot = np.column_stack((pdot, fdot, gdot, hdot, kdot, Ldot)).astype(float)
    acc = np.array([MEEdot2Acc(m, md, weights=weights) for m, md in zip(mee, meedot)])
    return DataTable.from_arrays(["Time", "a_r", "a_t", "a_n"], [time, acc[:, 0], acc[:, 1], acc[:, 2]], "AccData")


def subtract_acceleration(total: DataTable, natural: DataTable) -> DataTable:
    """Thrust estimate = total acceleration - natural acceleration."""
    time = natural.get("Time")
    u_r = total.get("a_r").add(natural.get("a_r"), k=-1.0, new_name="u_r")
    u_t = total.get("a_t").add(natural.get("a_t"), k=-1.0, new_name="u_t")
    u_n = total.get("a_n").add(natural.get("a_n"), k=-1.0, new_name="u_n")
    return DataTable([time, u_r, u_t, u_n], "ThrustData")


def prepare_training_input(raw_thrust_path: str):
    """Align/normalise the unsmoothed MEEFD inversion for the sparse-regression stage.

    返回 ``(coeffs, training_table)``，其中 ``coeffs`` 用于反归一化。
    """
    table = DataTable.from_csv(raw_thrust_path)
    time = np.asarray(table.get("Time").data, dtype=float)
    time0 = float(time[0])
    time_aligned = DataSeries(time - time0, "Time")

    coeff_time, time_u = time_aligned.normalize()
    coeff_r, u_r = table.get("u_r").normalize()
    coeff_t, u_t = table.get("u_t").normalize()
    coeff_n, u_n = table.get("u_n").normalize()

    training = DataTable([time_u, u_r, u_t, u_n], "AccDataTrain")
    coeffs = (time0, coeff_time, coeff_r, coeff_t, coeff_n)
    return coeffs, training


def mee_state_matrix(table: DataTable) -> np.ndarray:
    """Build the MEE design matrix from the observed elements.

    Scale differences between columns are handled downstream by the library
    column normalisation (``normalize_columns``), not by pre-scaling ``p``.
    """
    _, p, f, g, h, k, L = table.to_tuple()
    return np.column_stack((p, f, g, h, k, L))


def truncate_fraction(table: DataTable, fraction: float) -> DataTable:
    """Return the first ``fraction`` of ``table`` rows (at least one).

    取前 ``fraction`` 比例的行（至少一行），用于部分窗口拟合。
    """
    fraction = float(np.clip(fraction, 0.0, 1.0))
    end = max(1, int(np.ceil(table.n * fraction)))
    return table.filter((0, end))


def evaluation_time(path: str, interpolation_rate: int) -> np.ndarray:
    """Upsampled time grid read from a CSV with a ``Time`` column."""
    table = DataTable.from_csv(path, columns=["Time"])
    series = DataSeries(np.asarray(table.get("Time").data, dtype=float), "Time")
    return np.asarray(series.linear_interp(interpolation_rate).data, dtype=float)


def saturate_thrust(thrust_path: str, u_max: float) -> DataTable:
    """Clip every thrust component to +/-u_max (actuator saturation), in place.

    将重构推力的三分量逐分量裁剪到 ±u_max（推力器饱和上限，m/s²），
    并就地写回 CSV。对四种方法的最终输出统一施加。
    """
    table = DataTable.from_csv(thrust_path)
    for name in ("u_r", "u_t", "u_n"):
        series = table.get(name)
        series.data = np.clip(np.asarray(series.data, dtype=float), -u_max, u_max)
    table.save(thrust_path)
    return table