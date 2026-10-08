"""Non-sparse baselines that post-process the MEEFD inversion."""

from __future__ import annotations

import os

import numpy as np

from ..config import AppConfig
from ..data import DataSeries, DataTable
from . import preprocessing as pre
from .libraries import time_library
from .regression import save_matrix

_OUTPUTS = ["u_r", "u_t", "u_n"]


def _save_coeffs(out_dir: str, Xi: np.ndarray, names: list[str]) -> None:
    """Save the least-squares coefficient vector for inspection."""
    save_matrix(Xi, os.path.join(out_dir, "Xi.csv"), row_headers=names, column_headers=_OUTPUTS)


def run_time_ols(raw_path: str, cfg: AppConfig, out_dir: str, thrust_path: str,
                 eval_raw_path: str | None = None) -> DataTable:
    """Run a non-sparse least-squares fit on the same time library as SINDy-TIME.

    ``eval_raw_path`` (optional) supplies the full time grid to evaluate on,
    enabling partial-window future prediction.
    """
    coeffs, training = pre.prepare_training_input(raw_path)
    t = np.array([training.get("Time").data]).T
    u = np.column_stack([training.get(name).data for name in _OUTPUTS])
    spec = time_library("sparse", alpha=0.0, max_iter=0, kappa=0.0)
    import pysindy as ps
    library = ps.CustomLibrary(library_functions=spec.functions).fit(t)
    theta = np.asarray(library.transform(t))
    Xi, *_ = np.linalg.lstsq(theta, u, rcond=None)
    os.makedirs(out_dir, exist_ok=True)
    _save_coeffs(out_dir, Xi, spec.names)
    save_matrix(theta, os.path.join(out_dir, "Theta.csv"), column_headers=spec.names)

    time0, coeff_time, coeff_r, coeff_t, coeff_n = coeffs
    if eval_raw_path is not None:
        full_time = pre.evaluation_time(eval_raw_path, cfg.sindy.interpolation_rate)
    else:
        t_up = training.get("Time").linear_interp(cfg.sindy.interpolation_rate)
        full_time = coeff_time * np.asarray(t_up.data, dtype=float) + time0
    tau = (full_time - time0) / coeff_time
    Xrec = np.asarray(library.transform(tau.reshape(-1, 1))) @ Xi
    save_matrix(Xrec, os.path.join(out_dir, "XRec.csv"), column_headers=_OUTPUTS)

    time = DataSeries(full_time, "Time")
    u_r = DataSeries(coeff_r * Xrec[:, 0], "u_r")
    u_t = DataSeries(coeff_t * Xrec[:, 1], "u_t")
    u_n = DataSeries(coeff_n * Xrec[:, 2], "u_n")
    result = DataTable([time, u_r, u_t, u_n], "TIMEOLS")
    result.save(thrust_path)
    return result


def run_meefd_spline(raw_path: str, cfg: AppConfig, out_dir: str, thrust_path: str,
                     smoothing_factor: float | None = None, eval_raw_path: str | None = None) -> DataTable:
    """Smooth each thrust component with a 1-D smoothing spline.

    ``smoothing_factor`` defaults to ``cfg.sindy.meefd_spline_smoothing_factor``;
    it scales the SciPy ``UnivariateSpline`` smoothing factor
    ``s = smoothing_factor * n * var(y)`` (larger -> smoother/fewer knots).
    ``eval_raw_path`` (optional) supplies the full time grid to evaluate on,
    enabling partial-window future prediction (fit on a truncated raw).
    """
    if smoothing_factor is None:
        smoothing_factor = cfg.sindy.meefd_spline_smoothing_factor
    coeffs, training = pre.prepare_training_input(raw_path)
    t = np.asarray(training.get("Time").data, dtype=float)
    u = np.column_stack([training.get(name).data for name in _OUTPUTS])
    from scipy.interpolate import UnivariateSpline
    splines = []
    for j in range(u.shape[1]):
        y = u[:, j]
        s = smoothing_factor * float(t.size) * float(np.var(y))
        spl = UnivariateSpline(t, y, s=s)
        splines.append(spl)
    os.makedirs(out_dir, exist_ok=True)

    time0, coeff_time, coeff_r, coeff_t, coeff_n = coeffs
    if eval_raw_path is not None:
        full_time = pre.evaluation_time(eval_raw_path, cfg.sindy.interpolation_rate)
    else:
        t_up = training.get("Time").linear_interp(cfg.sindy.interpolation_rate)
        full_time = coeff_time * np.asarray(t_up.data, dtype=float) + time0
    tau = (full_time - time0) / coeff_time
    Xrec = np.column_stack([spl(tau) for spl in splines])
    save_matrix(Xrec, os.path.join(out_dir, "XRec.csv"), column_headers=_OUTPUTS)

    time = DataSeries(full_time, "Time")
    u_r = DataSeries(coeff_r * Xrec[:, 0], "u_r")
    u_t = DataSeries(coeff_t * Xrec[:, 1], "u_t")
    u_n = DataSeries(coeff_n * Xrec[:, 2], "u_n")
    result = DataTable([time, u_r, u_t, u_n], "MEEFDSpline")
    result.save(thrust_path)
    return result
