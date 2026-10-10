"""SINDy-TIME: sparse regression of the MEEFD inversion on a time library.

SINDy-TIME：以时间为自变量的稀疏回归（对 MEEFD 反演做函数逼近）。
"""

from __future__ import annotations

import os

import numpy as np

from ..config import AppConfig
from ..data import DataSeries, DataTable
from . import preprocessing as pre
from .libraries import time_library
from .regression import fit_ssr, fit_ssr_auto, predict, save_matrix, save_selection

_OUTPUTS = ["u_r", "u_t", "u_n"]


def run_sindy_time(raw_path: str, cfg: AppConfig, out_dir: str, thrust_path: str, level: str = "sparse",
                   alpha: float | None = None, max_iter: int | None = None, kappa: float | None = None,
                   auto_select: bool = False, eval_raw_path: str | None = None) -> DataTable:
    """Run SINDy-TIME on the unsmoothed MEEFD inversion and save the reconstruction.

    ``auto_select=True`` ignores ``level`` and selects the number of SSR
    iterations by the L-curve elbow, keeping the ridge penalty from ``alpha``
    (default: ``cfg.sindy.time_alpha``).

    ``eval_raw_path`` (optional) supplies the full time grid to evaluate on,
    while the fit still uses ``raw_path``; this implements partial-window
    future prediction (fit on a truncated raw, extrapolate over the full arc).
    """
    coeffs, training = pre.prepare_training_input(raw_path)

    t = np.array([training.get("Time").data]).T
    u = np.column_stack([training.get(name).data for name in _OUTPUTS])

    if auto_select:
        alpha = alpha if alpha is not None else cfg.sindy.time_alpha
        tmp_spec = time_library(level, alpha=alpha, max_iter=0, kappa=0.0)
        Kmax = len(tmp_spec.functions) * u.shape[1]
        spec = time_library(level, alpha=alpha, max_iter=Kmax, kappa=0.0)
        library, optimizer, Xi, theta, selected_iter, path = fit_ssr_auto(
            spec, t, u, normalize_columns=cfg.sindy.time_normalize_columns,
            tol=cfg.sindy.sparsity_tol)
        save_selection(path, selected_iter, os.path.join(out_dir, "selection.json"),
                   tol=cfg.sindy.sparsity_tol, alpha=alpha)
    else:
        spec = time_library(level, alpha=alpha, max_iter=max_iter, kappa=kappa)
        library, optimizer, Xi, theta = fit_ssr(
            spec, t, u, normalize_columns=cfg.sindy.time_normalize_columns)

    os.makedirs(out_dir, exist_ok=True)
    save_matrix(Xi, os.path.join(out_dir, "Xi.csv"), row_headers=spec.names, column_headers=_OUTPUTS)
    save_matrix(theta, os.path.join(out_dir, "Theta.csv"), column_headers=spec.names)

    # evaluate on a (possibly full, upsampled) normalised time grid
    time0, coeff_time, coeff_r, coeff_t, coeff_n = coeffs
    if eval_raw_path is not None:
        full_time = pre.evaluation_time(eval_raw_path, cfg.sindy.interpolation_rate)
    else:
        t_up = training.get("Time").linear_interp(cfg.sindy.interpolation_rate)
        full_time = coeff_time * np.asarray(t_up.data, dtype=float) + time0
    tau = (full_time - time0) / coeff_time
    Xrec, _ = predict(optimizer, library, tau.reshape(-1, 1))
    save_matrix(Xrec, os.path.join(out_dir, "XRec.csv"), column_headers=_OUTPUTS)

    time = DataSeries(full_time, "Time")
    u_r = DataSeries(coeff_r * Xrec[:, 0], "u_r")
    u_t = DataSeries(coeff_t * Xrec[:, 1], "u_t")
    u_n = DataSeries(coeff_n * Xrec[:, 2], "u_n")

    result = DataTable([time, u_r, u_t, u_n], "SINDyTIME")
    result.save(thrust_path)
    return result
