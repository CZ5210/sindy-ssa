"""SINDy-UMEE: sparse regression of the MEEFD inversion on an MEE library.

SINDy-UMEE：以 MEE 状态为自变量的稀疏回归。与 SINDy-TIME 一致，回归全部 (R, T, N) 三分量。
"""

from __future__ import annotations

import os

import numpy as np

from ..config import AppConfig
from ..data import DataSeries, DataTable
from . import preprocessing as pre
from .libraries import umee_library
from .regression import fit_ssr, fit_ssr_auto, predict, save_matrix, save_selection

_OUTPUTS = ["u_r", "u_t", "u_n"]


def run_sindy_umee(observe_path: str, raw_path: str, cfg: AppConfig, out_dir: str, thrust_path: str,
                   auto_select: bool = False, eval_states: DataTable | None = None,
                   fit_fraction: float = 1.0) -> DataTable:
    """Run SINDy-UMEE on the unsmoothed MEEFD inversion and save the reconstruction.

    ``auto_select=True`` fixes the ridge penalty from the config and selects
    the number of SSR iterations by the L-curve elbow.

    ``fit_fraction`` (< 1) fits on the first fraction of the manoeuvre arc.
    ``eval_states`` (optional) supplies the MEE states on which to evaluate the
    identified ``u = f(MEE)`` model; used for partial-window future prediction
    (pass reference-orbit MEE for a causal forecast).
    """
    coeffs, training = pre.prepare_training_input(raw_path)

    observation = DataTable.from_csv(observe_path)
    intervals = pre.manoeuvre_intervals(observation)
    if not intervals:
        raise ValueError(f"no manoeuvre interval found in {observe_path}")
    start, stop = intervals[0]
    fit_stop = start + max(1, int(np.ceil((stop - start) * fit_fraction)))
    mee_data = observation.select(["Time", "p", "f", "g", "h", "k", "L"]).filter((start, fit_stop))

    X = pre.mee_state_matrix(mee_data)
    u = np.column_stack([training.get(name).data for name in _OUTPUTS])

    if auto_select:
        tmp_spec = umee_library(0, cfg.sindy.umee_alpha, kappa=0.0)
        Kmax = len(tmp_spec.functions) * u.shape[1]
        spec = umee_library(Kmax, cfg.sindy.umee_alpha, kappa=0.0)
        library, optimizer, Xi, theta, selected_iter, path = fit_ssr_auto(
            spec, X, u, normalize_columns=cfg.sindy.normalize_columns,
            tol=cfg.sindy.sparsity_tol)
        save_selection(path, selected_iter, os.path.join(out_dir, "selection.json"))
    else:
        spec = umee_library(cfg.sindy.umee_max_iter, cfg.sindy.umee_alpha)
        library, optimizer, Xi, theta = fit_ssr(
            spec, X, u, normalize_columns=cfg.sindy.normalize_columns)

    os.makedirs(out_dir, exist_ok=True)
    save_matrix(Xi, os.path.join(out_dir, "Xi.csv"), row_headers=spec.names, column_headers=_OUTPUTS)
    save_matrix(theta, os.path.join(out_dir, "Theta.csv"), column_headers=spec.names)

    if eval_states is not None:
        eval_data = eval_states.select(["Time", "p", "f", "g", "h", "k", "L"])
    else:
        eval_data = mee_data.linear_interp(cfg.sindy.interpolation_rate)
    Xf = pre.mee_state_matrix(eval_data)
    Xrec, _ = predict(optimizer, library, Xf)
    save_matrix(Xrec, os.path.join(out_dir, "XRec.csv"), column_headers=_OUTPUTS)

    time = np.asarray(eval_data.get("Time").data, dtype=float)
    _, _, coeff_r, coeff_t, coeff_n = coeffs
    result = DataTable(
        [
            DataSeries(time, "Time"),
            DataSeries(coeff_r * Xrec[:, 0], "u_r"),
            DataSeries(coeff_t * Xrec[:, 1], "u_t"),
            DataSeries(coeff_n * Xrec[:, 2], "u_n"),
        ],
        "SINDyUMEE",
    )
    result.save(thrust_path)
    return result
