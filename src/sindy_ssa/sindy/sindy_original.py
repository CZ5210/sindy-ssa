"""SINDy adapted to manoeuvre reconstruction (baseline).

标准 SINDy 用于机动重建的基线。
"""

from __future__ import annotations

import os

import numpy as np

from ..config import AppConfig
from ..data import DataTable
from ..noise import mee_weights_for_tier
from . import preprocessing as pre
from .libraries import original_library
from .regression import fit_ssr, fit_ssr_auto, predict, save_matrix, save_selection


def run_sindy_original(observe_path: str, cfg: AppConfig, out_dir: str, thrust_path: str,
                       auto_select: bool = False, eval_states: DataTable | None = None,
                       fit_fraction: float = 1.0) -> DataTable:
    """Run SINDy on the observed MEE derivatives.

    ``auto_select=True`` fixes the ridge penalty ``alpha`` from the config and
    selects the number of SSR iterations by the L-curve elbow.

    ``fit_fraction`` (< 1) fits on the first fraction of the manoeuvre arc.
    ``eval_states`` (optional) supplies state + perturbation columns on which to
    evaluate the identified derivative model; used for partial-window future
    prediction (pass reference-orbit states for a causal forecast).
    """
    observation = DataTable.from_csv(observe_path)
    intervals = pre.manoeuvre_intervals(observation)
    if not intervals:
        raise ValueError(f"no manoeuvre interval found in {observe_path}")
    start, stop = intervals[0]
    fit_stop = start + max(1, int(np.ceil((stop - start) * fit_fraction)))
    window = (start, fit_stop)

    derivatives = pre.compute_derivatives(observation).filter(window)
    mee_data = observation.select(["Time", "p", "f", "g", "h", "k", "L"]).filter(window)
    obs_win = observation.filter(window)

    X = pre.mee_state_matrix(mee_data)
    _, *dots = derivatives.to_tuple()
    Xdot = np.column_stack(dots)

    if auto_select:
        tmp_spec = original_library(0, cfg.sindy.original_alpha, kappa=0.0)
        Kmax = len(tmp_spec.functions) * Xdot.shape[1]
        spec = original_library(Kmax, cfg.sindy.original_alpha, kappa=0.0)
        library, optimizer, Xi, theta, selected_iter, path = fit_ssr_auto(
            spec, X, Xdot, normalize_columns=cfg.sindy.normalize_columns,
            tol=cfg.sindy.sparsity_tol)
        save_selection(path, selected_iter, os.path.join(out_dir, "selection.json"))
    else:
        spec = original_library(cfg.sindy.original_max_iter, cfg.sindy.original_alpha)
        library, optimizer, Xi, theta = fit_ssr(
            spec, X, Xdot, normalize_columns=cfg.sindy.normalize_columns)
    Xrec, theta_rec = predict(optimizer, library, X)

    os.makedirs(out_dir, exist_ok=True)
    derivative_names = ["p_dot", "f_dot", "g_dot", "h_dot", "k_dot", "L_dot"]
    save_matrix(Xi, os.path.join(out_dir, "Xi.csv"), row_headers=spec.names, column_headers=derivative_names)
    save_matrix(theta, os.path.join(out_dir, "Theta.csv"), column_headers=spec.names)
    save_matrix(Xrec, os.path.join(out_dir, "XRec.csv"), column_headers=derivative_names)

    weights = None if cfg.observation.noise.mode == "none" else mee_weights_for_tier(cfg.observation.noise.tier)

    if eval_states is not None:
        eval_win = eval_states.select(
            ["Time", "p", "f", "g", "h", "k", "L", "Apn_r", "Apn_t", "Apn_n"]
        )
        Xeval = pre.mee_state_matrix(eval_win.select(["Time", "p", "f", "g", "h", "k", "L"]))
        Xrec_eval, _ = predict(optimizer, library, Xeval)
        eval_time = np.asarray(eval_win.get("Time").data, dtype=float)
        meedot_eval = DataTable.from_arrays(
            ["Time", *derivative_names],
            [eval_time, *Xrec_eval.T],
            "MEEdotSINDy",
        )
        natural = pre.gve_meedot(eval_win)
        u_natural = pre.acceleration_from_meedot(eval_win, natural, weights=weights)
        u_total = pre.acceleration_from_meedot(eval_win, meedot_eval, weights=weights)
        # eval grid is already the target grid -> no up-sampling
        thrust = pre.subtract_acceleration(u_total, u_natural)
        thrust.save(thrust_path)
        del theta_rec
        return thrust

    time = np.asarray(mee_data.get("Time").data, dtype=float)
    meedot_rec = DataTable.from_arrays(
        ["Time", *derivative_names],
        [time, Xrec[:, 0], Xrec[:, 1], Xrec[:, 2], Xrec[:, 3], Xrec[:, 4], Xrec[:, 5]],
        "MEEdotSINDy",
    )

    natural = pre.gve_meedot(obs_win)
    u_natural = pre.acceleration_from_meedot(obs_win, natural, weights=weights)
    u_total = pre.acceleration_from_meedot(obs_win, meedot_rec, weights=weights)
    thrust = (
        pre.subtract_acceleration(u_total, u_natural)
        .linear_interp(cfg.sindy.interpolation_rate)
    )
    thrust.save(thrust_path)
    del theta_rec
    return thrust
