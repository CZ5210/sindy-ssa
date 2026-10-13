"""End-to-end reconstruction: observation -> six methods -> metrics.

端到端重建：观测 -> 六种方法 -> 指标。
"""

from __future__ import annotations

import os
import time

import numpy as np

from ..config import AppConfig
from ..data import DataSeries, DataTable
from ..dynamics import BaseSatellite, InitOrbit, SatStaticParam
from ..metrics import compute_thrust_metrics, end_position_error
from ..noise import simulate_observation
from ..registry import METHODS
from ..sindy import preprocessing as pre
from ..sindy import run_meefd_sg, run_meefd_spline, run_sindy_original, run_sindy_time, run_sindy_umee, run_time_ols
from ..sindy.preprocessing import saturate_thrust
from ..utils import ensure_dir, format_duration, progress


def sat_param_from_config(config: AppConfig) -> SatStaticParam:
    return SatStaticParam(
        Cd=config.satellite.drag_coefficient,
        Area=config.satellite.cross_section_area,
        c=config.satellite.exhaust_velocity,
        m0=config.satellite.dry_mass,
        m=config.satellite.initial_mass,
        Tmax=config.satellite.max_thrust,
    )


def ensure_dynamics(config: AppConfig) -> None:
    """Generate the truth dynamics if they are not present yet."""
    real_path = os.path.join(config.paths.data_dir, "Data_real.csv")
    if not os.path.exists(real_path):
        from .run_dynamics import run_dynamics

        run_dynamics(config)


def generate_observation(config: AppConfig, obs_path: str | None = None) -> str:
    data_dir = config.paths.data_dir
    obs_dir = config.paths.observation_dir
    ensure_dir(obs_dir)
    obs_path = obs_path or os.path.join(obs_dir, "ObserveData.csv")
    simulate_observation(
        os.path.join(data_dir, "Data_real.csv"),
        config.observation,
        obs_path,
    )
    return obs_path


_PREDICTIVE_METHODS = {"SINDy", "SINDy-TIME", "TIME-OLS", "SINDy-UMEE"}


def _manoeuvre_time_window(obs_path: str) -> tuple[float, float]:
    observation = DataTable.from_csv(obs_path)
    intervals = pre.manoeuvre_intervals(observation)
    obs_time = np.asarray(observation.get("Time").data, dtype=float)
    start, stop = intervals[0]
    return float(obs_time[start]), float(obs_time[stop - 1])


def _reference_eval_states(obs_path: str, ref_path: str) -> DataTable:
    """Reference-orbit states over the manoeuvre arc (causal future states)."""
    t0, t1 = _manoeuvre_time_window(obs_path)
    ref = DataTable.from_csv(ref_path)
    ref_time = np.asarray(ref.get("Time").data, dtype=float)
    i0 = int(np.argmin(np.abs(ref_time - t0)))
    i1 = int(np.argmin(np.abs(ref_time - t1))) + 1
    return ref.filter((i0, i1))


def _observed_eval_states(obs_path: str, interpolation_rate: int) -> DataTable:
    """Observed MEE + perturbation states over the manoeuvre arc.

    Used as the *known* part (fit window) of the ``propagated`` evaluation grid.
    """
    observation = DataTable.from_csv(obs_path)
    intervals = pre.manoeuvre_intervals(observation)
    cols = ["Time", "p", "f", "g", "h", "k", "L", "Apn_r", "Apn_t", "Apn_n"]
    win = observation.select(cols).filter(intervals[0])
    return win.linear_interp(interpolation_rate)


# --------------------------------------------------------------------------- #
# self-consistent evaluation grid (``future_state: propagated``)
#
# The two state-based methods are evaluated on MEE states, so a causal forecast
# needs states for the part of the arc that was not observed.  Instead of taking
# them from another trajectory (``reference``), ``propagated`` closes the loop: the
# identified thrust is used to re-propagate the reconstructed trajectory, and the
# model is then re-evaluated on those states.  Repeating this sweep makes the
# future states and the future thrust mutually consistent.
# --------------------------------------------------------------------------- #
_STATE_GRID_COLUMNS = ["Time", "p", "f", "g", "h", "k", "L", "Apn_r", "Apn_t", "Apn_n"]
_STATE_METHODS = ("SINDy", "SINDy-UMEE")
_THRUST_COMPONENTS = ("u_r", "u_t", "u_n")

# ``propagated`` sweep settings.  Internal constants (not user-facing): the
# sequence is under-relaxed so it stays monotone and cannot run away if the loop
# gain is high, and the sweep count is where the marginal EI improvement no longer
# justifies the extra model evaluations.  Measured on a station-keeping
# reconstruction (EI in m/s, actuator limit 2.125e-7 km/s^2): ``reference``
# grid 18.2 (42 % clipped),
# 1 sweep 6.7, 3 sweeps 6.6, 8 sweeps 6.0 (21 % clipped), 10 sweeps 5.8.
_PROPAGATED_SWEEPS = 3
_PROPAGATED_RELAXATION = 0.5


def _fit_horizon_time(obs_path: str, fit_fraction: float) -> float:
    """Time of the last observation sample used for identification.

    Mirrors the fit window of ``run_sindy_original`` / ``run_sindy_umee`` (first
    ``fit_fraction`` of the detected manoeuvre interval).
    """
    observation = DataTable.from_csv(obs_path)
    intervals = pre.manoeuvre_intervals(observation)
    time = np.asarray(observation.get("Time").data, dtype=float)
    start, stop = intervals[0]
    fit_stop = start + max(1, int(np.ceil((stop - start) * fit_fraction)))
    return float(time[fit_stop - 1])


def _propagated_state_grid(satellite: BaseSatellite) -> DataTable:
    """``(Time, MEE, Apn)`` of the trajectory re-propagated with the thrust estimate."""
    rows = np.asarray(
        [(data.t, *data.MEE, *data.Apn) for data in satellite.Data_reprop_real_list],
        dtype=float,
    )
    return DataTable.from_arrays(
        _STATE_GRID_COLUMNS, [rows[:, i] for i in range(rows.shape[1])], "PropagatedStates"
    )


def _hybrid_state_grid(observed: DataTable, propagated: DataTable, horizon: float) -> DataTable:
    """Observed states up to ``horizon``, re-propagated states after it.

    Inside the fit window the states are known (and define the domain the model
    was identified on); only the forecast part is taken from the re-propagation.
    """
    obs = observed.select(_STATE_GRID_COLUMNS)
    prop = propagated.select(_STATE_GRID_COLUMNS)
    keep_obs = np.asarray(obs.get("Time").data, dtype=float) <= horizon
    keep_prop = np.asarray(prop.get("Time").data, dtype=float) > horizon
    columns = [
        np.concatenate([obs.get(name).data[keep_obs], prop.get(name).data[keep_prop]])
        for name in _STATE_GRID_COLUMNS
    ]
    return DataTable.from_arrays(_STATE_GRID_COLUMNS, columns, "EvalStates")


def _clip_thrust(table: DataTable, u_max: float) -> DataTable:
    """Clip the RTN thrust components to the actuator limit (returns a copy)."""
    return DataTable(
        [
            DataSeries(np.clip(np.asarray(s.data, dtype=float), -u_max, u_max), s.name)
            if s.name in _THRUST_COMPONENTS
            else s.copy()
            for s in table.series
        ],
        name=table.name,
    )


def _relax_thrust(previous: DataTable | None, current: DataTable, relaxation: float) -> DataTable:
    """Under-relaxed thrust history driving the re-propagation."""
    lam = float(np.clip(relaxation, 0.0, 1.0))
    if previous is None or lam >= 1.0 or previous.n != current.n:
        return current
    series = []
    for column in current.series:
        if column.name in _THRUST_COMPONENTS:
            blended = (1.0 - lam) * np.asarray(previous.get(column.name).data, dtype=float)
            blended = blended + lam * np.asarray(column.data, dtype=float)
            series.append(DataSeries(blended, column.name))
        else:
            series.append(column.copy())
    return DataTable(series, name=current.name)


def _evaluate_state_method(name: str, config: AppConfig, obs_path: str, raw_fit: str,
                           fraction: float, eval_grid: DataTable, out_dir: str) -> DataTable:
    """Run one state-based method on ``eval_grid`` (used by the sweep only)."""
    thrust_path = os.path.join(out_dir, f"U_{name}.csv")
    if name == "SINDy":
        return run_sindy_original(obs_path, config, out_dir, thrust_path, auto_select=True,
                                  eval_states=eval_grid, fit_fraction=fraction)
    return run_sindy_umee(obs_path, raw_fit, config, out_dir, thrust_path, auto_select=True,
                          eval_states=eval_grid, fit_fraction=fraction)


def _self_consistent_state_grids(config: AppConfig, obs_path: str, real_path: str, ref_path: str,
                                 raw_fit: str, fraction: float, u_max: float, dirs: dict,
                                 verbose: bool = False) -> dict[str, DataTable]:
    """Fixed-point state grids for SINDy / SINDy-UMEE (``future_state: propagated``).

    Each method is swept on its own grid: evaluate -> clip the thrust to the
    actuator limit -> re-propagate the reconstructed trajectory -> keep the
    propagated states beyond the fit horizon, ``_PROPAGATED_SWEEPS`` times with
    ``_PROPAGATED_RELAXATION`` under-relaxation.
    """
    iterations = _PROPAGATED_SWEEPS
    observed = _observed_eval_states(obs_path, config.sindy.interpolation_rate)
    horizon = _fit_horizon_time(obs_path, fraction)
    satellite = BaseSatellite(
        sat_param_from_config(config),
        *InitOrbit(config.scenario.orbit_altitude, config.scenario.orbit_inclination),
        config.satellite.initial_mass,
    )
    grids = {name: _reference_eval_states(obs_path, ref_path) for name in _STATE_METHODS}
    history: dict[str, DataTable | None] = {name: None for name in _STATE_METHODS}

    for k in range(iterations):
        if verbose:
            progress(f"自洽推演迭代 {k + 1}/{iterations}",
                     f"self-consistent sweep {k + 1}/{iterations}")
        for name in _STATE_METHODS:
            thrust = _evaluate_state_method(
                name, config, obs_path, raw_fit, fraction, grids[name],
                os.path.join(dirs[name], "propagation_iter", f"iter{k}"),
            )
            history[name] = _relax_thrust(history[name], thrust, _PROPAGATED_RELAXATION)
            thrust_path = os.path.join(dirs[name], "propagation_iter", f"U_iter{k}.csv")
            _clip_thrust(history[name], u_max).save(thrust_path)
            satellite.repropagate(thrust_path, real_path, ref_path,
                                  config.scenario.step_seconds, zero_normal=False)
            grids[name] = _hybrid_state_grid(observed, _propagated_state_grid(satellite), horizon)
    return grids


def _truncate_thrust_to_time(thrust_path: str, horizon: float) -> None:
    """Keep only thrust samples up to ``horizon`` (seconds), in place."""
    table = DataTable.from_csv(thrust_path)
    time = np.asarray(table.get("Time").data, dtype=float)
    idx = np.flatnonzero(time <= horizon)
    end = int(idx[-1]) + 1 if idx.size else 1
    table.filter((0, end)).save(thrust_path)


def run_reconstruction(config: AppConfig, obs_path: str | None = None, verbose: bool = False) -> dict:
    """Run the full pipeline and return per-method metric dictionaries.

    When ``config.sindy.prediction.enabled`` the methods are fit on the first
    ``fit_fraction`` of the manoeuvre arc and predict the rest:

    - predictive (``SINDy``, ``SINDy-TIME``, ``TIME-OLS``, ``SINDy-UMEE``) are
      evaluated over the full arc (re-propagation + EPE/EI);
    - non-predictive (``MEEFD-SG``, ``MEEFD-Spline``) only expose the fit-horizon
      segment and do **not** participate in re-propagation / EPE / EI.

    The two state-based methods (``SINDy``, ``SINDy-UMEE``) are evaluated on an
    MEE grid selected by ``config.sindy.prediction.future_state``
    (``reference`` / ``propagated``); see
    :class:`sindy_ssa.config.PredictionConfig` and :func:`_self_consistent_state_grids`.

    ``verbose=True`` prints bilingual progress lines and timings.

    ``verbose`` 为 ``True`` 时打印中英双语进度与耗时；开启 ``prediction`` 时
    采用部分窗口拟合-预测模式。
    """
    t_start = time.perf_counter()
    scenario = config.scenario.name
    pred = config.sindy.prediction
    partial = bool(pred.enabled) and 0.0 < float(pred.fit_fraction) < 1.0
    fraction = float(pred.fit_fraction) if partial else 1.0
    if partial and pred.future_state not in ("reference", "propagated"):
        raise ValueError(
            "unknown prediction.future_state: "
            f"{pred.future_state!r}; expected reference | propagated"
        )
    if verbose:
        mode = "partial-window prediction" if partial else "full"
        progress(f"端到端重建开始（{scenario}, {mode}）",
                 f"reconstruction started ({scenario}, {mode})")

    data_dir = config.paths.data_dir
    methods_dir = config.paths.methods_dir
    ensure_dir(methods_dir)
    ensure_dynamics(config)

    real_path = os.path.join(data_dir, "Data_real.csv")
    ref_path = os.path.join(data_dir, "Data_ref.csv")
    truth_lvlh = os.path.join(data_dir, "Data_LVLH.csv")
    obs_path = obs_path or os.path.join(config.paths.observation_dir, "ObserveData.csv")
    t0 = time.perf_counter()
    obs_path = generate_observation(config, obs_path)
    if verbose:
        progress("观测生成完成", "observation generated", time.perf_counter() - t0)

    dirs = {name: os.path.join(methods_dir, name) for name in METHODS}
    for path in dirs.values():
        ensure_dir(path)

    raw_path = os.path.join(dirs["MEEFD-SG"], "U_MEEFD_inv.csv")
    meefd_thrust = os.path.join(dirs["MEEFD-SG"], METHODS["MEEFD-SG"])

    t0 = time.perf_counter()
    run_meefd_sg(obs_path, config, raw_path, meefd_thrust)
    runtime_meefd = time.perf_counter() - t0
    if verbose:
        progress("MEEFD 反演与 SG 完成", "MEEFD inversion + SG done", runtime_meefd)

    # Partial-window bookkeeping: truncated fit inputs + full evaluation grid.
    if partial:
        # Drop stale re-propagation outputs of non-predictive methods so the
        # reprop figure never shows methods that cannot forecast.
        for name in METHODS:
            if name not in _PREDICTIVE_METHODS:
                for suffix in ("", "_3D"):
                    stale = os.path.join(dirs[name], f"Data_reLVLH{suffix}.csv")
                    if os.path.exists(stale):
                        os.remove(stale)
        raw_fit = os.path.join(dirs["MEEFD-SG"], "U_MEEFD_inv_fit.csv")
        pre.truncate_fraction(DataTable.from_csv(raw_path), fraction).save(raw_fit)
        eval_raw_path = raw_path
        if pred.future_state == "propagated":
            # Built below, after the MEEFD inversion is available.
            eval_states = None
        else:
            eval_states = _reference_eval_states(obs_path, ref_path)
        th0, th1 = _manoeuvre_time_window(obs_path)
        _truncate_thrust_to_time(meefd_thrust, th0 + fraction * (th1 - th0))
    else:
        raw_fit = raw_path
        eval_raw_path = None
        eval_states = None

    # Actuator saturation limit as acceleration; a spacecraft-class prior
    # (Tmax / m), independent of the unknown true thrust.
    # Tmax/m is in m/s^2; thrust CSVs are stored in km/s^2 -> divide by 1000.
    u_max = config.satellite.max_thrust / config.satellite.initial_mass / 1000.0

    # Evaluation grid of the state-based methods (SINDy / SINDy-UMEE).
    if partial and pred.future_state == "propagated":
        state_grids = _self_consistent_state_grids(
            config, obs_path, real_path, ref_path, raw_fit, fraction, u_max, dirs, verbose=verbose
        )
    else:
        state_grids = {name: eval_states for name in _STATE_METHODS}

    t0 = time.perf_counter()
    run_sindy_original(obs_path, config, dirs["SINDy"], os.path.join(dirs["SINDy"], METHODS["SINDy"]),
                       auto_select=True, eval_states=state_grids["SINDy"], fit_fraction=fraction)
    runtime_sindy = time.perf_counter() - t0
    if verbose:
        progress("SINDy 重建完成", "SINDy reconstruction done", runtime_sindy)

    t0 = time.perf_counter()
    run_sindy_time(raw_fit, config, dirs["SINDy-TIME"], os.path.join(dirs["SINDy-TIME"], METHODS["SINDy-TIME"]),
                   auto_select=True, eval_raw_path=eval_raw_path)
    runtime_sindy_time = time.perf_counter() - t0
    if verbose:
        progress("SINDy-TIME 重建完成", "SINDy-TIME reconstruction done", runtime_sindy_time)

    t0 = time.perf_counter()
    run_time_ols(raw_fit, config, dirs["TIME-OLS"], os.path.join(dirs["TIME-OLS"], METHODS["TIME-OLS"]),
                 eval_raw_path=eval_raw_path)
    runtime_time_ols = time.perf_counter() - t0
    if verbose:
        progress("TIME-OLS 拟合完成", "TIME-OLS fit done", runtime_time_ols)

    t0 = time.perf_counter()
    run_meefd_spline(raw_fit, config, dirs["MEEFD-Spline"], os.path.join(dirs["MEEFD-Spline"], METHODS["MEEFD-Spline"]))
    runtime_meefd_spline = time.perf_counter() - t0
    if verbose:
        progress("MEEFD-Spline 拟合完成", "MEEFD-Spline fit done", runtime_meefd_spline)

    t0 = time.perf_counter()
    run_sindy_umee(obs_path, raw_fit, config, dirs["SINDy-UMEE"], os.path.join(dirs["SINDy-UMEE"], METHODS["SINDy-UMEE"]),
                   auto_select=True, eval_states=state_grids["SINDy-UMEE"], fit_fraction=fraction)
    runtime_sindy_umee = time.perf_counter() - t0
    if verbose:
        progress("SINDy-UMEE 重建完成", "SINDy-UMEE reconstruction done", runtime_sindy_umee)

    runtimes = {
        "MEEFD-SG": runtime_meefd,
        "TIME-OLS": runtime_time_ols,
        "MEEFD-Spline": runtime_meefd_spline,
        "SINDy": runtime_sindy,
        "SINDy-TIME": runtime_sindy_time,
        "SINDy-UMEE": runtime_sindy_umee,
    }

    truth_thrust = DataTable.from_csv(real_path, columns=["Time", "u_r", "u_t", "u_n"])
    sat_param = sat_param_from_config(config)
    R0, V0 = InitOrbit(config.scenario.orbit_altitude, config.scenario.orbit_inclination)
    m = config.satellite.initial_mass

    results: dict[str, dict] = {}
    if verbose:
        progress("重传播与评估", "re-propagating reconstructed thrust and computing metrics")
    for name, filename in METHODS.items():
        if partial and name not in _PREDICTIVE_METHODS:
            # Non-predictive: no full-arc thrust -> skip re-propagation and EI/EPE.
            continue
        t_rep = time.perf_counter()
        thrust_path = os.path.join(dirs[name], filename)
        saturate_thrust(thrust_path, u_max)

        satellite = BaseSatellite(sat_param, R0, V0, m)
        # Primary: re-propagate the full 3D reconstructed thrust.
        satellite.repropagate(thrust_path, real_path, ref_path,
                              config.scenario.step_seconds, zero_normal=False)
        satellite.save_repropagate_csv(dirs[name], suffix="_3D")
        epe_3d, _ = end_position_error(
            truth_lvlh, os.path.join(dirs[name], "Data_reLVLH_3D.csv"), planar=False)
        # Secondary: re-propagate under the coplanar-maneuver prior (u_n = 0).
        satellite.repropagate(thrust_path, real_path, ref_path,
                              config.scenario.step_seconds, zero_normal=True)
        satellite.save_repropagate_csv(dirs[name])
        epe_planar, _ = end_position_error(
            truth_lvlh, os.path.join(dirs[name], "Data_reLVLH.csv"), planar=True)

        rec = DataTable.from_csv(thrust_path)
        metrics = compute_thrust_metrics(truth_thrust, rec)
        metrics["epe"] = epe_3d
        metrics["epe_planar"] = epe_planar
        metrics["method"] = name
        metrics["runtime_seconds"] = runtimes[name]
        results[name] = metrics
        if verbose:
            progress(f"{name} 重传播完成（EPE {epe_3d:.3f} km）",
                     f"{name} re-propagated (EPE {epe_3d:.3f} km)",
                     time.perf_counter() - t_rep)

    total = time.perf_counter() - t_start
    if verbose:
        progress(f"端到端重建完成（{scenario}）", f"reconstruction done ({scenario})", total)
        print("    单次运行各方法耗时 / per-method runtime:", flush=True)
        for name, rt in runtimes.items():
            print(f"      {name:<12s} {format_duration(rt):>10s}", flush=True)
        print(f"    方法合计 / sum of method runtimes: {format_duration(sum(runtimes.values()))}", flush=True)
        print(f"    端到端总耗时 / total wall time: {format_duration(total)}", flush=True)

    return results
