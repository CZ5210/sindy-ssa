"""End-to-end reconstruction: observation -> six methods -> metrics.

端到端重建：观测 -> 六种方法 -> 指标。
"""

from __future__ import annotations

import os
import time

import numpy as np

from ..config import AppConfig
from ..data import DataTable
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
    """Observed (non-causal) MEE + perturbation states over the manoeuvre arc."""
    observation = DataTable.from_csv(obs_path)
    intervals = pre.manoeuvre_intervals(observation)
    cols = ["Time", "p", "f", "g", "h", "k", "L", "Apn_r", "Apn_t", "Apn_n"]
    win = observation.select(cols).filter(intervals[0])
    return win.linear_interp(interpolation_rate)


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

    ``verbose=True`` prints bilingual progress lines and timings.

    ``verbose`` 为 ``True`` 时打印中英双语进度与耗时；开启 ``prediction`` 时
    采用部分窗口拟合-预测模式。
    """
    t_start = time.perf_counter()
    scenario = config.scenario.name
    pred = config.sindy.prediction
    partial = bool(pred.enabled) and 0.0 < float(pred.fit_fraction) < 1.0
    fraction = float(pred.fit_fraction) if partial else 1.0
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
        if pred.future_state == "observed":
            eval_states = _observed_eval_states(obs_path, config.sindy.interpolation_rate)
        else:
            eval_states = _reference_eval_states(obs_path, ref_path)
        th0, th1 = _manoeuvre_time_window(obs_path)
        _truncate_thrust_to_time(meefd_thrust, th0 + fraction * (th1 - th0))
    else:
        raw_fit = raw_path
        eval_raw_path = None
        eval_states = None

    t0 = time.perf_counter()
    run_sindy_original(obs_path, config, dirs["SINDy"], os.path.join(dirs["SINDy"], METHODS["SINDy"]),
                       auto_select=True, eval_states=eval_states, fit_fraction=fraction)
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
                   auto_select=True, eval_states=eval_states, fit_fraction=fraction)
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
    # Actuator saturation limit as acceleration; a spacecraft-class prior
    # (Tmax / m), independent of the unknown true thrust.
    # Tmax/m is in m/s^2; thrust CSVs are stored in km/s^2 -> divide by 1000.
    u_max = config.satellite.max_thrust / config.satellite.initial_mass / 1000.0

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
