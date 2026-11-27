"""Measure the finite-difference contribution to unsmoothed MEEFD thrust EI."""

from pathlib import Path

import numpy as np
import pandas as pd

from sindy_ssa.config import load_config
from sindy_ssa.data import DataTable
from sindy_ssa.metrics.evaluation import error_integral
from sindy_ssa.sindy import preprocessing as pre
from sindy_ssa.sindy.meefd import meefd_thrust

STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[3]  # <repo>/src/sindy_ssa/studies/m3_fd_error
DATA_DIR = ROOT / "data" / "stationkeeping"
TRUTH_PATH = DATA_DIR / "truth" / "Data_real.csv"
TRUTH_STEP_S = 10
EVAL_STEP_S = 60
MEE = ("p", "f", "g", "h", "k", "L")
THRUST = ("u_r", "u_t", "u_n")


def at_times(table: DataTable, times: np.ndarray) -> DataTable:
    """Select exact timestamps from a table on a strictly increasing grid."""
    source_times = table.get("Time").data
    indices = np.searchsorted(source_times, times)
    if np.any(indices == table.n) or not np.array_equal(source_times[indices], times):
        raise ValueError(f"{table.name} is missing requested evaluation timestamps")
    return DataTable.from_arrays(table.columns, [series.data[indices] for series in table.series], table.name)


def check_saved_inversion(fd60: DataTable) -> None:
    recomputed = pd.DataFrame(fd60.to_dict()).set_index("Time")
    for task in ("typical", "batch"):
        path = DATA_DIR / task / "ideal" / "methods" / "MEEFD-SG" / "U_MEEFD_inv.csv"
        if not path.is_file():
            print(f"No saved inversion at {path.relative_to(ROOT)}")
            continue
        columns = ("Time", *THRUST)
        saved = pd.read_csv(path, usecols=columns, index_col="Time")
        if not saved.index.is_unique:
            raise ValueError(f"Duplicate timestamps in {path.relative_to(ROOT)}")
        common = recomputed.index.intersection(saved.index)
        if common.empty:
            print(f"No common timestamps with {path.relative_to(ROOT)}")
            continue
        actual = recomputed.loc[common, list(THRUST)].to_numpy()
        archived = saved.loc[common, list(THRUST)].to_numpy()
        max_error = np.max(np.abs(actual - archived))
        # DataTable.save writes eight decimal places in scientific notation.
        matches = np.allclose(actual, archived, rtol=5e-9, atol=1e-18)
        print(
            f"{path.relative_to(ROOT)}: {len(common)}/{fd60.n} times, "
            f"max RTN difference {max_error:.6e} km/s^2, match at CSV precision: {matches}"
        )


def main() -> None:
    rate_names = tuple(f"{element}dot" for element in MEE)
    truth = DataTable.from_csv(TRUTH_PATH, name="truth")
    time = truth.get("Time").data
    if truth.n < 2 or not np.all(np.diff(time) == TRUTH_STEP_S):
        raise ValueError("Expected a strictly increasing truth grid with 10 s steps")
    flags = truth.get("Flag").data
    intervals = pre.manoeuvre_intervals(truth)
    if not np.all(np.isin(flags, ("normal", "manoeuvre"))) or len(intervals) != 1:
        raise ValueError("Expected normal/manoeuvre flags with one contiguous manoeuvre")
    start, stop = intervals[0]
    manoeuvre_start, manoeuvre_end = time[start], time[stop - 1]
    print(f"Truth: {truth.n} rows at {TRUTH_STEP_S} s, {time[0]:g} to {time[-1]:g} s")
    print(f"Flag=manoeuvre: rows {start}:{stop}, {manoeuvre_start:g} to {manoeuvre_end:g} s")

    cfg = load_config(str(ROOT / "configs" / "config.yaml"), scenario="stationkeeping")
    cfg.observation.noise = cfg.observation.get_noise("ideal")
    if cfg.observation.noise.mode != "none" or cfg.observation.sampling_rate != EVAL_STEP_S // TRUTH_STEP_S:
        raise ValueError("Expected the paper's ideal configuration with 6:1 downsampling")

    obs10 = truth.select(("Time", *MEE, "Apn_r", "Apn_t", "Apn_n", "Flag"))
    obs60 = obs10.sample(cfg.observation.sampling_rate)
    fd60 = meefd_thrust(obs60, cfg)
    fd10 = meefd_thrust(obs10, cfg)
    time60 = fd60.get("Time").data
    # The entire centered 60 s stencil must lie inside the contiguous manoeuvre.
    eval_times = time60[(time60 - EVAL_STEP_S >= manoeuvre_start) & (time60 + EVAL_STEP_S <= manoeuvre_end)]
    if eval_times.size < 2 or not np.all(np.diff(eval_times) == EVAL_STEP_S):
        raise ValueError("Expected a contiguous 60 s evaluation grid inside the manoeuvre")
    print(f"Evaluation: {eval_times[0]:g} to {eval_times[-1]:g} s, {eval_times.size} samples")

    eval_truth = at_times(truth.select(("Time", *rate_names, *THRUST)), eval_times)
    eval_obs = at_times(obs60, eval_times)
    rates = DataTable.from_arrays(
        ("Time", *(f"{element}_dot" for element in MEE)),
        eval_truth.select(("Time", *rate_names)).to_tuple(),
        "dynamics rates",
    )
    natural = pre.acceleration_from_meedot(eval_obs, pre.gve_meedot(eval_obs), weights=None)
    total = pre.acceleration_from_meedot(eval_obs, rates, weights=None)
    dynamics = pre.subtract_acceleration(total, natural)

    rows = []
    for method, step, reconstructed in (
        ("FD60", EVAL_STEP_S, at_times(fd60, eval_times)),
        ("FD10", TRUTH_STEP_S, at_times(fd10, eval_times)),
        ("dynamics_rates", None, dynamics),
    ):
        ei_r, ei_t, _, _ = error_integral(eval_truth, reconstructed)
        row = {
            "method": method,
            "derivative_step_s": step,
            "EI_r_vs_truth_m_s": ei_r,
            "EI_t_vs_truth_m_s": ei_t,
            "EI_2D_vs_truth_m_s": ei_r + ei_t,
        }
        if step is not None:
            # The reference is replaced before the pointwise difference and absolute sum.
            fd_r, fd_t, _, _ = error_integral(dynamics, reconstructed)
            row.update(EI_FD_r_m_s=fd_r, EI_FD_t_m_s=fd_t, EI_FD_2D_m_s=fd_r + fd_t)
        rows.append(row)

    max_errors = [np.max(np.abs(dynamics.get(axis).data - eval_truth.get(axis).data)) for axis in THRUST]
    print(
        "Dynamics rates vs. truth, max |error| in R/T/N (km/s^2): " + ", ".join(f"{error:.6e}" for error in max_errors)
    )
    check_saved_inversion(fd60)
    summary = pd.DataFrame(rows)
    summary["derivative_step_s"] = summary["derivative_step_s"].astype("Int64")
    summary.to_csv(STUDY_DIR / "summary.csv", index=False, float_format="%.15g")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.9g}"))


if __name__ == "__main__":
    main()
