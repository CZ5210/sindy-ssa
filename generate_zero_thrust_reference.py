"""Regenerate the two zero-thrust reference CSVs used by the batch figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from sindy_ssa.config import load_config
from sindy_ssa.data import DataTable
from sindy_ssa.dynamics import BaseSatellite, InitOrbit
from sindy_ssa.experiments.run_dynamics import run_dynamics
from sindy_ssa.experiments.run_sindy import sat_param_from_config
from sindy_ssa.metrics import compute_thrust_metrics, end_position_error
from sindy_ssa.registry import DEFAULT_ARCHIVE_ROOT, DEFAULT_BATCH_CONFIG
from sindy_ssa.sindy.preprocessing import manoeuvre_intervals


def calculate_reference(scenario: str, work_dir: Path) -> dict[str, float]:
    """Evaluate zero thrust over the full batch evaluation grid for one scenario."""
    config = load_config(str(ROOT / DEFAULT_BATCH_CONFIG), scenario=scenario)
    run_dynamics(config, str(work_dir))
    real_path = work_dir / "Data_real.csv"
    ref_path = work_dir / "Data_ref.csv"
    lvlh_path = work_dir / "Data_LVLH.csv"
    truth = DataTable.from_csv(real_path, columns=["Time", "u_r", "u_t", "u_n", "Flag"])

    sampled = truth.sample(config.observation.sampling_rate)
    interval = manoeuvre_intervals(sampled)[0]
    observed_time = sampled.get("Time").filter(interval)
    times = observed_time.linear_interp(config.sindy.interpolation_rate).data
    zeros = np.zeros_like(times)
    zero_path = work_dir / "U_zero.csv"
    # Match the thrust CSV precision used by the batch reconstruction.
    DataTable.from_arrays(["Time", "u_r", "u_t", "u_n"], [times, zeros, zeros, zeros]).save(zero_path)
    zero_thrust = DataTable.from_csv(zero_path)
    step = config.scenario.step_seconds
    steps = np.diff(zero_thrust.get("Time").data)
    if not steps.size or not np.allclose(steps, step, rtol=0, atol=1e-8):
        raise ValueError(f"{scenario}: batch evaluation grid must have a {step:g} s step")

    satellite = BaseSatellite(
        sat_param_from_config(config),
        *InitOrbit(config.scenario.orbit_altitude, config.scenario.orbit_inclination),
        config.satellite.initial_mass,
    )
    satellite.repropagate(zero_path, real_path, ref_path, step, zero_normal=False)
    satellite.save_repropagate_csv(str(work_dir))
    propagated = work_dir / "Data_reLVLH.csv"
    epe_planar, _ = end_position_error(lvlh_path, propagated, planar=True)
    epe, _ = end_position_error(lvlh_path, propagated, planar=False)
    ei = compute_thrust_metrics(truth, zero_thrust)
    return {
        "ei_planar": ei["ei_planar"],
        "ei_total": ei["ei_total"],
        "epe_planar": epe_planar,
        "epe": epe,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=ROOT / DEFAULT_ARCHIVE_ROOT,
        help="batch archive root (default: ./batch_archive beside this script)",
    )
    args = parser.parse_args(argv)

    output_dir = args.archive.resolve() / "zero_thrust_reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".generate-", dir=output_dir) as temporary:
        scratch = Path(temporary)
        pending = []
        for scenario in ("evasive", "stationkeeping"):
            print(f"Calculating zero-thrust reference for {scenario}", flush=True)
            metrics = calculate_reference(scenario, scratch / scenario)
            csv_path = scratch / f"{scenario}.csv"
            pd.DataFrame([metrics]).to_csv(csv_path, index=False)
            pending.append(csv_path)

        for csv_path in pending:
            output_path = csv_path.replace(output_dir / csv_path.name)
            print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
