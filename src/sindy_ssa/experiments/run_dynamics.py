"""Scenario truth-dynamics generation. / 场景真值动力学生成。"""

from __future__ import annotations

import os

from ..config import AppConfig
from ..dynamics import (
    InitOrbit,
    SatelliteCollisionAvoidance,
    SatelliteStationKeeping,
    SatStaticParam,
    TimeParams,
)
from ..utils import ensure_dir, save_json


def build_satellite(config: AppConfig):
    """Instantiate the scenario satellite (without propagating)."""
    sat_param = SatStaticParam(
        Cd=config.satellite.drag_coefficient,
        Area=config.satellite.cross_section_area,
        c=config.satellite.exhaust_velocity,
        m0=config.satellite.dry_mass,
        m=config.satellite.initial_mass,
        Tmax=config.satellite.max_thrust,
    )
    R0, V0 = InitOrbit(config.scenario.orbit_altitude, config.scenario.orbit_inclination)
    m = config.satellite.initial_mass

    if config.scenario.name == "stationkeeping":
        satellite = SatelliteStationKeeping(
            sat_param, R0, V0, m,
            BoxSize=config.scenario.box_size,
            control_config=config.scenario.stationkeeping_control,
        )
    elif config.scenario.name == "evasive":
        satellite = SatelliteCollisionAvoidance(
            sat_param, R0, V0, m,
            TargetPos=config.scenario.target_position,
            TOF=config.scenario.time_of_flight_hours,
            control_config=config.scenario.evasive_control,
        )
    else:
        raise ValueError(f"unknown scenario: {config.scenario.name!r}")
    return satellite, sat_param


def run_dynamics(config: AppConfig, output_dir: str | None = None) -> str:
    """Propagate the truth/reference satellites and persist the CSVs."""
    output_dir = output_dir or config.paths.data_dir
    ensure_dir(output_dir)

    satellite, sat_param = build_satellite(config)
    satellite.reference_dir = output_dir

    time_param = TimeParams(config.scenario.duration_hours, config.scenario.step_seconds)
    satellite.propagate(time_param)
    satellite.save_csv(output_dir, precision=8)

    save_json(sat_param.to_dict(), os.path.join(output_dir, "SatParam.json"))
    save_json(time_param.to_dict(), os.path.join(output_dir, "TimeParam.json"))
    return output_dir
