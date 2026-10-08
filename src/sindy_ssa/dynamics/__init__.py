"""Orbital dynamics, elements and propagation.

轨道动力学、根数与递推。

Module layout / 模块划分:
- ``state``         data records (time / satellite / LVLH parameters)
- ``elements``      modified equinoctial elements and variational equations
- ``frames``        frame transforms and linear relative-motion helpers
- ``perturbations`` gravity, J2 and atmospheric drag
- ``propagation``   RK4 integrator and the truth/reference propagators
- ``satellite``     scenario level satellite simulators
"""

from .elements import MEE2RV, RV2MEE, MEE_Dynamics, MEEdot2Acc
from .frames import (
    ECI2LVLH,
    LVLH2ECI,
    ComputeLQRControl,
    CWEquation,
    InitOrbit,
    Outside_Box,
)
from .perturbations import atmosphere_density, drag_acceleration_rtn, gravity, j2_acceleration_rtn
from .propagation import (
    RK4_Step,
    dynamics_perturbed,
    dynamics_reference,
    next_lvlh,
    next_perturbed,
    next_reference,
)
from .satellite import (
    BaseSatellite,
    SatelliteCollisionAvoidance,
    SatelliteStationKeeping,
)
from .state import SatDynamicParam, SatLVLHParam, SatStaticParam, TimeParams

__all__ = [
    "ECI2LVLH",
    "LVLH2ECI",
    "MEE2RV",
    "RV2MEE",
    "BaseSatellite",
    "CWEquation",
    "ComputeLQRControl",
    "InitOrbit",
    "MEE_Dynamics",
    "MEEdot2Acc",
    "Outside_Box",
    "RK4_Step",
    "SatDynamicParam",
    "SatLVLHParam",
    "SatStaticParam",
    "SatelliteCollisionAvoidance",
    "SatelliteStationKeeping",
    "TimeParams",
    "atmosphere_density",
    "drag_acceleration_rtn",
    "dynamics_perturbed",
    "dynamics_reference",
    "gravity",
    "j2_acceleration_rtn",
    "next_lvlh",
    "next_perturbed",
    "next_reference",
]
