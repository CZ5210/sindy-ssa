"""Data records shared by the dynamics and satellite modules.

动力学与卫星模块共用的数据记录。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimeParams:
    """Time grid definition. / 时间网格定义。"""

    hour: float
    dt: float
    tspan: tuple = field(init=False, default=None)
    n: int = field(init=False, default=None)
    point_per_hour: int = field(init=False, default=None)

    def __post_init__(self):
        self.tspan = (0.0, 3600.0 * self.hour)
        self.n = int((3600.0 * self.hour) // self.dt)
        self.point_per_hour = int(3600 // self.dt)

    @classmethod
    def from_dict(cls, data: dict) -> TimeParams:
        return cls(hour=data["hour"], dt=data["dt"])

    def to_dict(self) -> dict:
        return {"hour": self.hour, "dt": self.dt}


@dataclass
class SatStaticParam:
    """Constant satellite parameters. / 卫星静态参数。"""

    Cd: float
    Area: float
    c: float
    m0: float
    m: float
    Tmax: float

    def to_dict(self) -> dict:
        return {
            "Cd": self.Cd,
            "Area": self.Area,
            "c": self.c,
            "m0": self.m0,
            "m": self.m,
            "Tmax": self.Tmax,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SatStaticParam:
        return cls(**{k: data[k] for k in ("Cd", "Area", "c", "m0", "m", "Tmax")})


@dataclass
class SatDynamicParam:
    """Instantaneous satellite state. / 卫星瞬时状态。"""

    t: float
    MEE: np.ndarray
    MEEdot: np.ndarray
    Apn: np.ndarray
    U_LVLH: np.ndarray
    R: np.ndarray
    V: np.ndarray
    m: float


@dataclass
class SatLVLHParam:
    """Relative state expressed in the LVLH/RTN frame. / LVLH 相对状态。"""

    R: np.ndarray
    V: np.ndarray
    U: np.ndarray
