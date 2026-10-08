"""Perturbation accelerations: gravity, J2 and atmospheric drag.

摄动加速度：引力、J2 与大气阻力。
"""

from __future__ import annotations

import numpy as np

from ..constants import J2, MU, R_EARTH_EQ, R_EARTH_MEAN

# US Standard Atmosphere 1976 lookup tables (Curtis, 2014).
_ALT_KM = [
    0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
    150, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000,
]
_DENSITY = [
    1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5,
    1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9,
    2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12,
    2.803e-12, 1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14,
    5.759e-15, 3.561e-15,
]
_SCALE_HEIGHT = [
    7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661, 5.927, 5.533, 5.703,
    6.782, 9.973, 13.243, 16.322, 21.652, 27.974, 34.934, 43.342, 49.755,
    54.513, 58.019, 60.980, 65.654, 76.377, 100.587, 147.203, 208.020,
]


def atmosphere_density(altitude_km: float) -> float:
    """US76 atmospheric density [kg/m^3] at a geometric altitude [km]."""
    z = altitude_km
    if z > 1000:
        return 0.0
    if z < 0:
        return 1.225

    index = 0
    for j in range(27):
        if _ALT_KM[j] <= z < _ALT_KM[j + 1]:
            index = j
            break
    if z == 1000:
        index = 26

    return _DENSITY[index] * np.exp(-(z - _ALT_KM[index]) / _SCALE_HEIGHT[index])


def gravity(R, mu: float = MU) -> np.ndarray:
    """Two-body gravitational acceleration [km/s^2]."""
    R = np.asarray(R, dtype=float)
    return -(mu / np.linalg.norm(R) ** 3) * R


def j2_acceleration_rtn(MEE, J2_coeff: float = J2, mu: float = MU, Re: float = R_EARTH_EQ) -> np.ndarray:
    """J2 acceleration expressed in the RTN frame.

    J2 摄动在 RTN 系下的表达式，其中 ``s^2 = 1 + h^2 + k^2``。
    """
    p, f, g, h, k, L = MEE
    sinL = np.sin(L)
    cosL = np.cos(L)
    s2 = 1.0 + h * h + k * k
    w = 1.0 + f * cosL + g * sinL
    r = p / w

    coeff = mu * J2_coeff * Re * Re / (r**4)
    R_acc = -3.0 * coeff * (1.0 - 12.0 * (h * sinL - k * cosL) ** 2 / (s2 * s2))
    T_acc = -12.0 * coeff * ((h * sinL - k * cosL) * (h * cosL + k * sinL) / (s2 * s2))
    N_acc = -6.0 * coeff * ((1.0 - h * h - k * k) * (h * sinL - k * cosL) / (s2 * s2))
    return np.array([R_acc, T_acc, N_acc])


def drag_acceleration_rtn(MEE, R, V, Cd: float, area_m2: float, mass_kg: float, mu: float = MU) -> np.ndarray:
    """Atmospheric drag acceleration in the RTN frame [km/s^2].

    大气阻力加速度（RTN 系）。The inertial velocity is used, i.e. the rotation
    of the atmosphere is neglected. Inputs use SI for the spacecraft geometry
    and km for the state, matching the original implementation.
    """
    v = np.linalg.norm(V)
    density = atmosphere_density(np.linalg.norm(R) - R_EARTH_MEAN)

    # density kg/m^3 -> kg/km^3 (1e9), area m^2 -> km^2 (1e-6)
    factor = -0.5 * (density * 1e9) * v * (Cd * (area_m2 / 1e6) / mass_kg)

    p, f, g, _, _, L = MEE
    sinL = np.sin(L)
    cosL = np.cos(L)
    Vr = np.sqrt(mu / p) * (f * sinL - g * cosL)
    Vt = np.sqrt(mu / p) * (1.0 + f * cosL + g * sinL)
    return np.array([factor * Vr, factor * Vt, 0.0])
