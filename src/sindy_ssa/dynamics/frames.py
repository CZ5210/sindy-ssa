"""Frame transforms and linearised relative-motion helpers.

坐标系变换与线性化相对运动工具。
"""

from __future__ import annotations

import numpy as np

from ..constants import MU, R_EARTH_MEAN


def Outside_Box(R_LVLH, BoxSize, shift: float) -> bool:
    """Return ``True`` when the relative position leaves the control box."""
    sx, sy, sz = BoxSize[0], BoxSize[1], BoxSize[2]
    x, y, z = R_LVLH
    return bool(abs(x) > sx / 2.0 or abs(y + shift) > sy / 2.0 or abs(z) > sz / 2.0)


def ComputeLQRControl(A, B, Q, R):
    """Solve the continuous-time algebraic Riccati equation for gain ``K``."""
    import scipy.linalg as la

    P = la.solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P


def CWEquation(r: float):
    """Clohessy-Wiltshire state-space matrices for radius ``r`` [km]."""
    omega = np.sqrt(MU / r**3)

    A = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [3.0 * omega**2, 0.0, 0.0, 0.0, 2.0 * omega, 0.0],
            [0.0, 0.0, 0.0, -2.0 * omega, 0.0, 0.0],
            [0.0, 0.0, -omega**2, 0.0, 0.0, 0.0],
        ]
    )
    B = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return A, B


def _rtn_basis(R, V):
    i_r = R / np.linalg.norm(R)
    i_n = np.cross(R, V) / np.linalg.norm(np.cross(R, V))
    i_t = np.cross(i_n, i_r) / np.linalg.norm(np.cross(i_n, i_r))
    return i_r, i_t, i_n


def ECI2LVLH(R, V, Vec):
    """Rotate an ECI vector into the RTN/LVLH frame."""
    i_r, i_t, i_n = _rtn_basis(R, V)
    rotation = np.array([i_r, i_t, i_n])
    return np.dot(rotation, Vec).flatten()


def LVLH2ECI(R, V, U_LVLH):
    """Rotate an RTN/LVLH vector into the ECI frame."""
    i_r, i_t, i_n = _rtn_basis(R, V)
    return U_LVLH[0] * i_r + U_LVLH[1] * i_t + U_LVLH[2] * i_n


def InitOrbit(alt: float = 500.0, inc: float = 0.0):
    """Return the initial ECI position/velocity for a circular orbit."""
    inc = np.deg2rad(inc)
    radius = alt + R_EARTH_MEAN
    R = np.array([radius, 0.0, 0.0])
    v = np.sqrt(MU / radius)
    V = np.array([0.0, v * np.cos(inc), v * np.sin(inc)])
    return R, V
