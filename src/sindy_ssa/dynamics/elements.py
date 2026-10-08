"""Modified equinoctial elements (MEE) and their variational equations.

修正春分点根数（MEE）及其摄动方程。

References / 参考:
- Walker, Ireland & Owens (1985), "A set of modified equinoctial orbit elements".
"""

from __future__ import annotations

import numpy as np

from ..constants import MU


def RV2MEE(R, V, mu: float = MU) -> np.ndarray:
    """Convert an inertial position/velocity pair to MEE.

    将惯性系位置/速度转换到 MEE。
    """
    R = np.asarray(R, dtype=float)
    V = np.asarray(V, dtype=float)
    r = np.linalg.norm(R)
    v = np.linalg.norm(V)

    h_vec = np.cross(R, V)
    h = np.linalg.norm(h_vec)
    i = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    e_vec = (np.cross(V, h_vec) / mu) - R / r
    e = np.linalg.norm(e_vec)

    energy = v**2 / 2.0 - mu / r
    a = -mu / (2.0 * energy)
    p = a * (1.0 - e**2)

    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec)

    if n != 0:
        raan = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
        if n_vec[1] < 0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    if e > 1e-10 and n != 0:
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[2] < 0:
            omega = 2.0 * np.pi - omega
    else:
        omega = 0.0

    if e > 1e-10:
        nu = np.arccos(np.clip(np.dot(e_vec, R) / (e * r), -1.0, 1.0))
        if np.dot(R, V) < 0:
            nu = 2.0 * np.pi - nu
    else:
        nu = 0.0

    L = (raan + omega + nu) % (2.0 * np.pi)
    f = e * np.cos(raan + omega)
    g = e * np.sin(raan + omega)
    h_mee = np.tan(i / 2.0) * np.cos(raan)
    k_mee = np.tan(i / 2.0) * np.sin(raan)

    return np.array([p, f, g, h_mee, k_mee, L])


def MEE2RV(MEE, mu: float = MU):
    """Convert MEE to an inertial position/velocity pair.

    将 MEE 转换到惯性系位置/速度。
    """
    p, f, g, h, k, L = MEE
    sinL = np.sin(L)
    cosL = np.cos(L)

    alpha = h * h - k * k
    s2 = 1.0 + h * h + k * k
    w = 1.0 + f * cosL + g * sinL
    r = p / w
    sqrt = np.sqrt(mu / p)

    Rx = (r / s2) * (cosL + alpha * cosL + 2.0 * h * k * sinL)
    Ry = (r / s2) * (sinL - alpha * sinL + 2.0 * h * k * cosL)
    Rz = (2.0 * r / s2) * (h * sinL - k * cosL)

    Vx = (-1.0 / s2) * sqrt * (sinL + alpha * sinL - 2.0 * h * k * cosL + g - 2.0 * f * h * k + alpha * g)
    Vy = (-1.0 / s2) * sqrt * (-cosL + alpha * cosL + 2.0 * h * k * sinL - f + 2.0 * g * h * k + alpha * f)
    Vz = (2.0 / s2) * sqrt * (h * cosL + k * sinL + f * h + g * k)

    return np.array([Rx, Ry, Rz]), np.array([Vx, Vy, Vz])


def MEE_Dynamics(MEE, Acc, mu: float = MU):
    """MEE variational equations driven by an RTN acceleration.

    由 RTN 加速度驱动的 MEE 摄动方程。The non-planar ``gdot`` term uses
    ``f`` as in the standard (errata-corrected) Walker et al. form.
    """
    p, f, g, h, k, L = MEE
    Pr, Pt, Pn = Acc

    sinL = np.sin(L)
    cosL = np.cos(L)
    w = 1.0 + f * cosL + g * sinL
    sqrt = np.sqrt(p / mu)
    s2 = 1.0 + h * h + k * k

    pdot = 2.0 * p / w * sqrt * Pt
    fdot = sqrt * (Pr * sinL + ((w + 1.0) * cosL + f) * Pt / w - (h * sinL - k * cosL) * g * Pn / w)
    gdot = sqrt * (-Pr * cosL + ((w + 1.0) * sinL + g) * Pt / w + (h * sinL - k * cosL) * f * Pn / w)
    hdot = sqrt * s2 * Pn * cosL / (2.0 * w)
    kdot = sqrt * s2 * Pn * sinL / (2.0 * w)
    Ldot = np.sqrt(mu * p) * (w / p) ** 2 + sqrt * (h * sinL - k * cosL) * Pn / w

    return np.array([pdot, fdot, gdot, hdot, kdot, Ldot])


def MEEdot2Acc(MEE, MEEdot, weights=None, mu: float = MU) -> np.ndarray:
    """反解摄动方程 ``A(MEE) P = MEEdot - b`` 得到 RTN 加速度。

    若 ``weights`` 为 None，使用普通最小二乘（``np.linalg.lstsq``）；否则按对角
    加权矩阵 ``W = diag(weights)`` 解加权最小二乘：

        P = (A^T W A)^{-1} A^T W (MEEdot - b)

    权重通常取 ``1 / sigma^2``（论文 eq:InverseMEE 的 ``W_MEE``）。
    """
    p, f, g, h, k, L = MEE
    sinL = np.sin(L)
    cosL = np.cos(L)
    w = 1.0 + f * cosL + g * sinL
    sqrt = np.sqrt(p / mu)
    s = np.sqrt(1.0 + h * h + k * k)

    A = np.array(
        [
            [0.0, (2.0 * p / w) * sqrt, 0.0],
            [sqrt * sinL, sqrt / w * ((w + 1.0) * cosL + f), -sqrt * g / w * (h * sinL - k * cosL)],
            [-sqrt * cosL, sqrt / w * ((w + 1.0) * sinL + g), sqrt * f / w * (h * sinL - k * cosL)],
            [0.0, 0.0, sqrt * s * s * cosL / w / 2.0],
            [0.0, 0.0, sqrt * s * s * sinL / w / 2.0],
            [0.0, 0.0, sqrt * (h * sinL - k * cosL) / w],
        ],
        dtype=float,
    )
    b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(mu * p) * (w * w / p / p)])
    rhs = np.asarray(MEEdot, dtype=float) - b
    if weights is None:
        solution = np.linalg.lstsq(A, rhs, rcond=None)[0]
    else:
        w = np.asarray(weights, dtype=float)
        solution = np.linalg.solve(A.T @ np.diag(w) @ A, A.T @ (w * rhs))
    return solution.flatten()
