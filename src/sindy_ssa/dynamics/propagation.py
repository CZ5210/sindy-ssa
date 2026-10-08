"""RK4 propagation of the truth (perturbed) and reference dynamics.

真值（含摄动）与参考动力学的 RK4 递推。
"""

from __future__ import annotations

import numpy as np

from .elements import MEE2RV, MEE_Dynamics
from .frames import ECI2LVLH
from .perturbations import drag_acceleration_rtn, j2_acceleration_rtn
from .state import SatDynamicParam, SatLVLHParam, SatStaticParam


def dynamics_reference(t, s, para, u, if_acc: bool = False):
    """Reference dynamics: central gravity + J2 only. / 参考动力学。"""
    p, f, g, h, k, L, _ = s
    MEE = (p, f, g, h, k, L)

    apn = j2_acceleration_rtn(MEE)
    meedot = MEE_Dynamics(MEE, apn)
    sdot = np.array([*meedot, 0.0])

    return (sdot, apn) if if_acc else sdot


def dynamics_perturbed(t, s, para, u, if_acc: bool = False):
    """Truth dynamics: gravity + J2 + drag + thrust. / 真值动力学。"""
    Cd, area, exhaust_velocity = para
    p, f, g, h, k, L, m = s
    MEE = (p, f, g, h, k, L)
    R, V = MEE2RV(MEE)

    a_j2 = j2_acceleration_rtn(MEE)
    a_drag = drag_acceleration_rtn(MEE, R, V, Cd, area, m)
    a_thrust = np.asarray(u, dtype=float)

    apn = a_j2 + a_drag
    meedot = MEE_Dynamics(MEE, apn + a_thrust)
    mdot = -m * np.linalg.norm(a_thrust) * 1000.0 / exhaust_velocity
    sdot = np.array([*meedot, mdot])

    return (sdot, apn) if if_acc else sdot


def RK4_Step(dynamics, dt, t, s, para, u):
    """One classical fourth-order Runge-Kutta step. / 单步 RK4。"""
    k1 = dynamics(t, s, para, u)
    k2 = dynamics(t + dt / 2.0, s + k1 * dt / 2.0, para, u)
    k3 = dynamics(t + dt / 2.0, s + k2 * dt / 2.0, para, u)
    k4 = dynamics(t + dt, s + k3 * dt, para, u)
    return s + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0


def _params(sat_param: SatStaticParam):
    return [sat_param.Cd, sat_param.Area, sat_param.c]


def next_reference(dt, sat_param: SatStaticParam, data_ref: SatDynamicParam) -> SatDynamicParam:
    """Advance the reference satellite by one step. / 参考卫星推进一步。"""
    t, MEE, m = data_ref.t, data_ref.MEE, data_ref.m
    para = _params(sat_param)
    s = [*MEE, m]
    u = [0.0, 0.0, 0.0]

    s_next = RK4_Step(dynamics_reference, dt, t, s, para, u)
    s_next[5] %= 2.0 * np.pi

    MEE_next = np.array(s_next[0:6])
    m_next = s_next[6]
    R, V = MEE2RV(MEE_next)
    sdot, apn = dynamics_reference(t + dt, s_next, para, u, if_acc=True)
    return SatDynamicParam(t + dt, MEE_next, sdot[0:6], apn, u, R, V, m_next)


def next_perturbed(dt, sat_param: SatStaticParam, data_real: SatDynamicParam, u) -> SatDynamicParam:
    """Advance the truth satellite by one step. / 真值卫星推进一步。"""
    t, MEE, m = data_real.t, data_real.MEE, data_real.m
    para = _params(sat_param)
    s = [*MEE, m]

    s_next = RK4_Step(dynamics_perturbed, dt, t, s, para, u)
    s_next[5] %= 2.0 * np.pi

    MEE_next = np.array(s_next[0:6])
    m_next = s_next[6]
    R, V = MEE2RV(MEE_next)
    sdot, apn = dynamics_perturbed(t + dt, s_next, para, u, if_acc=True)
    return SatDynamicParam(t + dt, MEE_next, sdot[0:6], apn, u, R, V, m_next)


def next_lvlh(data_real: SatDynamicParam, data_ref: SatDynamicParam) -> SatLVLHParam:
    """Relative state of the truth satellite w.r.t. the reference, in RTN."""
    R_ecI = data_real.R
    V_eci = data_real.V
    R_drift = R_ecI - data_ref.R
    V_drift = V_eci - data_ref.V

    R_lvlh = ECI2LVLH(data_ref.R, data_ref.V, R_drift)
    V_lvlh = ECI2LVLH(data_ref.R, data_ref.V, V_drift)
    return SatLVLHParam(R_lvlh, V_lvlh, data_real.U_LVLH)
