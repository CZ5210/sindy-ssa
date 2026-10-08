"""Scenario-level satellite simulators.

场景级卫星仿真器。

``BaseSatellite`` holds the common propagation/IO logic; the two concrete
subclasses implement the station-keeping and evasive scenario control laws.
The LQR gains and thrust limits follow the manuscript.

``BaseSatellite`` 负责通用递推与读写，两个子类分别实现站位保持与规避场景
的控制律。LQR 增益与推力上限与论文一致。
"""

from __future__ import annotations

import csv
import os

import numpy as np

from ..data import DataTable
from ..utils import ensure_dir
from .elements import RV2MEE
from .frames import ComputeLQRControl, CWEquation, Outside_Box
from .propagation import (
    dynamics_perturbed,
    dynamics_reference,
    next_lvlh,
    next_perturbed,
    next_reference,
)
from .state import SatDynamicParam, SatStaticParam, TimeParams

# --------------------------------------------------------------------------- #
# Fixed scaled-LQR controller constants.
#
# Only the Q/R weights (see :class:`sindy_ssa.config.LQRControlConfig`) are
# user-facing; these state/control scalings, the trailing-average window and the
# manoeuvre-termination tolerances are internal implementation details and are
# deliberately not part of the configuration.
# --------------------------------------------------------------------------- #
LQR_POSITION_DIVISOR = 15.0
LQR_VELOCITY_GAINS = (100.0, 1000.0, -10000.0)
LQR_CONTROL_GAIN = 1.0e-8
LQR_WINDOW_SECONDS = 3000.0
LQR_TERMINAL_POS_ERROR_KM = 2.0
LQR_TERMINAL_VEL_ERROR_MS = 1.0
LQR_MANOEUVRE_TIMEOUT_HOURS = None


def scaled_lqr_thrust(error_state, K, max_thrust):
    """Scaled-LQR thrust shared by both scenes.

    The relative error state ``[x, y, z, vx, vy, vz]`` is scaled per axis, mapped
    through the LQR gain ``K``, scaled globally and clipped to ``max_thrust``; the
    out-of-plane component is kept at zero (planar manoeuvres).  The scaling
    constants are fixed at module level (see :data:`LQR_POSITION_DIVISOR` etc.).
    """
    s = np.asarray(error_state, dtype=float).copy()
    s[0:3] = s[0:3] / LQR_POSITION_DIVISOR
    vg = LQR_VELOCITY_GAINS
    s[3] *= vg[0]
    s[4] *= vg[1]
    s[5] *= vg[2]
    u = K @ s * LQR_CONTROL_GAIN
    u = np.clip(u, -max_thrust, max_thrust)
    u[2] = 0.0
    return u


class BaseSatellite:
    """Common satellite state container and propagator.

    通用卫星状态容器与递推器。
    """

    def __init__(self, Sat_Param: SatStaticParam, R0, V0, m):
        self.Param = Sat_Param
        R0 = np.asarray(R0, dtype=float)
        V0 = np.asarray(V0, dtype=float)
        MEE0 = RV2MEE(R0, V0)

        t0 = 0.0
        s0 = [*MEE0, m]
        para = [self.Param.Cd, self.Param.Area, self.Param.c]
        u0 = [0.0, 0.0, 0.0]

        sdot_ref, apn_ref = dynamics_reference(t0, s0, para, u0, if_acc=True)
        sdot_real, apn_real = dynamics_perturbed(t0, s0, para, u0, if_acc=True)

        self.Data_ref = SatDynamicParam(t0, MEE0, sdot_ref[0:6], apn_ref, u0, R0, V0, m)
        self.Data_real = SatDynamicParam(t0, MEE0, sdot_real[0:6], apn_real, u0, R0, V0, m)

        self.Data_ref_list = [self.Data_ref]
        self.Data_real_list = [self.Data_real]

        self.Data_LVLH = next_lvlh(self.Data_real, self.Data_ref)
        self.Data_LVLH_list = [self.Data_LVLH]

        self.Flag_list: list = []

        # Manoeuvre bookkeeping / 机动统计
        self.Manoeuvre = type("Manoeuvre", (), {})()
        self.Manoeuvre.freq = 0
        self.Manoeuvre.start_time = []
        self.Manoeuvre.TOF = []
        self.Manoeuvre.DV = []

    # -- propagation -------------------------------------------------------
    def update(self, time_param: TimeParams, u_lvlh):
        """Advance both satellites by one step and record the state."""
        data_ref_new = next_reference(time_param.dt, self.Param, self.Data_ref)
        data_real_new = next_perturbed(time_param.dt, self.Param, self.Data_real, u_lvlh)
        # NOTE: as in the original code, the relative state is derived from the
        # *previous* truth/reference states (one-step alignment quirk).
        data_lvlh_new = next_lvlh(self.Data_real, self.Data_ref)

        self.Data_ref, self.Data_real, self.Data_LVLH = data_ref_new, data_real_new, data_lvlh_new
        self.Data_ref_list.append(data_ref_new)
        self.Data_real_list.append(data_real_new)
        self.Data_LVLH_list.append(data_lvlh_new)
        self.Flag_list.append(self.Flag)

    def check_conditions(self, t) -> bool:
        alt = np.linalg.norm(self.Data_real.R) - 6371.0
        return alt <= 0 or self.Data_real.m < self.Param.m0

    # -- control helpers ---------------------------------------------------
    def _window_indices(self, time_param: TimeParams, window_seconds: float):
        """Return ``(i_max, step)`` bounding the trailing-window state average."""
        dt = time_param.dt
        i_theo = int(np.floor(window_seconds / dt)) if dt > 0 else 0
        i_max = min(i_theo, len(self.Data_LVLH_list) - 1)
        step = max(1, i_max // 200) if i_max > 200 else 1
        return i_max, step

    def windowed_lvlh_state(self, time_param: TimeParams, window_seconds: float, target_pos=None):
        """Mean ``[R, V]`` over the trailing ``window_seconds`` of stored states.

        When ``target_pos`` is given (a callable returning the RTN target position
        for a stored-sample index ``k``), the average is taken over the position
        *error* ``R - target`` and the raw velocity; otherwise the raw relative
        position is averaged (the station-keeping case, whose target is the origin).
        """
        nds = np.zeros(6)
        count = 0
        i_max, step = self._window_indices(time_param, window_seconds)
        for i in range(0, i_max + 1, step):
            k = len(self.Data_LVLH_list) - i - 1
            hist = self.Data_LVLH_list[k]
            pos = hist.R if target_pos is None else hist.R - target_pos(k)
            nds[:3] += pos
            nds[3:] += hist.V
            count += 1
        if count > 0:
            nds /= count
        return nds

    # -- re-propagation with reconstructed thrust --------------------------
    def repropagate(self, thrust_path: str, real_path: str, ref_path: str, dt: float, zero_normal: bool = True):
        """Re-propagate the truth orbit using a reconstructed thrust history."""
        reconstructed = DataTable.from_csv(thrust_path)
        u_r_list = np.array(reconstructed.get("u_r").data, dtype=float)
        u_t_list = np.array(reconstructed.get("u_t").data, dtype=float)
        u_n_list = np.array(reconstructed.get("u_n").data, dtype=float)
        time0 = float(reconstructed.get("Time").data[0])
        thrust_steps = len(u_r_list)

        # initial states aligned with the reconstructed thrust start time
        real_full = DataTable.from_csv(real_path)
        ref_full = DataTable.from_csv(ref_path)
        idx_real = int(np.argmin(np.abs(real_full.get("Time").data - time0)))
        idx_ref = int(np.argmin(np.abs(ref_full.get("Time").data - time0)))

        def pick(table, idx, names):
            return np.array([table.get(n).data[idx] for n in names], dtype=float)

        self.Data_real = SatDynamicParam(
            t=time0,
            MEE=pick(real_full, idx_real, ["p", "f", "g", "h", "k", "L"]),
            MEEdot=pick(real_full, idx_real, ["pdot", "fdot", "gdot", "hdot", "kdot", "Ldot"]),
            Apn=pick(real_full, idx_real, ["Apn_r", "Apn_t", "Apn_n"]),
            U_LVLH=pick(real_full, idx_real, ["u_r", "u_t", "u_n"]),
            R=pick(real_full, idx_real, ["R_x", "R_y", "R_z"]),
            V=pick(real_full, idx_real, ["V_x", "V_y", "V_z"]),
            m=float(real_full.get("Mass").data[idx_real]),
        )
        self.Data_ref = SatDynamicParam(
            t=time0,
            MEE=pick(ref_full, idx_ref, ["p", "f", "g", "h", "k", "L"]),
            MEEdot=pick(ref_full, idx_ref, ["pdot", "fdot", "gdot", "hdot", "kdot", "Ldot"]),
            Apn=pick(ref_full, idx_ref, ["Apn_r", "Apn_t", "Apn_n"]),
            U_LVLH=np.zeros(3),
            R=pick(ref_full, idx_ref, ["R_x", "R_y", "R_z"]),
            V=pick(ref_full, idx_ref, ["V_x", "V_y", "V_z"]),
            m=float(real_full.get("Mass").data[idx_real]),
        )
        self.Data_LVLH = next_lvlh(self.Data_real, self.Data_ref)

        self.Data_reprop_real_list = [self.Data_real]
        self.Data_reprop_ref_list = [self.Data_ref]
        self.Data_reprop_LVLH_list = [self.Data_LVLH]

        for step in range(thrust_steps - 1):
            u_lvlh = [u_r_list[step], u_t_list[step], 0.0 if zero_normal else u_n_list[step]]
            self.Data_ref = next_reference(dt, self.Param, self.Data_ref)
            self.Data_real = next_perturbed(dt, self.Param, self.Data_real, u_lvlh)
            self.Data_LVLH = next_lvlh(self.Data_real, self.Data_ref)
            self.Data_reprop_real_list.append(self.Data_real)
            self.Data_reprop_ref_list.append(self.Data_ref)
            self.Data_reprop_LVLH_list.append(self.Data_LVLH)

    # -- IO ----------------------------------------------------------------
    def save_csv(self, output_dir: str, precision: int = 8):
        ensure_dir(output_dir)
        fmt = f"{{:.{precision}e}}"

        real_headers = [
            "Time", "p", "f", "g", "h", "k", "L",
            "pdot", "fdot", "gdot", "hdot", "kdot", "Ldot",
            "Apn_r", "Apn_t", "Apn_n",
            "R_x", "R_y", "R_z", "V_x", "V_y", "V_z",
            "Mass", "u_r", "u_t", "u_n", "Flag",
        ]
        ref_headers = real_headers[:-5]
        lvlh_headers = ["Time", "R_x", "R_y", "R_z", "V_x", "V_y", "V_z", "u_x", "u_y", "u_z", "Flag"]

        with open(os.path.join(output_dir, "Data_real.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(real_headers)
            for idx, data in enumerate(self.Data_real_list):
                flag = self.Flag_list[idx] if idx < len(self.Flag_list) else "unknown"
                writer.writerow([
                    data.t,
                    *[fmt.format(v) for v in data.MEE],
                    *[fmt.format(v) for v in data.MEEdot],
                    *[fmt.format(v) for v in data.Apn],
                    *[fmt.format(v) for v in data.R],
                    *[fmt.format(v) for v in data.V],
                    fmt.format(data.m),
                    *[fmt.format(v) for v in data.U_LVLH],
                    flag,
                ])

        with open(os.path.join(output_dir, "Data_ref.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(ref_headers)
            for data in self.Data_ref_list:
                writer.writerow([
                    data.t,
                    *[fmt.format(v) for v in data.MEE],
                    *[fmt.format(v) for v in data.MEEdot],
                    *[fmt.format(v) for v in data.Apn],
                    *[fmt.format(v) for v in data.R],
                    *[fmt.format(v) for v in data.V],
                ])

        with open(os.path.join(output_dir, "Data_LVLH.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(lvlh_headers)
            for idx, lvlh in enumerate(self.Data_LVLH_list):
                flag = self.Flag_list[idx] if idx < len(self.Flag_list) else "unknown"
                writer.writerow([
                    self.Data_ref_list[idx].t,
                    *[fmt.format(v) for v in lvlh.R],
                    *[fmt.format(v) for v in lvlh.V],
                    *[fmt.format(v) for v in lvlh.U],
                    flag,
                ])

        with open(os.path.join(output_dir, "Data_mano.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Frequency", "StartTime", "TOF", "DV"])
            for freq, (start, tof, dv) in enumerate(
                zip(self.Manoeuvre.start_time, self.Manoeuvre.TOF, self.Manoeuvre.DV), start=1
            ):
                writer.writerow([freq, fmt.format(start), fmt.format(tof), fmt.format(dv)])

    def save_repropagate_csv(self, output_dir: str, precision: int = 8, suffix: str = ""):
        ensure_dir(output_dir)
        fmt = f"{{:.{precision}e}}"
        real_headers = [
            "Time", "p", "f", "g", "h", "k", "L",
            "pdot", "fdot", "gdot", "hdot", "kdot", "Ldot",
            "Apn_r", "Apn_t", "Apn_n",
            "R_x", "R_y", "R_z", "V_x", "V_y", "V_z", "Mass", "u_r", "u_t", "u_n",
        ]
        lvlh_headers = ["Time", "R_x", "R_y", "R_z", "V_x", "V_y", "V_z", "u_x", "u_y", "u_z"]

        with open(os.path.join(output_dir, f"Data_rereal{suffix}.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(real_headers)
            for data in self.Data_reprop_real_list:
                writer.writerow([
                    data.t,
                    *[fmt.format(v) for v in data.MEE],
                    *[fmt.format(v) for v in data.MEEdot],
                    *[fmt.format(v) for v in data.Apn],
                    *[fmt.format(v) for v in data.R],
                    *[fmt.format(v) for v in data.V],
                    fmt.format(data.m),
                    *[fmt.format(v) for v in data.U_LVLH],
                ])

        with open(os.path.join(output_dir, f"Data_reLVLH{suffix}.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(lvlh_headers)
            for idx, lvlh in enumerate(self.Data_reprop_LVLH_list):
                writer.writerow([
                    self.Data_reprop_real_list[idx].t,
                    *[fmt.format(v) for v in lvlh.R],
                    *[fmt.format(v) for v in lvlh.V],
                    *[fmt.format(v) for v in lvlh.U],
                ])

    def propagate(self, time_param: TimeParams):
        raise NotImplementedError


class SatelliteStationKeeping(BaseSatellite):
    """Drag-compensating station-keeping scenario. / 站位保持场景。"""

    def __init__(self, Sat_Param, R0, V0, m, BoxSize, control_config=None):
        super().__init__(Sat_Param, R0, V0, m)
        self.BoxSize = BoxSize
        self.Flag = "normal"
        self.Flag_list = [self.Flag]
        from ..config import LQRControlConfig
        self.control_config = control_config or LQRControlConfig()
        self.terminalpos_error = LQR_TERMINAL_POS_ERROR_KM
        self.terminalvel_error = LQR_TERMINAL_VEL_ERROR_MS

    def _normal(self, t, time_param):
        self.update(time_param, [0.0, 0.0, 0.0])
        if Outside_Box(self.Data_LVLH.R, self.BoxSize, shift=0):
            self.Flag = "stationkeeping"

    def _generate_manoeuvre_lqr(self):
        self.terminalpos_error = LQR_TERMINAL_POS_ERROR_KM
        self.terminalvel_error = LQR_TERMINAL_VEL_ERROR_MS
        r = np.linalg.norm(self.Data_real.R)
        A, B = CWEquation(r)
        Q = np.diag(self.control_config.q_diag)
        R = np.diag(self.control_config.r_diag)
        self.K1 = ComputeLQRControl(A, B, Q, R)
        self.Flag = "manoeuvre"
        self.Manoeuvre.start_time.append(self.Data_real.t)
        self.thisDV = 0.0

    def _manoeuvre_lqr(self, t, time_param):
        S_LVLH = self.Data_LVLH_list[-1]
        s = np.array([*S_LVLH.R, *S_LVLH.V])
        pos_error = np.linalg.norm(s[0:3])
        vel_error = np.linalg.norm(s[3:6])

        # windowed mean of the recent relative state (as in the original code)
        nds = self.windowed_lvlh_state(time_param, LQR_WINDOW_SECONDS)

        max_thrust = self.Param.Tmax / self.Param.m / 1000.0  # km/s^2
        u_lvlh = scaled_lqr_thrust(nds, self.K1, max_thrust)
        self.thisDV += np.linalg.norm(u_lvlh) * time_param.dt * 1000.0

        self.update(time_param, u_lvlh)

        timeout_reached = False
        if LQR_MANOEUVRE_TIMEOUT_HOURS is not None:
            elapsed = t - self.Manoeuvre.start_time[-1]
            if elapsed >= LQR_MANOEUVRE_TIMEOUT_HOURS * 3600.0:
                timeout_reached = True

        if (pos_error < self.terminalpos_error and vel_error * 1000.0 < self.terminalvel_error) or timeout_reached:
            self.Flag = "normal"
            self.Manoeuvre.freq += 1
            self.Manoeuvre.TOF.append(t - self.Manoeuvre.start_time[-1])
            self.Manoeuvre.DV.append(self.thisDV)

    def propagate(self, time_param: TimeParams):
        for _ in range(time_param.n + 1):
            t = self.Data_ref.t
            if self.check_conditions(t):
                break
            if self.Flag == "normal":
                self._normal(t, time_param)
            if self.Flag == "stationkeeping":
                self._generate_manoeuvre_lqr()
            if self.Flag == "manoeuvre":
                self._manoeuvre_lqr(t, time_param)


class SatelliteCollisionAvoidance(BaseSatellite):
    """Emergency evasive manoeuvre scenario. / 紧急规避场景。"""

    def __init__(self, Sat_Param, R0, V0, m, TargetPos, TOF, control_config=None):
        super().__init__(Sat_Param, R0, V0, m)
        self.TargetPos = np.asarray(TargetPos, dtype=float)
        self.TOF = TOF
        self.Flag = "ConjunctionDetected"
        # When given, the evasive scene mirrors the station-keeping scaled-LQR law
        # (Q/R + position_divisor/velocity_gains/control_gain) instead of the
        # native bang-bang controller; ``None`` keeps the original behaviour.
        self.control_config = control_config

    def _normal(self, t, time_param):
        self.update(time_param, [0.0, 0.0, 0.0])

    def _generate_reference(self, time_param):
        n = 2
        a = abs(self.TargetPos[1]) / n
        b = 4.0
        length = int(3600.0 * self.TOF // time_param.dt)
        x = np.zeros(length)
        y = np.zeros(length)
        z = np.zeros(length)
        for i in range(n):
            start = i * (length // n)
            end = (i + 1) * (length // n)
            theta = np.linspace(0.0, np.pi, end - start)
            x[start:end] = (b / 2.0) * np.sin(theta)
            y[start:end] = (a / 2.0) * np.cos(theta) - (a / 2.0) - (a * i)
        self.RefPos = np.vstack([x, y, z]).T
        reference_dir = getattr(self, "reference_dir", ".")
        DataTable.from_arrays(["x", "y", "z"], [x, y, z]).save(
            os.path.join(reference_dir, "ReferenceTrajectory.csv"), precision=8
        )

    def _ref_pos_at(self, k):
        """RTN target position for stored-sample index ``k`` (0-based)."""
        n = len(self.RefPos)
        return self.RefPos[min(max(k - 1, 0), n - 1)]

    def _generate_manoeuvre_lqr(self, time_param):
        self._generate_reference(time_param)
        self.terminalpos_error = 1.0
        self.terminalvel_error = 30.0
        r = np.linalg.norm(self.Data_real.R)
        A, B = CWEquation(r)
        if self.control_config is not None:
            Q = np.diag(self.control_config.q_diag)
            R = np.diag(self.control_config.r_diag)
        else:
            Q = np.diag([1e2, 1e2, 1.0, 1e3, 5e7, 1.0])
            R = np.diag([2.0, 2.0, 2.0])
        self.K1 = ComputeLQRControl(A, B, Q, R)
        self.Flag = "manoeuvre"
        self.Manoeuvre.start_time.append(self.Data_real.t)
        self.thisDV = 0.0

    def _manoeuvre_lqr(self, t, time_param):
        max_thrust = self.Param.Tmax / self.Param.m / 1000.0  # km/s^2

        if self.control_config is not None:
            # Mirror the station-keeping law: regulate [R - target, V] to zero.
            nds = self.windowed_lvlh_state(time_param, LQR_WINDOW_SECONDS, target_pos=self._ref_pos_at)
            u_lvlh = scaled_lqr_thrust(nds, self.K1, max_thrust)
        else:
            idx = int(t / time_param.dt)
            ref_pos = self.RefPos[idx - 1]
            S_LVLH = self.Data_LVLH_list[-1]
            s = np.array([*ref_pos - S_LVLH.R, *S_LVLH.V])
            u_raw = -self.K1 @ s
            theta = np.arctan2(u_raw[0], u_raw[1])
            u_lvlh = np.array([max_thrust * np.sin(theta), max_thrust * np.cos(theta), 0.0])

        self.thisDV += np.linalg.norm(u_lvlh) * time_param.dt * 1000.0

        self.update(time_param, u_lvlh)

        if t >= self.TOF * 3600.0:
            self.Flag = "normal"
            self.Manoeuvre.freq += 1
            self.Manoeuvre.TOF.append(t - self.Manoeuvre.start_time[-1])
            self.Manoeuvre.DV.append(self.thisDV)

    def propagate(self, time_param: TimeParams):
        for _ in range(time_param.n + 1):
            t = self.Data_ref.t
            if self.check_conditions(t):
                break
            if self.Flag == "normal":
                self._normal(t, time_param)
            if self.Flag == "ConjunctionDetected":
                self._generate_manoeuvre_lqr(time_param)
            if self.Flag == "manoeuvre":
                self._manoeuvre_lqr(t, time_param)
