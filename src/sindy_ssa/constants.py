"""Physical constants and unit conventions.

物理常数与单位约定。

Unit conventions follow the original study:
- length: km, time: s, mass: kg
- accelerations: km/s^2 unless a `_si` suffix is used
- atmospheric density: kg/m^3 (converted internally to kg/km^3)
"""

from __future__ import annotations

# Earth gravitational parameter [km^3 / s^2]
MU = 398600.0

# Earth equatorial radius used by the J2 model [km]
R_EARTH_EQ = 6378.0

# Earth mean radius used for altitude / geometry helpers [km]
R_EARTH_MEAN = 6371.0

# Earth rotation rate [rad/s] (one revolution per 24 h, as in the original code)
OMEGA_EARTH = 2.0 * 3.141592653589793 / (24.0 * 3600.0)

# J2 zonal coefficient [-]
J2 = 0.00108263

# Standard gravity [m/s^2]
G0 = 9.81

# Default data column layouts (kept in one place for clarity)
MEE_COLUMNS = ["p", "f", "g", "h", "k", "L"]
MEE_DOT_COLUMNS = ["pdot", "fdot", "gdot", "hdot", "kdot", "Ldot"]
RTN_ACC_COLUMNS = ["Apn_r", "Apn_t", "Apn_n"]
THRUST_COLUMNS = ["u_r", "u_t", "u_n"]
POSITION_COLUMNS = ["R_x", "R_y", "R_z"]
VELOCITY_COLUMNS = ["V_x", "V_y", "V_z"]
