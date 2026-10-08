"""SINDy-based manoeuvre reconstruction methods.

基于 SINDy 的机动重建方法。

- ``MEEFD-SG``      finite-difference inversion + Savitzky-Golay baseline
- ``SINDy``         sparse identification on MEE derivatives
- ``SINDy-TIME``    sparse regression on a time library
- ``SINDy-UMEE``    sparse regression on an MEE library
- ``TIME-OLS``      ordinary least-squares fit on the time library
- ``MEEFD-Spline``  smoothing-spline fit on the MEEFD inversion
"""

from . import preprocessing
from .baselines import run_meefd_spline, run_time_ols
from .libraries import LibrarySpec, original_library, time_library, umee_library
from .meefd import run_meefd_sg
from .regression import fit_ssr, predict, save_matrix
from .sindy_original import run_sindy_original
from .sindy_time import run_sindy_time
from .sindy_umee import run_sindy_umee

__all__ = [
    "LibrarySpec",
    "fit_ssr",
    "original_library",
    "predict",
    "preprocessing",
    "run_meefd_sg",
    "run_meefd_spline",
    "run_sindy_original",
    "run_sindy_time",
    "run_sindy_umee",
    "run_time_ols",
    "save_matrix",
    "time_library",
    "umee_library",
]
