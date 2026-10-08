"""Global registry for scenarios, cases, methods, and study defaults.

This module is the single source of truth for scenario/case lists that used to be
copied in several entry scripts.  It intentionally has no dependencies on the rest
of the package so that ``cli.py`` can import it safely.
"""

from __future__ import annotations

# Single unified configuration file holding both scenarios under ``scenarios:``.
DEFAULT_CONFIG: str = "configs/config.yaml"

# Dedicated configuration for the Monte-Carlo batch study.  The ``batch run`` /
# ``batch plot`` subcommands default to this file so future prediction stays off
# and the ``batch:`` block controls runs/seed/cases.  Override with ``--config``.
DEFAULT_BATCH_CONFIG: str = "configs/batch_run.yaml"

# case_name, noise_mode, noise_tier, sampling_rate
NOISE_CASES: tuple[tuple[str, str, str, int], ...] = (
    ("high", "gaussian", "high", 6),
    ("low", "gaussian", "low", 6),
    ("reduced", "gaussian", "reduced", 6),
)

IDEAL_CASE: tuple[tuple[str, str, str | None, int], ...] = (
    ("ideal", "none", None, 1),
)

TYPICAL_CASES: tuple[tuple[str, str, str | None, int], ...] = IDEAL_CASE + NOISE_CASES

# case_name, noise_tier
BATCH_CASES: tuple[tuple[str, str], ...] = (
    ("high", "high"),
    ("low", "low"),
    ("reduced", "reduced"),
)

# Deterministic noise-free reference case for the batch (single run); plotted as
# a marker rather than a violin since it has no distribution.
BATCH_IDEAL_CASE: str = "ideal"

# method label -> reconstructed-thrust CSV filename (inside methods/<label>/)
# Non-predictive methods (MEEFD-SG, MEEFD-Spline) are listed first so that in
# partial-window mode the thrust figure shows their horizon-only curves first.
METHODS: dict[str, str] = {
    "MEEFD-SG": "U_MEEFD_SG.csv",
    "MEEFD-Spline": "U_MEEFD_Spline.csv",
    "TIME-OLS": "U_TIME_OLS.csv",
    "SINDy": "U_SINDy.csv",
    "SINDy-TIME": "U_SINDyTIME.csv",
    "SINDy-UMEE": "U_SINDyUMEE.csv",
}

DEFAULT_BATCH_RUNS: int = 100
DEFAULT_SEED0: int = 0
