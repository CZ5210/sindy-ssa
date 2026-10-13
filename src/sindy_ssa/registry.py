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
)

# Deterministic noise-free reference case.  It uses the *same* observation
# cadence as the noisy tiers so that comparing it with them isolates the absence
# of noise instead of also giving it a denser sampling grid.
IDEAL_CASE: tuple[tuple[str, str, str | None, int], ...] = (
    ("ideal", "none", None, 6),
)

TYPICAL_CASES: tuple[tuple[str, str, str | None, int], ...] = IDEAL_CASE + NOISE_CASES

# case_name, noise_tier
BATCH_CASES: tuple[tuple[str, str], ...] = (
    ("high", "high"),
    ("low", "low"),
)

# Deterministic noise-free reference case for the batch (single run); plotted as
# a bar / median-only column rather than a violin since it has no distribution.
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

# --------------------------------------------------------------------------- #
# paper batch campaign
# --------------------------------------------------------------------------- #
# The manuscript's batch figures need three campaigns of the same Monte-Carlo
# study: the fully observed manoeuvre plus two partial-window forecasts.  A
# window directory name maps to the observed fraction of the manoeuvre arc used
# for identification (``None`` -> fully observed, prediction disabled).
PAPER_WINDOWS: tuple[tuple[str, float | None], ...] = (
    ("full", None),
    ("window70", 0.70),
    ("window30", 0.30),
)

# Root of the archived Monte-Carlo results consumed by ``plot_batch_tiers.py``
# and produced by ``sindy-ssa batch paper``.
DEFAULT_ARCHIVE_ROOT: str = "batch_archive"


def paper_window(name: str) -> tuple[str, float | None]:
    """Validate a paper-window directory name and return ``(name, fit_fraction)``.

    Accepted names follow the archive layout: ``full`` for the fully observed
    manoeuvre and ``windowNN`` for a partial window observed over the first
    ``NN`` percent of the arc (e.g. ``window70`` -> 0.70).
    """
    for window, fraction in PAPER_WINDOWS:
        if name == window:
            return window, fraction
    if name.startswith("window") and name[6:].isdigit():
        percent = int(name[6:])
        if 0 < percent < 100:
            return name, percent / 100.0
    raise ValueError(
        f"unknown paper window {name!r}; expected 'full' or 'windowNN' "
        f"(e.g. window70), or one of {[w for w, _ in PAPER_WINDOWS]}"
    )


def paper_window_prefix(name: str) -> str:
    """Figure filename prefix of a window (``full`` -> ``''``, ``window70`` -> ``'70'``)."""
    return "" if name == "full" else name[len("window"):]
