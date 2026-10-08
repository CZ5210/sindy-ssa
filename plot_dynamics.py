"""Standalone figure (re)generation entry for the Tavotto figure editor.

Tavotto imports a *top-level* script and calls ``main()`` to re-render the
panels it manages.  This thin wrapper imports the installed ``sindy_ssa``
package with absolute imports (so it can be imported standalone) and
regenerates the truth-dynamics trajectory/thrust figures and the global noise
calibration figures.  It matches the ``plot_dynamics.py`` entry in
``tavotto_registry.json``.

Usage::

    python plot_dynamics.py
"""

from __future__ import annotations

import os
import sys

# Make ``sindy_ssa`` importable when this file is imported/run as a script
# without the package being pip-installed.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sindy_ssa.config import load_scenarios
from sindy_ssa.experiments import plot_scenario_truth
from sindy_ssa.registry import DEFAULT_CONFIG
from sindy_ssa.studies import noise


def main(argv=None) -> int:
    """Regenerate the truth-dynamics and noise figures from the current config."""
    config_path = DEFAULT_CONFIG if not argv else argv[0]
    scenarios = load_scenarios(config_path)

    for name, cfg in scenarios.items():
        cfg.paths.data_dir = os.path.join(cfg.paths.data_root, name, "truth")
        cfg.paths.figures_dir = os.path.join(cfg.paths.figures_root, name, "truth")
        plot_scenario_truth(cfg)

    reference = next(iter(scenarios.values()))
    noise.plot(
        data_dir=reference.paths.noise_data_dir,
        figures_dir=reference.paths.noise_figures_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
