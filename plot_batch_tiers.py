"""Standalone regeneration of the batch figures from an archived batch run.

The Monte-Carlo results live under::

    batch_archive/<window>/<scenario>/<tier>/batch_evaluation.csv

where ``<window>`` is one of ``full``/``window30``/``window70``,
``<scenario>`` is ``evasive``/``stationkeeping`` and ``<tier>`` is
``ideal``/``high``/``low``.  This script reads those long-format evaluation
tables directly (no config or CLI needed) and renders the ``batch_tiers``
violin/box grid, the ``batch_vs_window`` convergence chart and the two
``batch_heatmap`` summary tables.  See
:mod:`sindy_ssa.experiments.archive_figures` for the figure definitions.

Figures are written as PDF only; ``--png`` adds a raster preview.

Output names follow the convention already used in the paper:

* ``batch_tiers_<scenario>.pdf``         (full manoeuvre window)
* ``batch_tiers_<30|70><scenario>.pdf``     (partial-window prediction)
* ``batch_vs_window_<scenario>.pdf``       (median/IQR convergence line chart)
* ``batch_heatmap.pdf``                    (full-observation table, both scenarios)
* ``batch_heatmap_windows.pdf``            (partial-window table, both scenarios)

Usage::

    python plot_batch_tiers.py
    python plot_batch_tiers.py --png
    python plot_batch_tiers.py --log
    python plot_batch_tiers.py --archive path/to/batch_archive --out path/to/figs

``sindy-ssa batch paper --figures`` produces the same figures at the end of a
full campaign, so this script is only needed to re-render them afterwards.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make ``sindy_ssa`` importable when run as a script without pip-installing it.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sindy_ssa.experiments.archive_figures import plot_archive_figures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--archive",
        default=os.path.join(_HERE, "batch_archive"),
        help="root of the batch archive (default: ./batch_archive)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output directory for the figures (default: <archive>/figs)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="use a logarithmic y-axis on the tier and convergence charts (default: linear)",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="also write a PNG preview next to each PDF (default: PDF only)",
    )
    args = parser.parse_args(argv)

    saved = plot_archive_figures(args.archive, args.out, log_scale=args.log, png=args.png)
    for path in saved:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
