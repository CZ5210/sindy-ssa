"""``python -m sindy_ssa.noise`` 的入口。"""

from __future__ import annotations

import argparse

from .calibration import DEFAULT_ANALYSIS90_DIR, run_noise_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Starlink RTN noise calibration (CSV only)")
    parser.add_argument("--input-dir", default=None, help="directory containing MEME/ITC *.txt files")
    parser.add_argument("--output-dir", default=None, help="directory for calibration CSVs")
    parser.add_argument("--n-samples", type=int, default=5000, help="Monte Carlo samples per tier")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    args = parser.parse_args(argv)
    run_noise_study(
        input_dir=args.input_dir or DEFAULT_ANALYSIS90_DIR,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
