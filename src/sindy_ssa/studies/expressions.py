"""Export identified sparse expressions from Xi.csv to readable text files.

Used by ``sindy-ssa sindy plot``: for a single typical run it writes
``IdentifiedExpressions_<Method>.txt`` per method plus a combined
``IdentifiedExpressions.txt`` into the run's figures directory.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ..utils import ensure_dir

_COMPONENTS = ["u_r", "u_t", "u_n"]


def _format_coeff(x: float) -> str:
    if abs(x) < 1e-12:
        return "0"
    return f"{x:+.6e}"


def _build_expression(xi: pd.DataFrame, col: str) -> str:
    """Format one column of Xi as a readable linear combination."""
    terms = []
    for term_name, value in xi[col].items():
        if abs(float(value)) < 1e-12:
            continue
        display_name = str(term_name).replace("**", "^")
        terms.append(f"{_format_coeff(float(value))} * {display_name}")
    if not terms:
        return "0"
    return " ".join(terms)


def _read_normalization(raw_thrust_path: str | None) -> dict | None:
    """Reproduce the normalization coefficients used by prepare_training_input."""
    if raw_thrust_path is None or not os.path.exists(raw_thrust_path):
        return None
    df = pd.read_csv(raw_thrust_path)
    time = np.asarray(df["Time"], dtype=float)
    time0 = float(time[0])
    coeff_time = float(np.max(time - time0)) if (time.size > 0 and np.max(time - time0) > 0) else 1.0
    coeffs = {"time0_s": time0, "coeff_time_s": coeff_time}
    for comp in _COMPONENTS:
        data = np.asarray(df[comp], dtype=float)
        max_abs = float(np.max(np.abs(data))) if data.size > 0 else 1.0
        coeffs[comp] = max_abs if max_abs > 0 else 1.0
    return coeffs


def _process_method(method_dir: str, out_path: str | None = None, partial: bool = False) -> dict:
    xi_path = os.path.join(method_dir, "Xi.csv")
    selection_path = os.path.join(method_dir, "selection.json")
    if not os.path.exists(xi_path):
        return {}

    xi = pd.read_csv(xi_path, index_col=0)
    selection = {}
    if os.path.exists(selection_path):
        with open(selection_path, "r", encoding="utf-8") as fh:
            selection = json.load(fh)

    method_name = os.path.basename(method_dir)
    raw_path = None
    if method_name in ("SINDy-TIME", "SINDy-UMEE", "TIME-OLS"):
        sg_dir = os.path.join(os.path.dirname(method_dir), "MEEFD-SG")
        # In partial-window mode the model is fit on the truncated inversion, so
        # the reported time normalisation must come from that same file;
        # otherwise the header would quote the full-arc window and be wrong.
        raw_path = os.path.join(sg_dir, "U_MEEFD_inv_fit.csv" if partial else "U_MEEFD_inv.csv")
        if not os.path.exists(raw_path):
            raw_path = os.path.join(sg_dir, "U_MEEFD_inv.csv")

    norm = _read_normalization(raw_path)

    selected_iter = selection.get("selected_max_iter")
    n_active = next((p["n_active"] for p in selection.get("path", []) if p["max_iter"] == selected_iter), "N/A")
    lines = [f"# Method: {method_name}", f"# Active terms: {n_active}"]
    if norm is not None:
        lines.append(f"# Normalization: tau = (t - {norm['time0_s']:.3f}) / {norm['coeff_time_s']:.3f}")
        for comp in _COMPONENTS:
            lines.append(f"#   {comp}_actual = {norm[comp]:.6e} * {comp}_normalized")

    for col in xi.columns:
        expr = _build_expression(xi, col)
        lines.append(f"{col}(x) = {expr}")

    out_path = out_path or os.path.join(method_dir, "IdentifiedExpressions.txt")
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"method": method_name, "path": out_path, "lines": lines}


def export_run(methods_dir: str, output_dir: str, names=None, partial: bool = False) -> list[str]:
    """Export sparse expressions for a single typical run into ``output_dir``.

    Writes one ``IdentifiedExpressions_<Method>.txt`` per method plus a combined
    ``IdentifiedExpressions.txt`` (used by ``sindy-ssa sindy plot``).

    ``partial=True`` selects the truncated ``U_MEEFD_inv_fit.csv`` as the
    normalisation reference (partial-window prediction runs).
    """
    if names is None:
        names = sorted(
            d for d in os.listdir(methods_dir)
            if os.path.isdir(os.path.join(methods_dir, d))
        )
    ensure_dir(output_dir)

    saved: list[str] = []
    all_lines = ["# Identified sparse expressions (typical run)", ""]
    for name in names:
        info = _process_method(
            os.path.join(methods_dir, name),
            out_path=os.path.join(output_dir, f"IdentifiedExpressions_{name}.txt"),
            partial=partial,
        )
        if info:
            saved.append(info["path"])
            all_lines.extend(info["lines"])
            all_lines.append("")

    combined = os.path.join(output_dir, "IdentifiedExpressions.txt")
    with open(combined, "w", encoding="utf-8") as fh:
        fh.write("\n".join(all_lines) + "\n")
    saved.append(combined)
    return saved

