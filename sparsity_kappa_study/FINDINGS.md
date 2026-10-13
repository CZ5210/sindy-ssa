# Findings — why SINDy (original) collapses in the ideal case

All numbers below were produced by `run_kappa_study.py`; raw tables are in
`out/` (partial-window identification, `configs/config.yaml` default
`fit_fraction = 0.75`) and `out_full/` (full-arc identification, which is the
mode the archived `data/<scenario>/typical/*` tables were generated with).

## 0. Harness validation

Running the real `run_reconstruction` pipeline with the study settings
reproduces the archived ideal-case table to 4 significant digits
(stationkeeping: SINDy 24.5923 / 18.6954, SINDy-TIME 0.1166 / 0.2794,
SINDy-UMEE 0.4328 / 0.1478 — see `out_full/stationkeeping/pipeline_check.csv`).
Replacing the identified derivative model with the **true** MEE derivatives
(`ORACLE__true_derivatives`) gives EI = 4.4e-4 m/s against a true manoeuvre
impulse of 3.751 m/s, i.e. the inversion front end, the re-propagation and the
metrics are all sound.

## 1. The reported EI of the failing SINDy runs is a saturation artefact

The pipeline clips the reconstructed thrust at the actuator limit
`T_max/m = 2.125e-7 km/s^2` before computing EI/EPE. SINDy's raw reconstruction
peaks at 1.5e-3 km/s^2 — four orders above the truth (`1.3e-7`) — so
93-100 % of samples are clipped, and the resulting EI converges to
`sum_c ∫ (u_max - |u_true|) dt ≈ 24.6 m/s` **whatever the model does**.
That is why every sparsity setting produces EI ≈ 24.6 and why the archived
ideal-case value (24.59 m/s) carries no information about the model quality.
`clipped_frac` and `ei_raw_total` (pre-clipping EI) are the discriminating
columns in `summary.csv`.

## 2. Two different parameters are both called `kappa`

| symbol | meaning | where it lives | value | live in the shipped pipeline? |
|---|---|---|---|---|
| κ (manuscript Eq. `nastar`) | relative RSS tolerance on the lower envelope: sparsest model with `RSS̄(n_a) - RSS_min <= κ (RSS_max - RSS_min)` | `sindy.sparsity_tol` → `fit_ssr_auto(tol=)` → `select_elbow(tol=)` in `sindy/regression.py` | `0.02` | **yes** — used by SINDy, SINDy-TIME and SINDy-UMEE alike |
| `kappa` (PySINDy SSR) | L0 penalty multiplier `kappa * cond(Theta) * ||Xi||_0`; it only chooses *which* index of the SSR deletion path is returned | `LibrarySpec.kappa`, `sindy.time_kappa`, the `time_library` presets | `1e-19` (config), presets `0.0 / 1e-19 / 1e-15` | **no** — hard-coded to `0.0` whenever `auto_select=True`, and the custom path never reads it; reachable only with `auto_select=False` |

The manuscript's κ is therefore *not* the parameter that is dead. Verified on
the very case plotted in `fig:selection_path`
(`evasive` / `typical` / `reduced`, rerun with this study,
`out_case/evasive-reduced/`):

| method | n_a at κ = 0.02 (this rerun) | n_a printed in the paper | n_a at κ = 0.05 |
|---|---|---|---|
| SINDy (20 terms/column) | 120 | 120 | 114 |
| SINDy-TIME (10 terms/column) | 30 | 30 | 24 |
| SINDy-UMEE (18 terms/column) | 54 | 54 | 42 |

So κ = 0.02 is the value that actually drove the published selections, exactly
as the manuscript states. What *is* wrong is the record: every
`selection.json` in `data/` and `batch_archive/` stores `"sparsity_tol": 0.05`
and no `ridge_alpha`, because the pre-fix `save_selection(...)` call did not
forward the tolerance/penalty and wrote the module default
(`DEFAULT_SPARSITY_TOL`, which was `0.05` until commit `6512024`). The archived
`selected_max_iter` values match the κ = 0.02 selections rather than the
κ = 0.05 ones (stationkeeping/ideal/full: SINDy `sel=12` → `n_a=144`, SINDy-TIME
`sel=15` → `n_a=21`, both reproduced here at κ = 0.02; κ = 0.05 gives 126 and
15). Commit `6512024` ("small bug fixes") already passes `tol=`/`alpha=`
explicitly, so the code is consistent now — but the archived JSON artefacts
still carry the misleading `0.05`.

## 3. It is not the sparsity level, and not PySINDy's `kappa`

PySINDy's `kappa` is unreachable in the shipped pipeline: `auto_select=True`
hard-codes `kappa = 0.0` and the custom deletion path never reads `Spec.kappa`.
Sweeping PySINDy's SSR directly (group `ssr`/`raw`) shows it only selects
*which* point of a `kappa`-independent deletion path is returned:

| variant (SINDy, full arc) | retained coeffs | EI (clipped) | raw EI |
|---|---|---|---|
| `fit_ssr` kappa = 0 or >= 1e-16 | 6 | 23.65 | 534 |
| `fit_ssr` kappa = 1e-19 | 54 | 24.30 | 308 |
| PySINDy `kappa=None` (inflection) | 24 | 23.05 | 388 |
| project auto-selection (`fit_ssr_path` + L-curve) | 144 | 24.65 | 19 343 |
| auto-selection, `sparsity_tol` 0.5 -> 0.001 | 24 -> 198 | 24.85 -> 24.62 | 38 846 -> 18 945 |

So PySINDy's `kappa` moves the retained-term count (6 <-> 54 for SINDy,
3 <-> 33 for SINDy-TIME) and changes the error within the family by a factor of
a few, but it never changes the outcome, and the pipeline's own κ-based
selection is already as good as any `kappa` value for SINDy-TIME / SINDy-UMEE.
The manuscript's κ behaves the same way: sweeping it from 0.001 to 0.5 moves
SINDy's `n_a` from 198 down to 24 while the (clipped) EI stays at 24.56-24.85
and the raw EI stays at 1.9e4-3.9e4 — the sparsity level is simply not what is
wrong.

Two genuine auto-selection defects do show up, though:

* `fit_ssr_path` refits with **ridge only** and never de-biases, while PySINDy's
  SSR refits the selected support by OLS (`unbias=True`). With
  `original_alpha = 1.0` the derivative fit residual is 1.5e-3 (project path)
  versus 1.9e-6 (PySINDy), i.e. three orders worse; the raw EI improves from
  19 343 to 29 when the PySINDy path is used. That is worth ~1e3 in raw EI, but
  it still leaves SINDy 8x above the true impulse.
* `select_elbow` uses the **total** RSS of the six derivative columns, whose
  scales span ~7 decades; the tolerance is therefore dominated by `L_dot`.

## 4. Root cause: the library cannot represent the natural dynamics

The decisive diagnostic is `SINDy__natural__*`: the same library and selector,
but regressed on the **exactly known natural MEE derivative** (no thrust signal,
no noise, no identification ambiguity):

| variant | observed-derivative EI | known-natural-derivative EI |
|---|---|---|
| project auto-selection | 24.65 | 24.67 |
| PySINDy SSR (kappa = 0) | 16.72 | 16.99 |
| PySINDy SSR (kappa = None) | 23.05 | 23.13 |

The two columns agree to within a few percent, so essentially *all* of SINDy's
error comes from the library's inability to represent the natural MEE dynamics,
not from the thrust identification, the selector or `kappa`. The residual is
then mapped through the inverse Gauss variational equations, which are extremely
ill-conditioned: on the fit window the natural background of each derivative is

| derivative | natural RMS | thrust-induced RMS | ratio |
|---|---|---|---|
| p_dot | 6.94e-03 | 4.60e-05 | 1.5e+02 |
| f_dot | 1.32e-06 | 1.04e-08 | 1.3e+02 |
| g_dot | 1.56e-06 | 1.05e-08 | 1.5e+02 |
| h_dot | 3.14e-07 | 3.24e-11 | 9.7e+03 |
| k_dot | 5.41e-07 | 2.56e-11 | 2.1e+04 |
| L_dot | 1.10e-03 | 1.88e-10 | **5.9e+06** |

To recover a 3.75 m/s impulse the derivative model must resolve `L_dot` to
~1e-11 rad/s, i.e. a *relative* accuracy of ~1e-8 with respect to the natural
background. The polynomial/trigonometric library (manuscript Eq. 23) is 4-7
orders of magnitude away from that. `SINDy-TIME` and `SINDy-UMEE` avoid the
problem because they regress the *already inverted* thrust, where the target is
O(1) and no cancellation is involved.

## 5. Same conclusion in the second scenario and in the partial-window mode

Evasive ideal (partial window): SINDy EI 7.2-9.5, EPE 11-42 km; SINDy-TIME
EI 1.08-2.44, EPE 1.1-18 km; SINDy-UMEE EI 1.3-2.7, EPE 1.9-17 km; the
known-natural-derivative fits again land at 8.2-9.5.

The flaw is masked in the noisy Monte-Carlo tiers, where every method sits on
the saturation floor (`batch_archive/full/stationkeeping/high`: median EPE
31-59 km for all six methods), and only becomes visible in the noise-free tier
(`ideal`: SINDy 18.7 km versus 0.03-0.8 km for the other five).

## 6. Dedicated scan: the manuscript's κ (`sparsity_tol`) on SINDy only

Ideal case, full-arc identification, 100 % observation cadence, SINDy only,
everything else at the configured values (`original_alpha = 1.0`,
`normalize_columns = true`).  Run with
`run.py --case ideal --fit-fraction 1.0 --groups tol --reprop`; tables in
`out_tol/<scenario>/tol_scan.csv`, figures in
`out_tol/<scenario>/tol_scan_<scenario>.png`.

**Stationkeeping** (true manoeuvre impulse 3.751 m/s; reference values:
SINDy-TIME 0.117 m/s / 0.279 km, SINDy-UMEE 0.433 m/s / 0.169 km at κ = 0.02,
oracle 4.4e-4 m/s):

| κ | deletions/column | n_a | terms/column | EI [m/s] | EI before clipping | clipped | EPE 3D [km] | EPE 2D [km] |
|---|---|---|---|---|---|---|---|---|
| 0.001 | 3 | 198 | 33 | 24.62 | 18 945 | 0.997 | 24.57 | 24.45 |
| 0.005 | 5 | 186 | 31 | 24.62 | 18 992 | 0.997 | 23.31 | 23.18 |
| 0.01 | 8 | 168 | 28 | 24.63 | 19 165 | 0.999 | 22.33 | 22.22 |
| **0.02** | 12 | 144 | 24 | 24.65 | 19 343 | 0.999 | 21.72 | 21.51 |
| 0.05 | 15 | 126 | 21 | 24.65 | 19 661 | 0.999 | 20.72 | 20.51 |
| 0.1 | 20 | 96 | 16 | 24.77 | 20 382 | 0.999 | 18.87 | 18.77 |
| 0.2 | 27 | 54 | 9 | 24.76 | 21 730 | 1.000 | 10.79 | 10.39 |
| 0.3 | 29 | 42 | 7 | 24.78 | 25 410 | 1.000 | 7.08 | 6.46 |
| 0.5 | 32 | 24 | 4 | 24.85 | 38 846 | 1.000 | 14.88 | 14.59 |

**Evasive** (true impulse 2.597 m/s; oracle 1.6e-4 m/s): n_a falls from 156 to
48 (26 -> 8 terms per column) as κ goes from 0.001 to 0.5, while EI stays in
9.12-9.25 m/s and the pre-clipping EI in 7.3e3-8.7e3 m/s; EPE drifts from
17.6 km to 12.8 km with no systematic relation to κ.

Conclusions from the scan:

* κ does exactly what Eq. `nastar` prescribes - it moves the retained-term
  count monotonically (stationkeeping: 4 -> 33 terms per column, n_a 24 -> 198;
  evasive: 8 -> 26 per column).
* SINDy does **not** improve.  The reported EI is pinned at the saturation
  floor (24.6 m/s stationkeeping, 9.2 m/s evasive) for every κ because
  99.7-100 % of the samples are still clipped; the pre-clipping EI stays
  5000x (stationkeeping) / 3000x (evasive) the true impulse.  A perfect
  derivative model would give 4.4e-4 m/s, three orders below even the best
  point of this scan.
* EPE moves by at most a factor of ~3 with κ, is *worst* at the manuscript's
  κ = 0.02 for stationkeeping, and never comes within 25x of SINDy-TIME /
  SINDy-UMEE (0.17-0.28 km).
* The 2D (planar-prior) and 3D re-propagation errors differ by < 6 %, i.e. the
  failure is not a normal-direction artefact but a wrong thrust magnitude
  everywhere.

### 6.1 EPE-optimal κ (ideal, full arc, 100 % observation)

Convention: `run_reconstruction` in full-arc mode evaluates the state-based
methods on the **observed** states, so the sweep was repeated with
`--future-state observed` (`out_tol_obs/`).  It reproduces the pipeline exactly
at κ = 0.02: SINDy 24.592 m/s / 18.695 km, SINDy-TIME 0.1166 / 0.2794,
SINDy-UMEE 0.4328 / 0.1478 (stationkeeping).

**Stationkeeping**

| κ | n_a | terms/column | EI [m/s] | EPE 3D [km] | EPE 2D [km] |
|---|---|---|---|---|---|
| 0.005 | 186 | 31 | 24.59 | 20.42 | 20.29 |
| **0.02 (manuscript)** | 144 | 24 | 24.59 | 18.70 | 18.47 |
| 0.05 | 126 | 21 | 24.60 | 17.84 | 17.61 |
| 0.1 | 96 | 16 | 24.72 | 16.19 | 16.08 |
| 0.15 | 66 | 11 | 24.74 | 12.63 | 12.38 |
| 0.2 | 54 | 9 | 24.72 | 8.32 | 7.80 |
| 0.22-0.25 | 48 | 8 | 24.73 | 4.69 | 3.91 |
| **0.28-0.30 (optimum)** | **42** | **7** | 24.74 | **4.51** | **3.47** |
| 0.32-0.40 | 36 | 6 | 24.76 | 5.38 | 4.54 |
| 0.45-0.60 | 24 | 4 | 24.82 | 12.75 | 12.41 |
| 0.7-0.8 | 18 | 3 | 24.95 | 20.63 | 20.41 |
| >= 1.0 | 6 | 1 | 24.95 | 30.30 | 30.15 |

**Evasive**: optimum at κ = 0.80 (n_a = 12, 2 terms/column) with EPE 9.67 km
(2D 9.50) versus 17.32 km (17.26) at κ = 0.02; the curve runs
16.84 -> 15.72 -> 14.45 -> 12.48 -> 11.60 -> 12.04 -> 12.69 -> 10.27 -> **9.67**
-> 13.60 km as κ goes 0.005 -> 1.0.

So the EPE-minimising κ exists and is worth **4.1x** (stationkeeping:
18.70 -> 4.51 km) / **1.8x** (evasive: 17.32 -> 9.67 km) relative to the
manuscript's κ = 0.02, and 3D/2D agree on the same κ.  Two caveats:

* the curve is **not monotonic** in κ (stationkeeping dips at κ≈0.3 and rises
  again to 30 km at κ>=1), so κ* is a notch in a one-dimensional discrete
  sparsity path rather than a transferable tuning; it must be re-found per
  scenario.
* EPE improves while the thrust does not: the unclipped model output is
  1e3-1e4 mm/s^2, three to four decades above the true thrust (~0.13 mm/s^2),
  so after saturation the reconstruction is a **square wave at ±T_max/m**
  (`best_kappa/best_rtn_<scenario>.png`, `best_rtn_paper_style_<scenario>.png`).
  At κ* the square wave's sign pattern happens to nearly cancel at the end,
  which is what lowers EPE; EI stays pinned at the saturation floor
  (24.74 m/s / 8.96 m/s).
* even the best κ leaves SINDy 31x (stationkeeping) / 66x (evasive) worse than
  SINDy-UMEE (0.169 km / 0.146 km) and 16x / 13x worse than SINDy-TIME
  (0.279 km / 0.739 km) in the same full-arc run.

## 7. Legacy vs current SINDy: which step changed

The pre-refactor pipeline is in `FYP_Development_V8/`
(`SINDy/SINDy_Ori.py`, `SINDy/SINDy_Main.py`, `Evaluate.py`).  `legacy_vs_current.py`
re-runs both recipes on the same ideal data and reproduces the archived legacy
thrust (`FYP_Development_V8/ReportFigs/FreqBatch/*/Output/SINDyOri/U_SINDyOri.csv`)
for reference.

| step | legacy `FYP_Development_V8` | current `sindy-ssa` |
|---|---|---|
| observation cadence | downsample ×12 -> 120 s | ×6 -> 60 s (`ideal`: ×1 -> 10 s) |
| design matrix | `p/6371`, library built on that | raw `p` in km; `normalize_columns=True` inside the optimiser |
| SSR refit | `ps.SSR(...)` defaults = **`unbias=True`** (OLS refit on the selected support) | `fit_ssr_path` refits with **ridge only, never de-biased** |
| sparsity selection | PySINDy's own error-inflection rule (`kappa` not passed) | custom deletion path + the manuscript κ L-curve |
| thrust post-processing | `MinusAcc(...).smooth(window=51)` — SG smoothing of the RTN thrust | none (only MEEFD-SG smooths) |
| actuator saturation | **none** | `saturate_thrust` at `T_max/m` before EI and before re-propagation |
| evaluation window | whole observation arc; `Evaluate.py` reports R, T, N separately and has an "excluding N" variant | manoeuvre interval, three components |

Note `p` does not appear in the SINDy library (Eq. 23), so the `p/6371` scaling
is a no-op; the load-bearing legacy switches are `unbias`, `normalize_columns`
and the absence of clipping.

Ablation on the same ideal data (`legacy_ablation.csv`, stationkeeping; EI in
m/s, `max|u_n|` in km/s^2, actuator limit `2.125e-7`):

| selection | unbias (OLS refit) | normalize_columns | n_a | max u_n | EI | EI after clipping |
|---|---|---|---|---|---|---|
| inflection | **True** | False | 102 | 6.2e-07 | 16.7 | 12.7 |
| inflection | True | True | 114 | 1.1e-06 | 17.9 | 11.1 |
| inflection | **False** | False | 102 | 3.1e-05 | 506 | 22.7 |
| inflection | **False** | True | 114 | **1.6e-03** | **18 870** | 24.6 |
| L-curve (κ=0.02) | **True** | False | 102 | **1.1e-07** | **1.96** | 1.96 |
| L-curve (κ=0.02) | True | True | 144 | 1.3e-07 | 3.66 | 3.35 |
| L-curve (κ=0.02) | **False** | False | 102 | 3.1e-05 | 506 | 22.7 |
| L-curve (κ=0.02) | **False** | True | 144 | **1.6e-03** | **19 040** | 24.6 |

Conclusions:

1. **The lost de-biasing dominates.** `unbias=False` (the current custom path)
costs 3-4 orders of magnitude in EI and turns `max|u_n|` from ~1e-7 into
1.6e-3 (7500x the actuator limit).  The current pipeline sits exactly in the
worst cell (L-curve + no de-bias + column normalisation).
2. **Column normalisation is a consistent second-order penalty** (EI 3.66 vs
1.96, 17.9 vs 16.7) because it interacts badly with the ridge penalty
`alpha = 1`.
3. **The manuscript's κ selection is not at fault** — with de-biasing it is the
*best* setting tested (EI 1.96 m/s, below the true impulse 3.75 m/s, no
saturation).  The legacy inflection rule is 8x worse.
4. **Saturation is new and hides the failure.** The legacy code never clipped;
the current one clips before both EI and re-propagation, so any over-limit
history becomes a ±`T_max/m` square wave and EI collapses onto the ~24.6 m/s
floor.  Applying the legacy (unclipped) convention to the current model gives
EI = 1.9e4 instead.
5. SG smoothing of the thrust (legacy only) removes a further factor ~2.3
(EI 16.7 -> 7.1 for the legacy recipe) and is what made the old curves look
smooth.
6. The archived legacy output itself (`U_SINDyOri.csv`, 2-min campaign) has
`max u_r = 5.8e-07`, `max u_t = 3.9e-07` (the true level) and `max u_n = 8.1e-05`
(380x the actuator limit) — bounded and smooth, which matches the appearance of
`FYP_Development_V8/Figures/AccCompare1.png`; the current model's `u_n` is 20x
larger and then clipped.

Minimal repair: in `fit_ssr_auto`/`fit_ssr_path`, refit the selected support by
OLS (`unbias` behaviour) and switch the derivative library to
`normalize_columns=False` (or rescale after the OLS refit).  That alone takes
the ideal-case SINDy from EI 19 040 -> 1.96 m/s (clipped 24.6 -> 1.96) and
removes the saturation entirely.

## 8. Practical implications

1. For the ideal tier, report `clipped_frac` (or the unsaturated EI) alongside
   EI; EI alone collapses onto the saturation floor and cannot separate a
   broken model from a merely optimistic one.
2. If SINDy-on-derivatives is to be kept, the identification target has to be
   the *thrust-induced* derivative (observation minus the known natural GVE
   term) rather than the total derivative, or the library has to contain the
   natural two-body/J2 terms.
3. De-bias `fit_ssr_path` (OLS refit on the selected support) and normalise the
   RSS per target column in `select_elbow`; both are cheap and remove 3 orders
   of spurious residual.
4. PySINDy's `kappa` is dead code in the pipeline and shares its name with the
   manuscript's κ, which is what makes the two easy to confuse. Rename it
   (`ssr_kappa`) and either wire it through the auto path (also stop overwriting
   `None` with `0.0` in `_ssr_kwargs`, which silently disables PySINDy's own
   error-inflection rule) or drop it, including `sindy.time_kappa`.
5. Re-record `sparsity_tol`/`ridge_alpha` in the archived `selection.json`
   files (or regenerate them): they currently show `0.05` although the
   selections were computed with `0.02`.
