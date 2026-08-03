# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Before publishing — do these first

- **Email Bobby (Robert Gramacy, rbg@vt.edu) before any public release.** `lagpax` is a translation of
  his laGP source, not a clean-room reimplementation. LGPL-2.0-or-later covers it legally; the email
  is about doing it right, and he is a plausible JOSS co-author or reviewer. Do this *before* the
  first PyPI upload, not after.
- Reserve the `lagpax` name on PyPI. Plain `lagp` was already squatted by an unrelated project.
- CI running the parity suite is the credibility artifact for this package. Needs R + laGP in the
  runner (`r-lib/actions/setup-r`, then laGP from CRAN).
- Do not weaken the license to MIT/Apache. See "License" below.

## What this is

A JAX port of the laGP R package (1.5.9), separable/`GPsep` family. Bit-parity with R is the design
goal and is enforced by `tests/`. Extracted from the LocalVorcans prototype repo, which still holds
an older, independent local-GP/Bayesian-optimization stack that does **not** implement the same math
— don't import assumptions from there.

The pitch is nonstationarity + differentiability + verified parity, not raw speed. See README.

## Commands

```bash
module load r/4.4.0 && export R_HOME=$(R RHOME)   # needed for rpy2 in tests/benchmarks
uv sync --extra tests                              # pytest + rpy2 (rpy2 must stay <3.6)
uv run --no-sync pytest tests -q                   # parity suite vs R
uv run -m bench.accuracy                           # JAX vs laGP predictions -> bench/accuracy.csv
uv run -m bench.timing                             # wall clock -> bench/timing.csv
uv run -m bench.plot                               # -> bench/benchmark.png
```

## src/lagpax

Parameterization is laGP's throughout: correlation `exp(-sum_k (x1_k - x2_k)^2 / d_k)`, nugget `g` on
the diagonal, zero mean, scale profiled out as `phi = Z' Ki Z`, Student-t predictive with `df = n`.

- `covar.py` — `distance` (squared euclidean), `covar_sep`, `covar_sep_symm`, `diff_covar_sep`.
- `gp.py` — `GPsep` NamedTuple caching `K`, `Ki`, `ldetK`, `KiZ`, `phi`. `Ki` is materialized (not just
  a Cholesky factor) because ALC needs it. `llik_gpsep` adds the Gamma log-priors when `dab`/`gab` are
  non-zero. Gradients come from `jax.grad`, not hand-written derivatives.
- `prior.py` — host-side numpy/scipy port of `darg`/`garg`/`getDs`/`Igamma.inv`. Deliberately *not*
  jitted: it runs once per call and R's `quantile` type 7 == numpy's default.
- `optim.py` — `lbfgsb` (projected L-BFGS, Armijo on the projected arc, R's `factr`/`pgtol` stopping
  rules) and `bracketed_newton` for the nugget. Neither reproduces R's iterate trajectory; they find
  the same KKT point. Tests assert that, plus "our llik is no worse than laGP's".
- `mle.py` — `mle_gpsep_d` (with laGP's `0.9*dmin + 0.1*dmax` boundary reset, conv=102),
  `mle_gpsep_nug`, `jmle_gpsep`, `mle_switch`.
- `alc.py` — Active Learning Cohn, port of `alcGPsep`.
- `local.py` — `lagpsep`/`agpsep`. `_lagpsep_core` is jitted with the sizes static; `agpsep` maps it
  over predictive locations with `jax.lax.map(batch_size=)`.

### Why the mask exists

Greedy ALC design selection grows the local design one point at a time, but XLA needs fixed shapes.
`GPsep.mask` marks the active rows: masked rows/columns of `K` are set to the identity, so `ldetK`
gains 0, masked `Z` entries are zeroed so `phi` is unaffected, and the identity block in `Ki` is
killed by masking `k`/`kx` in `pred_gpsep`/`alc_gpsep`. Break any one of those three and the padded
GP silently stops matching the unpadded one — `tests/test_gp.py::test_mask_matches_a_smaller_design`
and `tests/test_alc.py::test_alc_ignores_masked_rows` are the guards.

### laGP conventions worth remembering

- `laGPsep` returns raw `s2`; `aGPsep` returns `var = s2 * df/(df-2)`. `agpsep` reproduces both.
- `Xi` is 1-based in R, 0-based here.
- `darg`/`garg` subsample the design when `n > 1000` using R's RNG, so parity tests stay under 1000.
- `dllikGPsep` is not exported from the laGP namespace; call it as `laGP:::dllikGPsep`.

## tests/

`tests/conftest.py` provides the `R` fixture (rpy2 + laGP from `~/R/library`, skipping when either is
unavailable) and the shared designs: `small` (40x3), `medium` (600x4), `wide` (200x8), `noisy` (300x3
with observation noise). Everything except `tests/test_transforms.py` compares against R.

`noisy` exists because on a noiseless design the nugget MLE is pinned at the lower bound in a badly
conditioned regime — laGP and the port disagree there in the 8th digit while our likelihood is
*higher*. Those cases are tested as "no worse than laGP", not as equality; interior-optimum parity is
tested on `noisy`.

Tolerances are not uniform on purpose. `s2` is computed as `1 + g - k' Ki k`, a cancellation, so its
absolute error tracks the O(1) intermediate rather than the ~1e-3 result; comparisons use
`rtol=1e-8, atol=1e-11` rather than the 1e-12 used for `mean`/`llik`/`phi`. Reassociation under `jit`
and `vmap` costs another few ulps, which is why batch-invariance checks are 1e-6..1e-9, not exact.

## bench/

`bench/common.py` holds the Ackley designs, the `RlaGP` bridge and `timeit` (best of 3, after a
warm-up call that absorbs JIT). `accuracy.py` measures predictive agreement, `timing.py` wall clock
(JAX vs laGP at 1 thread and at `os.cpu_count()`), `plot.py` draws both.

`accuracy.py` computes `darg` once and hands the same prior to both sides. Without that it measures
the RNG gap, not the port: at n=2000 the median relative error is ~1e-4 with independent priors and
~1e-8 with a shared one. Keep that; it is the difference between a benchmark and noise.

`OPENBLAS_NUM_THREADS=1` is set at the top of `bench/common.py`, before jax or R load, so laGP's
`omp.threads` is the only thing that varies between the 1-core and N-core rows.

### Reading the timing CSVs

The `vs16c` ratio is **not comparable across CSV files** — the GPU runs were done on different SLURM
nodes whose host CPUs differ by ~3x (V100 node's 16-core laGP baseline is 13.70s where the L40S
node's is 4.70s for identical work). Only GPU-to-GPU wall clock is clean. Fix this before publishing
any cross-chip comparison: pin the same host CPU, or report laGP baselines per node.

Current state: a 25x fp64-throughput range across GTX 1080 / L40S / V100 produces only a 2.9x spread
in wall clock, so the workload is bound by kernel-launch latency and the sequential ALC greedy loop,
not arithmetic. Buying a better GPU does not fix that. The open levers are a rank-1 Cholesky update
in the ALC loop, `lax.scan` over the greedy steps, fp32 for the ALC selection stage with fp64 for the
final fit, and a batch-size sweep.

## License

LGPL-2.0-or-later, inherited from laGP. laGP's DESCRIPTION declares a bare `License: LGPL`; per
CRAN's *Writing R Extensions* the unversioned form means version 2 or later. This is a derivative
work — it was translated from laGP's source, so relicensing to MIT/Apache is not available without
Gramacy's permission. `NOTICE` carries the attribution and citations; keep it accurate if modules are
added.

## Gotchas

- A bare `uv sync` prunes the `tests` extra, and rpy2 then dies with
  `ModuleNotFoundError: No module named '_cffi_backend'`. Re-run `uv sync --extra tests`.
- Piping a long background run through `tail` buffers everything until it exits, so the output file
  stays empty. Let it write its CSV and read that instead.
