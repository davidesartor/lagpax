# CLAUDE.md

Operating manual for Claude Code in this repo, loaded every session: how the code works, how to run
it, and what will bite you. It states facts about the repo *as it is*.

- `ROADMAP.md` — status, the rules that govern new work, what was learned (every measurement lives
  there), and what is next. Read its "Rules" before changing anything numerical.
- `README.md` — the user-facing pitch.

Keep the three in sync and don't duplicate: a number belongs in `ROADMAP.md`, a how-to here.

## What this is

A JAX port of the laGP R package (1.5.9), separable/`GPsep` family. Parity with R is the design goal
and is enforced by `tests/`. The pitch is nonstationarity + differentiability + verified parity, not
raw speed. Extracted from the LocalVorcans prototype repo, which holds an older, independent local-GP
stack that does **not** implement the same math — don't import assumptions from there.

The laGP capability list is closed. Box-constrained solves are delegated to `vlse.optim.minimise`
(the `jaxvlse` package, same author), whose SFU test functions also drive a parity module on real
response surfaces. What is open is in `ROADMAP.md` § Next.

### Before publishing

Full checklist in `ROADMAP.md`. The parts you must not get wrong from memory:

- **The PyPI placeholder ships as `0.0.1`, never `0.1.0`.** PyPI never lets a version number be
  reused and deleting a release does not free its filename. Plain `lagp` was already squatted by an
  unrelated project, so `lagpax` being free today proves nothing.
- **CI must fail on a nonzero skip count.** `.github/workflows/tests.yml` installs R + laGP and sets
  `LAGPAX_STRICT=1`, which turns any skip into a nonzero exit — keep it set; it is the whole point.
- **The license question is open.** lagpax is LGPL-2.0-or-later and now depends on `jaxvlse`, which
  is GPL-2.0-only. See "License" below; resolve it before the first real release.

## Commands

```bash
export R_HOME=$(R RHOME)                              # rpy2 needs it (`module load r/4.4.0` on the cluster)

uv sync --extra tests                                 # pytest + rpy2 (rpy2 must stay <3.6)
LAGPAX_STRICT=1 uv run --no-sync pytest tests -q      # parity suite vs R; a skip is a failure
uv run --no-sync pytest tests/test_mle.py -q          # one module
uv run --no-sync pytest tests/test_gp.py::test_mask_matches_a_smaller_design   # one test
uv run --no-sync pytest tests -q -k "alc and not mask" # by name
python -m tests.snapshot check ~/.cache/lagpax/snapshot.npz   # did anything move?

uv sync --extra bench                                 # + matplotlib/pandas; NOT in the tests extra
uv run --no-sync -m bench.accuracy                    # JAX vs laGP predictions -> bench/accuracy.csv
uv run --no-sync -m bench.timing --suite locations    # wall clock vs predictive-location count
uv run --no-sync -m bench.plot                        # -> bench/benchmark.png
```

`--no-sync` on every `uv run` is load-bearing: a bare `uv run` re-syncs the default env and prunes
whichever extra is installed. `tests` and `bench` are separate extras and each `uv sync` drops the
other.

Without R + laGP the `R` fixture skips and only the property tests run, so a green `pytest` proves
nothing unless R loaded — hence `LAGPAX_STRICT=1`. `LAGPAX_R_LIB` overrides the `~/R/library` search
path; empty means R's own default library, which is what CI uses.

### R in a Claude Code cloud container

CRAN (`cran.r-project.org`, `cloud.r-project.org`) is blocked by the container's egress policy, but
apt and GitHub are not. This gets laGP 1.5.9 into `~/R/library`:

```bash
apt-get install -y r-base-core r-base-dev r-cran-tgp      # tgp is laGP's only non-base import
git clone https://github.com/cran/laGP ~/R/src/laGP       # CRAN's read-only mirror, tagged per release
git -C ~/R/src/laGP fetch --depth 1 origin tag 1.5-9 && git -C ~/R/src/laGP checkout 1.5-9
mkdir -p ~/R/library && R CMD INSTALL -l ~/R/library ~/R/src/laGP
```

The mirror's default branch is the newest release (1.5-10 at the time of writing), not the 1.5.9
this port targets — check out the tag.

## src/lagpax

Parameterization is laGP's throughout: correlation `exp(-sum_k (x1_k - x2_k)^2 / d_k)`, nugget `g` on
the diagonal, scale profiled out as `phi = resid' Ki resid`, Student-t predictive with `df = n - p`.
`p` is the trend basis width and is 0 under the laGP-equivalent default `trend="zero"`, where `resid`
is `Z` and `df` is `n`.

- `covar.py` — `distance` (squared euclidean), `covar`, `covar_symm`, `dcovar`.
- `gp.py` — `GP` NamedTuple caching `K`, `Ki`, `ldetK`, `KiZ`, `phi`, plus `F`, `b`, `Kfi`, `ldetF`
  for the trend. `Ki` is materialized (not just a Cholesky factor) because ALC needs it. `_llik` adds
  the Gamma log-priors when `dab`/`gab` are non-zero. `_profile` is the MLE inner-loop objective —
  one Cholesky, two triangular solves, no explicit inverse. Gradients come from `jax.grad`.
- `prior.py` — host-side numpy/scipy port of `darg`/`garg`/`getDs`/`Igamma.inv`, behind
  `Prior.from_design`. Deliberately *not* jitted: it runs once per call and R's `quantile` type 7 ==
  numpy's default.
- `optim.py` — `bracketed_newton` for the nugget, `implicit_argmin`, and `R_FTOL`. There is no
  L-BFGS here any more: `mle._solve_d` and `search.continuous` call `vlse.optim.minimise` directly
  with `tol=0.0, ftol=R_FTOL` — R's `optim` default `factr=1e7` (times machine epsilon) and
  `pgtol=0`. `mle._solve_d` turns vlse's state into R's convergence code (1 only when `maxit` ran out
  before a stopping rule fired). `bracketed_newton` takes one `dfun` returning `(d, d2)`, so the
  nugget MLE gets both derivatives from a single factorization per iteration. Neither solver
  reproduces R's iterate trajectory; they find the same KKT point.
- `mle.py` — `mle_d` (with laGP's `0.9*dmin + 0.1*dmax` boundary reset, conv=102), `mle_g`, `jmle`,
  and `fit` dispatching on the prior's `estimate` flags. `jmle` carries `(d, g)` through its
  `while_loop` rather than a GP, so the alternating half-steps stop factorizing a `K` nobody reads.
- `criteria.py` — `nn`, `alc` (port of `alcGPsep`), `fisher` (port of `efiGP`/`fishGP`), and the
  `ieci(fmin=, w=, nonug=)` and `mspe(fi=)` factories. A `Criterion` carries `minimize`, so the
  greedy loop never hardcodes argmax vs argmin — and configurable criteria have to be factories
  rather than `functools.partial`, because a partial does not forward that attribute and
  `search._pick` reads it with `getattr(..., False)`.
- `blhs.py` — block Latin hypercube subsampling, port of `blhs`/`blhs.loop`. Host-side numpy like
  `prior.py`: every draw has a different row count, so there is nothing to batch. `bootstrap` returns
  a `Prior`.
- `search.py` — `exhaustive`, `ray`, `continuous`. All three take
  `(criterion, gp, Xcand, Xref, taken, step, rect)` and return one candidate index. `_block` picks
  "the offset-th *available* candidate" the way laGP's deletion from its candidate list does, by
  cumsum + `searchsorted`; an argsort there would dominate the search.
- `local.py` — `Local`, `Conditioned`, `lagp`/`agp`. `_core` is jitted with a `_Cfg` NamedTuple as the
  static config; `predict` maps it over predictive locations with `jax.lax.map(batch_size=)`.
  `_design` carries the candidate/design cross-covariance across greedy steps. `_buckets` groups the
  greedy steps by padded capacity — step `t` builds its GP at the next power of two above `t`, not at
  `end` — so the loop traces a logarithmic number of bodies.

The public API is flat: `__init__.py` re-exports every name and pins them in `__all__`, so a new
public function goes in two places there. Importing the package runs
`jax.config.update("jax_enable_x64", True)` for the whole process: every tolerance in `tests/` assumes
it and laGP is float64 throughout. (Removing that side effect is part of the redesign proposal in
`ROADMAP.md`.)

### Why the mask exists

Greedy design selection grows the local design one point at a time, but XLA needs fixed shapes.
`GP.mask` marks the active rows, and there are **four** invariants: masked rows/columns of `K` are set
to the identity, so `ldetK` gains 0; masked `Z` entries are zeroed so `phi` is unaffected; masked `F`
rows are zeroed so the GLS solve ignores them; and the identity block in `Ki` is killed by masking
`k`/`kx` in `_predict`/`alc`. Break any one and the padded GP silently stops matching the unpadded
one — `tests/test_gp.py::test_mask_matches_a_smaller_design`,
`tests/test_criteria.py::test_alc_ignores_masked_rows` and the trend mask test in `tests/test_trend.py`
are the guards.

The invariants are exact but the padding *width* is not neutral: `alc` differs by ~1e-16 across
widths, which flips near-tied argmaxes and cascades through the greedy loop. **Judge a change here on
the parity suite and on rmse, never on whether `Xi` is unchanged.**

### Gradients through the fit

`optim.implicit_argmin` is what makes `jax.grad` work through `mle_d`/`mle_g`/`jmle`. Both solvers are
`lax.while_loop`s with no reverse-mode rule, so the answer is computed under `stop_gradient` and the
derivative attached afterwards: one Newton step from the frozen solution carries exactly the
implicit-function term, and subtracting its own `stop_gradient` leaves the **value bit-identical**.
A hyperparameter on an active bound, or one produced by the conv=102 reset, is frozen and its
gradient is exactly zero.

The derivative is exact at the true argmin and is evaluated at the solver's, so its accuracy is the
solver's stationarity. Do not check it by finite-differencing the solver; `tests/test_diff.py`
compares against an argmin root-found with scipy instead. A finite difference of the *whole pipeline*
needs a step below the scale at which a perturbation changes which iterate L-BFGS-B stops on — `1e-5`
on the test designs (see `ROADMAP.md` § What we learned).

### laGP conventions worth remembering

- `laGPsep` returns raw `s2`; `aGPsep` returns `var = s2 * df/(df-2)`. `Prediction` carries both.
- `Xi` is 1-based in R, 0-based here.
- `darg`/`garg` subsample the design when `n > 1000` using R's RNG, so parity tests stay under 1000.
- `dllikGPsep` is not exported from the laGP namespace; call it as `laGP:::dllikGPsep`. Same for
  `mspeGP`, `fishGP` and `dmus2GP`.
- `mspeGP`, `fishGP` and `dmus2GP` are **isotropic-family** entry points (`newGP`, not `newGPsep`),
  so their parity tests build a separable GP with every `d_k` equal — which is that GP exactly.
- laGP is inconsistent with itself in one place the port has to copy: `efiGP` computes
  `sq(dmu)/s2` where `mspeGP` computes `sq(dmu/s2)` in the otherwise identical Fisher denominator.
  Both are reproduced as written. Do not "fix" either.
- `dmus2GP`'s `s2` divides by `n - 2`, so it is the Student-t *variance*; `predGP_lite`'s is the raw
  scale with `df = n`. `mspeGP` uses both, one line apart.
- `blhs` bins on the *open* interval `((i-1)/m, i/m)`, so a coordinate of exactly 0 or 1 lands in
  bin 1 regardless. Reproduced; it only bites on gridded designs.

## API

```python
gp = lagpax.GP(X, Z, d, g, mask=None)
gp.llik(prior=None)                      # MAP when a prior is passed, MLE otherwise
gp.predict(XX, lite=True, nugget=True)
gp.update(Xnew, Znew)
gp.at(d=..., g=...)

prior = lagpax.Prior.from_design(X, Z, rng=None)          # -> Prior(d=Range(...), g=Range(...))
prior = prior.at(g=lagpax.Range(start=1e-6, estimate=False))

lagpax.alc(gp, Xcand, Xref)              # criteria are (gp, Xcand, Xref) -> Float[Array, "ncand"]
lagpax.nn(gp, Xcand, Xref)
lagpax.fisher(gp, Xcand)                 # Xref optional here: it scores no reference set
lagpax.ieci(fmin=...)(gp, Xcand, Xref)   # criteria needing config are factories
lagpax.mspe(fi=True)(gp, Xcand, Xref)

scheme = lagpax.Local(
    start=6, end=50,
    criterion="alc",        # "nn" | "alc" | "mspe" | "fisher" | lagpax.ieci(fmin=...) | callable
    search="exhaustive",    # "exhaustive" | "ray" | "continuous"
    trend="zero",           # "zero" | "constant" | "linear"; "zero" reproduces laGP
    close=None, maxit=100,
)
emu = scheme.condition(X, Z, prior=None, rng=None)
out = emu.predict(XX, batch=256)
out = emu.refine(XX, passes=("nn", "alc"))

lagpax.agp(X, Z, XX, **opts)       # whole dataset
lagpax.lagp(X, Z, Xref, **opts)    # one location — data first, matching agp

out.mean, out.s2, out.df, out.var, out.Xi, out.fit    # var = s2*df/(df-2); Xi 0-based
```

Load-bearing design points:

- **`condition` is the seam.** Resolving the prior is the only host-side numpy/scipy step; putting it
  up front is what lets everything downstream be pure JAX. Do not let prior resolution leak back into
  the per-location path. It is also where a `bootstrap(...)` prior plugs in.
- **`refine` replaces `agpsep_seq`'s kwargs forwarding**, which is where a `d`-resolution bug lived:
  each pass re-ran `darg` and re-subsampled. Its pass loop stays a Python loop — passes may name
  different criteria that trace to different functions.
- **One `Prediction` for both entry points**, ending laGP's raw-`s2`-vs-scaled-`var` split.
- **`close`** bounds the candidate set for every search, defaulting to `10*(1000+end)` for ray and
  continuous and `1000+end` for exhaustive, matching laGP.

The public surface is seventeen names: `GP`, `Prior`, `Range`, `Prediction`, `Fit`, `Local`, `agp`,
`lagp`, `fit`, `alc`, `fisher`, `ieci`, `mspe`, `nn`, `bootstrap`, `distance`, `covar`. Everything
else stays importable from its own module; `tests/` reaches into `lagpax.optim` and `lagpax.search`.

Two things laGP cannot do, both shipped: **trend models** (`trend="constant"`/`"linear"`, sold as
interval calibration, not a better mean — `tests/test_trend.py` tests coverage) and **gradients
through fitting** with a compilable pipeline (no host callbacks after `condition`). Easy to
under-use: **iterated refinement** — `("nn", "alc")` picks neighbourhoods at fitted lengthscales and
cuts rmse ~1.7-2x on anisotropic problems.

## tests/

`tests/conftest.py` provides the `R` fixture (rpy2 + laGP, skipping when either is unavailable) and the
shared designs: `small` (40x3), `medium` (600x4), `wide` (200x8), `noisy` (300x3 with observation
noise).

- **Parity against R:** `test_covar`, `test_gp`, `test_prior`, `test_mle`, `test_criteria`,
  `test_ieci`, `test_mspe`, `test_blhs`, `test_local`, and `test_vlse`.
- **Property tests** (no laGP equivalent): `test_transforms`, `test_search`, `test_trend`,
  `test_diff`.

`test_vlse` runs `jmleGPsep` and `aGPsep` on eight SFU benchmark surfaces from `vlse` (Branin,
Hartmann3/6, Michalewicz, Ackley, Rosenbrock, Goldstein-Price, Shekel), unit-cube inputs, standardized
response. The random-feature designs are smooth by construction; these are where a lengthscale fit
meets narrow valleys, large dynamic range and many basins. Assertions are tiered by surface — read
its module docstring before touching a tolerance there.

`noisy` exists because on a noiseless design the nugget MLE is pinned at the lower bound in a badly
conditioned regime — laGP and the port disagree there in the 8th digit while our likelihood is
*higher*. Those cases are tested as "no worse than laGP", not as equality.

Tolerances are not uniform on purpose. `s2` is computed as `1 + g - k' Ki k`, a cancellation, so
comparisons use `rtol=1e-8, atol=1e-11` rather than the 1e-12 used for `mean`/`llik`/`phi`.
Reassociation under `jit` and `vmap` costs a few ulps, so batch-invariance checks are 1e-6..1e-9.

`tests/snapshot.py` is the regression oracle, not a test module: `write <path>` records 124 arrays
from every entry point, `check <path>` asserts bit-identity. It is a **change detector, not a spec**:
when it reports a difference the gate is the parity suite plus rmse.

## bench/

`common.py` holds the Ackley designs, the `RlaGP` bridge and `timeit` (best of 3, after a warm-up
that absorbs JIT). `accuracy.py` measures predictive agreement, `timing.py` wall clock (JAX vs laGP at
1 thread and at `os.cpu_count()`), `scaling.py` cost per location as `n` grows, `plot.py` draws them.
Results and how to read them — including the traps — are in `ROADMAP.md` § Performance. The rules
that keep a benchmark honest:

- **Quote `fixed + per_location * nn`, never a ratio at one `nn`.** Both engines are affine in the
  location count.
- `accuracy.py` computes `darg` once and hands the same prior to both sides; otherwise it measures
  the RNG gap, not the port.
- `scaling.py` pins `close=n`; with laGP's default cap every search looks flat.
- `OPENBLAS_NUM_THREADS=1` is set at the top of `common.py`, before jax or R load. Keep it there.
- A 16-core laGP number is only trustworthy next to a 1-core number from the same node (~6x on a
  healthy node; ~1x means `omp.threads` did not take and the file is scrap).

## Performance

**Read `ROADMAP.md` § Performance before optimizing anything.** Short version: the workload is bound
by kernel-launch latency and the sequential greedy loop, not arithmetic — a faster `covar` kernel made
the pipeline slower. The local hyperparameter fit is the dominant cost.

## License

LGPL-2.0-or-later, inherited from laGP: laGP's DESCRIPTION declares a bare `License: LGPL`, which per
CRAN's *Writing R Extensions* means version 2 or later. This is a derivative work translated from
laGP's source, so relicensing to MIT/Apache is not available without Gramacy's permission. `NOTICE`
carries the attribution and citations; keep it accurate if modules are added.

The `jaxvlse` dependency is **GPL-2.0-only** (its test functions are translations of GPL-2 R sources).
Depending on it does not change lagpax's own license, but anything distributing lagpax together with
it is bound by the GPL-2.0. The solver in `vlse/optim/lbfgsb.py` is original code, so the clean fix is
to publish it under a permissive or LGPL license; see `ROADMAP.md` § Next.

## Gotchas

- A bare `uv sync` prunes the `tests` extra, and rpy2 then dies with
  `ModuleNotFoundError: No module named '_cffi_backend'`. Re-run `uv sync --extra tests`.
- `uv sync --extra tests` without `R_HOME` set fails building rpy2. Export it first.
- `uv.lock` is in `.gitignore`, so nothing is pinned across machines. The `rpy2>=3.5.17,<3.6` bound
  in `pyproject.toml` is the only guard — do not relax it while resolving something else.
- In a 16GB cloud container the whole suite in **one** pytest process can abort partway (a JAX
  runtime error or a fatal abort, in a different module each time); every module passes on its own.
  Run modules in separate processes there: `for f in tests/test_*.py; do uv run --no-sync pytest $f -q; done`.
  CI's runner does not show this.
- Piping a long background run through `tail` buffers everything until it exits. Let it write its
  CSV and read that instead.
- `lagpax/__init__.py` does `from .covar import covar`, which shadows the submodule: both
  `from lagpax import covar as m` and `import lagpax.covar as m` give you the *function*. Use
  `sys.modules["lagpax.covar"]`.
- Monkeypatching a function that `local._core` closes over does nothing — `_core` is jitted and the
  closure is not part of its cache key. Call `local._core.clear_cache()` between configurations.
- vlse reads its default tolerances as arrays created at import; pass `tol`/`ftol` explicitly (as
  the call sites do) rather than relying on them under x64.
- There is no `gh` CLI on the cluster. Check CI with
  `curl -s https://api.github.com/repos/davidesartor/lagpax/actions/runs?per_page=3`.
