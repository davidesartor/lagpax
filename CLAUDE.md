# CLAUDE.md

Operating manual for Claude Code in this repo, loaded every session: how the code works, how to run
it, and what will bite you. It states facts about the repo *as it is*.

- `ROADMAP.md` — status, the rules that govern new work, what was learned (every measurement lives
  there), and what is next. Read its "Rules" before changing anything numerical.
- `README.md` — the user-facing pitch.

Keep the three in sync and don't duplicate: a number belongs in `ROADMAP.md`, a how-to here.

## What this is

Local approximate Gaussian processes in JAX. It began as a port of the laGP R package (1.5.9,
separable/`GPsep` family) and keeps laGP's numerical core — verified value-for-value against R by
`tests/` — under its own API, which deliberately does not follow laGP's names or conventions. The
pitch is nonstationarity + differentiability + verified parity, not raw speed. Extracted from the
LocalVorcans prototype repo, which holds an older, independent local-GP stack that does **not**
implement the same math — don't import assumptions from there.

Box-constrained solves are delegated to `vlse.optim.minimise` (the `jaxvlse` package, same author),
whose SFU test functions also drive a parity module on real response surfaces. What is open is in
`ROADMAP.md` § Next.

### Before publishing

Full checklist in `ROADMAP.md`. The parts you must not get wrong from memory:

- **The PyPI placeholder ships as `0.0.1`, never `0.1.0`.** PyPI never lets a version number be
  reused and deleting a release does not free its filename. Plain `lagp` was already squatted by an
  unrelated project, so `lagpax` being free today proves nothing.
- **CI must fail on a nonzero skip count.** `.github/workflows/tests.yml` installs R + laGP 1.5-9 and
  sets `LAGPAX_STRICT=1`, which turns any skip into a nonzero exit — keep it set; it is the whole point.
- **Require `jaxvlse>=0.1.4` before releasing.** 0.1.4 makes `vlse.optim` Apache-2.0 and stops
  `import vlse.optim` from loading the GPL-2.0 test functions; earlier releases are GPL-2.0-only
  throughout. See "License" below.

## Commands

```bash
export R_HOME=$(R RHOME)                              # rpy2 needs it (`module load r/4.4.0` on the cluster)

uv sync --extra tests                                 # pytest + rpy2 (rpy2 must stay <3.6)
LAGPAX_STRICT=1 uv run --no-sync pytest tests -q      # parity suite vs R; a skip is a failure
uv run --no-sync pytest tests/test_fit.py -q          # one module
uv run --no-sync pytest tests/test_gp.py::test_mask_matches_a_smaller_design   # one test
uv run --no-sync pytest tests -q -k "alc and not mask" # by name
uv run --no-sync python -m tests.snapshot check <file.npz>   # did anything move?

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
the parity suite targets — check out the tag.

## src/lagpax

Formatted by `ruff format` at 100 columns (`[tool.ruff]` in `pyproject.toml`). Written to read
without comments and to stay tight: descriptive names, type hints everywhere
(jaxtyping shapes on every array), docstrings on what is public, an inline comment only where a
name cannot carry the reason, and **code used once is inlined** — a helper exists only if it has
two or more callers, is public API, or is a unit a test exercises directly (`design_buckets`,
`_ray_ends`, `_golden_section_maximize`, `pairwise_distance_summary`). Prefer renaming over
commenting, inlining over a one-call helper.

The model is laGP's: correlation `exp(-sum_k (x_k - x'_k)^2 / s_k)` with `s` the
`squared_lengthscales` (laGP's `d`), a `nugget` on the diagonal (laGP's `g`), the scale profiled out
as `residual_sum_of_squares = r' K^-1 r`, and a Student-t predictive with
`degrees_of_freedom = active_count - trend_size`.

- `_pytree.py` — `pytree_dataclass` and `static()`: frozen dataclasses registered with
  `jax.tree_util.register_dataclass`. A `static()` field is pytree metadata (part of the compiled
  program, must be hashable); every other field is a child. Every config and state object in the
  package is one of these or a NamedTuple. No equinox.
- `kernel.py` — `squared_distances`, `correlation(left, right, squared_lengthscales)`,
  `correlation_with_nugget`. Data first, parameters last, everywhere.
- `priors.py` — `GammaPrior(shape, rate)` (`with_mass_below` is laGP's default rule),
  `Param(initial, lower, upper, prior, learn)`, and `Prior(nugget=, squared_lengthscale= |
  lengthscale=)`. A prior holds exactly one kernel `Param`; that choice is its `kernel_space`, and
  bounds and Gamma prior apply in that space (`s = 2 l^2`). `Prior.from_data` is the host-side
  numpy port of laGP's darg/garg (`default_squared_lengthscale`, `default_nugget`,
  `pairwise_distance_summary`), mapped to `l` for `kernel_space="lengthscale"`. Its >1000-row
  subsample needs an explicit `key`; without one it raises. `learn` is static.
- `gaussian_process.py` — `Hyperparameters`, `Prediction`, and `GaussianProcess`, a pytree
  dataclass built by `GaussianProcess.from_data(inputs, outputs, hyperparameters, active, trend)`
  that caches `inverse_correlation` (ALC needs it), `log_det_correlation`,
  `residual_sum_of_squares` and the trend's GLS pieces. `from_data` and `log_likelihood_at` (the
  fit's objective) share one `_factorize`: masked Cholesky, GLS trend, residual sum of squares, all
  by Cholesky solves; only `from_data` then forms the explicit inverse. `predictive_mean_and_scale` (per point or
  `full_covariance`) gives the Student-t *scale*; `predict` and the criteria both use it, and
  `Prediction` only ever carries the variance.
- `optimize.py` — `with_implicit_gradient` (the derivative of an argmin, attached to a solver's
  answer), `newton_maximize_on_interval` (the nugget solver), and `R_OPTIM_RELATIVE_TOLERANCE`,
  R's `factr=1e7` times *float64's* epsilon in every dtype (float32's would make it ~1.2 and stop
  every fit after one step). L-BFGS-B is `vlse.optim.minimise`, called directly.
- `fitting.py` — one `fit_hyperparameters(gp, prior)`: L-BFGS-B on the kernel parameter (in the
  prior's space), Newton on the nugget, or laGP's jmleGPsep alternation of the two when both are
  learned — chosen by the prior's `learn` flags. Returns `(GaussianProcess, FitResult(
  squared_lengthscales, nugget, iterations, converged))`. Hyperparameters that are not learned
  keep their exact values and their upstream gradients (the implicit gradient only covers learned
  coordinates). `_kernel_log_posterior` keeps laGP's summation order (likelihood, then kernel
  prior, then nugget prior), so squared-lengthscale fits are laGP's. Tests reach the
  single-parameter cases through `tests/helpers.py` (`fit_kernel_only`, `fit_nugget_only`,
  `fit_both`).
- `criteria.py` — `Criterion` subclasses `NearestNeighbor`, `ActiveLearningCohn`,
  `IntegratedExpectedConditionalImprovement(best_observed, weights=, include_nugget=)` and
  `CustomCriterion`. Class attributes `lower_is_better` and `reuses_candidate_correlation`.
  `variance_reduction` is the shared rank-one update. Names ("nn", "alc") are resolved in
  `LocalGP.__post_init__`, the only place that accepts them.
- `search.py` — `Search` subclasses `ExhaustiveSearch`, `RaySearch(rays=, golden_section_steps=)`,
  `ContinuousSearch(starts=, max_iterations=)`, all static and hashable by value (names resolved in
  `LocalGP.__post_init__`). Called as
  `(criterion, gp, candidates, references, taken, step, bounding_box, candidate_correlation)`.
  `untaken_by_rank` reproduces laGP's "k-th still-available candidate" by cumsum + searchsorted.
- `local.py` — `LocalGP(neighbors, initial_neighbors, candidates, criterion, search, trend,
  max_iterations)` and `Emulator(model, inputs, outputs, prior)`, both pytrees.
  `_predict_local(emulator, references, starting_squared_lengthscales, full_covariance=)` is the
  jitted core; `Emulator.predict_local` is the per-point primitive, `__call__` is
  `lax.map(batch_size=)` over it, `refine` swaps the criterion between passes. Inside
  `_predict_local` the greedy loop carries a `DesignState(chosen, active, taken,
  candidate_correlation)`, and `design_buckets` groups its steps by padded GP size (next power of
  two), so it traces logarithmically many loop bodies.

The public API is flat: `__init__.py` re-exports every name and pins them in `__all__`.
**Importing lagpax changes no global state.** Computation follows the input's floating dtype;
laGP parity needs float64, so `tests/conftest.py`, `tests/snapshot.py` and `bench/common.py` turn
`jax_enable_x64` on before any data exists, and so must a user who wants laGP's numbers. Without
it the pipeline runs in float32 (`test_transforms` pins that in a subprocess).

### Why the mask exists

Greedy design selection grows the local design one point at a time, but XLA needs fixed shapes.
`GaussianProcess.active` marks the rows in the model, and there are **four** invariants: inactive
rows/columns of the correlation are the identity, so `log_det_correlation` gains 0; inactive
outputs are zeroed so `residual_sum_of_squares` is unaffected; inactive trend rows are zeroed so
the GLS solve ignores them; and the identity block in `inverse_correlation` is killed by masking
the cross-correlations in `predictive_mean_and_scale`/`predict`/`variance_reduction`. Break any one and the padded GP silently stops matching the
unpadded one — `tests/test_gp.py::test_mask_matches_a_smaller_design`,
`tests/test_criteria.py::test_alc_ignores_masked_rows` and the trend mask test in `tests/test_trend.py`
are the guards.

The invariants are exact but the padding *width* is not neutral: `alc` differs by ~1e-16 across
widths, which flips near-tied argmaxes and cascades through the greedy loop. **Judge a change here on
the parity suite and on rmse, never on whether `neighbors` is unchanged.**

### Gradients through the fit

`optimize.with_implicit_gradient` is what makes `jax.grad` work through the fits. Both solvers are
`lax.while_loop`s with no reverse-mode rule, so the answer is computed under `stop_gradient` and the
derivative attached afterwards: one Newton step from the frozen solution carries exactly the
implicit-function term, and subtracting its own `stop_gradient` leaves the **value bit-identical**.
A hyperparameter on an active bound is frozen and its gradient is exactly zero.

The derivative is exact at the true argmin and is evaluated at the solver's, so its accuracy is the
solver's stationarity. Do not check it by finite-differencing the solver; `tests/test_diff.py`
compares against an argmin root-found with scipy instead. A finite difference of the *whole pipeline*
needs a step below the scale at which a perturbation changes which iterate L-BFGS-B stops on — `1e-5`
on the test designs (see `ROADMAP.md` § What we learned).

### laGP conventions the parity tests have to bridge

- laGP's `d`/`g` are `squared_lengthscales`/`nugget`; its `Xi` (1-based) is `neighbors` (0-based);
  its `ab` pairs are `GammaPrior(shape, rate)`, with `ab=c(0,0)` meaning `prior=None`.
- `laGPsep` and `predGPsep` return the raw Student-t scale `s2`; lagpax always returns the variance.
  Compare `variance` with `s2 * df / (df - 2)`. `aGPsep`'s `var` is already the variance.
- `darg`/`garg` subsample the design when `n > 1000` using R's RNG, so parity tests stay under 1000.
- `dllikGPsep` and `ieciGPsep` are not exported from the laGP namespace; call them with `laGP:::`.
- laGP's ray search draws its starting points with R's `sample()`; lagpax's are deterministic, so
  `test_search` asserts design quality, not identity.
- laGP computes the default Gamma rate as `Igamma.inv(a, 0.95 * gamma(a), ...)`, lagpax as
  `gammaincinv(a, 0.95)`; they agree to an ulp, which is all the fitted values move by.

## API

```python
jax.config.update("jax_enable_x64", True)   # the caller's choice; needed for laGP parity
key = jax.random.key(0)

model = lagpax.LocalGP(
    neighbors=50, initial_neighbors=6,
    criterion="alc",        # "nn" | "alc" | lagpax.IntegratedExpectedConditionalImprovement(...) | ...
    search="exhaustive",    # "exhaustive" | "ray" | "continuous" | lagpax.RaySearch(rays=...) | ...
    trend="zero",           # "zero" | "constant" | "linear"; "zero" is laGP's simple kriging
    candidates=None, max_iterations=100,
)
emulator = model.fit(inputs, outputs, prior=None, key=key)   # -> Emulator, a pytree
prediction = emulator(new_inputs, batch_size=256)           # one local GP per row (aGPsep)
prediction = emulator.predict_local(references, full_covariance=False)   # laGPsep
prediction = jax.vmap(lambda x: emulator.predict_local(x[None]))(new_inputs)
prediction = emulator.refine(new_inputs, passes=("nn", "alc"))
emulator = emulator._replace(outputs=new_outputs)
prediction.mean, .variance, .degrees_of_freedom, .log_likelihood, .covariance, .neighbors, .fit

prior = lagpax.Prior.from_data(inputs, outputs, key=key)                          # on s
prior = lagpax.Prior.from_data(inputs, outputs, key=key, kernel_space="lengthscale")  # on l
prior = prior.replace(nugget=lagpax.Param(learn=True))

gp = lagpax.GaussianProcess.from_data(inputs, outputs, lagpax.Hyperparameters(s, nugget))
gp.log_likelihood(prior=None); gp.predict(new_inputs, full_covariance=False, include_nugget=True)
gp.with_hyperparameters(h); gp.with_observations(more_inputs, more_outputs)
gp, result = lagpax.fit_hyperparameters(gp, prior)   # fits what prior.*.learn marks

lagpax.ActiveLearningCohn()(gp, candidates, references)   # criteria: -> one score per candidate
```

Load-bearing design points:

- **`LocalGP.fit` is the seam.** Resolving the prior is the only host-side numpy/scipy step;
  putting it up front is what lets everything downstream be pure JAX. Do not let prior resolution
  leak back into the per-location path. Hyperparameters are fitted per location when the emulator
  is called, not in `fit`.
- **Static vs leaf is a design decision per field.** Anything that selects code or sets a shape
  (`neighbors`, `initial_neighbors`, `candidates`, `trend`, `max_iterations`, `include_nugget`,
  `rays`, `Param.learn`, which kernel space a Prior is in) is static or structural; anything
  numeric is a leaf. Getting it wrong is silent: a number marked static recompiles on every change,
  a flag left as a leaf fails with a ConcretizationTypeError the first time Python branches on it.
- **No hidden randomness.** Every random draw takes a `jax.random` key. Where R would use its global
  RNG and no key was passed, raise — do not fall back to a default seed.
- **`refine`'s pass loop stays a Python loop** — passes may name different criteria that trace to
  different functions. It unrolls at trace time and still jits as one graph.
- **`candidates`** bounds the pool for every search, defaulting to `10*(1000+neighbors)` for ray and
  continuous and `1000+neighbors` for exhaustive, matching laGP.
- **Names are sugar, objects are the API.** `"alc"`, `"ray"` normalize to objects on construction;
  nothing downstream sees a string.

Beyond laGP: **trend models** (`trend="constant"`/`"linear"`, sold as interval calibration, not a
better mean — `tests/test_trend.py` tests coverage), **gradients through fitting** with a compilable
pipeline, and **a prior on the conventional lengthscale**. Easy to under-use: **iterated
refinement** — `("nn", "alc")` picks neighbourhoods at fitted squared lengthscales and cuts rmse
~1.7-2x on anisotropic problems.

## tests/

`tests/conftest.py` provides the `R` fixture (rpy2 + laGP, skipping when either is unavailable) and the
shared designs, and `tests/helpers.py` the constructors the modules share (`make_gp`, `lagp_prior`
from laGP's range/ab arguments, `log_likelihood_gradient`). The designs: `small` (40x3), `medium` (600x4), `wide` (200x8), `noisy` (300x3 with observation
noise).

- **Parity against R:** `test_covar` (kernel), `test_gp`, `test_prior`, `test_fit`, `test_criteria`,
  `test_ieci`, `test_model`, and `test_vlse`.
- **Property tests** (no laGP equivalent): `test_transforms`, `test_search`, `test_trend`,
  `test_diff`, plus the lengthscale-space tests in `test_prior` and `test_fit`.

`test_vlse` runs `jmleGPsep` and `aGPsep` on eight SFU benchmark surfaces from `vlse` (Branin,
Hartmann3/6, Michalewicz, Ackley, Rosenbrock, Goldstein-Price, Shekel), unit-cube inputs, standardized
response. The random-feature designs are smooth by construction; these are where a kernel fit meets
narrow valleys, large dynamic range and many basins. Assertions are tiered by surface — read its
module docstring before touching a tolerance there.

`noisy` exists because on a noiseless design the nugget fit is pinned at the lower bound in a badly
conditioned regime — laGP and lagpax disagree there in the 8th digit while our likelihood is
*higher*. Those cases are tested as "no worse than laGP", not as equality.

Tolerances are not uniform on purpose. The variance is `1 + g - k' Ki k`, a cancellation, so
comparisons use `rtol=1e-8, atol=1e-11` rather than the 1e-12 used for means and log-likelihoods.
Reassociation under `jit` and `vmap` costs a few ulps, so batch-invariance checks are 1e-6..1e-9.

`tests/snapshot.py` is the regression oracle, not a test module: `write <path>` records every
user-visible output, `check <path>` asserts bit-identity and lists keys that are new or missing. It
is a **change detector, not a spec**: when it reports a difference the gate is the parity suite plus
rmse. **A snapshot is only comparable within one machine**: XLA's CPU reductions depend on the core
count, so a baseline from another container moves by ~1e-11 with no code change. Write the baseline
from the old code in the same environment before judging a change.

## bench/

`common.py` holds the Ackley designs, the `RlaGP` bridge and `timeit` (best of 3, after a warm-up
that absorbs JIT). `accuracy.py` measures predictive agreement, `timing.py` wall clock (JAX vs laGP at
1 thread and at `os.cpu_count()`), `scaling.py` cost per location as `n` grows, `plot.py` draws them.
Results and how to read them — including the traps — are in `ROADMAP.md` § Performance. The rules
that keep a benchmark honest:

- **Quote `fixed + per_location * nn`, never a ratio at one `nn`.** Both engines are affine in the
  location count.
- `accuracy.py` builds the default prior once and hands the same one to both sides; otherwise it
  measures the RNG gap, not the implementation.
- `scaling.py` pins `candidates=n`; with laGP's default cap every search looks flat.
- `OPENBLAS_NUM_THREADS=1` is set at the top of `common.py`, before jax or R load. Keep it there.
- A 16-core laGP number is only trustworthy next to a 1-core number from the same node (~6x on a
  healthy node; ~1x means `omp.threads` did not take and the file is scrap).

## Performance

**Read `ROADMAP.md` § Performance before optimizing anything.** Short version: the workload is bound
by kernel-launch latency and the sequential greedy loop, not arithmetic — a faster correlation kernel
made the pipeline slower. The local hyperparameter fit is the dominant cost.

## License

LGPL-2.0-or-later, inherited from laGP: laGP's DESCRIPTION declares a bare `License: LGPL`, which per
CRAN's *Writing R Extensions* means version 2 or later. The numerical core is a translation of laGP's
source, so this stays a derivative work under any API, and relicensing to MIT/Apache is not available
without Gramacy's permission. `NOTICE` carries the attribution and citations; keep it accurate if
modules are added.

`jaxvlse` is mixed from 0.1.4: `vlse.functions` is GPL-2.0-only (translations of the SFU R sources),
`vlse.optim` is Apache-2.0, and the package root loads `vlse.functions` lazily. lagpax's runtime code
imports only `vlse.optim` — keep it that way; the test functions belong in `tests/` only. Apache-2.0
code combines with LGPL-2.0-or-later under its version-3 terms. Releases up to 0.1.3 are GPL-2.0-only
as a whole, which is why the floor has to move to 0.1.4 once it is on PyPI.

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
- Monkeypatching a function that `local._predict_local` closes over does nothing — it is jitted and
  the closure is not part of its cache key. Call `local._predict_local.clear_cache()` between
  configurations.
- vlse reads its default tolerances as arrays created at import; pass `tol`/`ftol` explicitly (as
  the call sites do) rather than relying on them under x64.
- Under an enclosing trace, `jnp.sqrt(constant)` is a tracer, so `float(jnp.sqrt(...))` raises a
  ConcretizationTypeError — but only when the caller is inside `jit`/`while_loop`, which eager tests
  and the snapshot never are. Compute host-side constants with `math`/`numpy`.
- `some_jitted_fn._cache_size()` counts top-level dispatches only. A jitted function called inside
  another trace (`_predict_local` inside the emulator's `lax.map`) is inlined and never touches that
  cache, so a recompilation test has to call it at top level (`emulator.predict_local`, not
  `emulator(...)`).
- There is no `gh` CLI on the cluster. Check CI with
  `curl -s https://api.github.com/repos/davidesartor/lagpax/actions/runs?per_page=3`.
