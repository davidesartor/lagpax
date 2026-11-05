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

The model is laGP's: correlation `exp(-sum_k (x_k - x'_k)^2 / theta_k)`, `nugget` on the diagonal,
scale profiled out as `phi = resid' Ki resid`, Student-t predictive with `dof = n - p`. `p` is the
trend basis width, 0 under the default `trend="zero"`, where `resid` is `y` and `dof` is `n`.
`theta` is laGP's `d` (a squared lengthscale); `nugget` is laGP's `g`.

- `kernel.py` — `sqdist`, `correlation(X1, X2, theta)`, `correlation_matrix(X, theta, nugget)`,
  `dcorrelation`. Data first, parameters last, everywhere.
- `gp.py` — `GP(X, y, theta, nugget, mask=None, trend=)`, a NamedTuple caching `K`, `Ki`, `ldetK`,
  `KiZ`, `phi`, plus `F`, `b`, `Kfi`, `ldetF` for the trend. `Ki` is materialized (not just a
  Cholesky factor) because ALC needs it. `log_likelihood(prior)`, `predict(Xnew, full_cov=,
  nugget=)`, `append`, `replace`. `_profile` is the fit's inner-loop objective — one Cholesky, two
  triangular solves, no inverse. `_moments` returns the mean and the Student-t *scale*, which the
  criteria need; `Prediction` only ever carries the variance (`scale * dof / (dof - 2)`). Internals
  still say `d`/`g`/`Z` in places; the public names are `theta`/`nugget`/`y`.
- `_config.py` — `pytree_config` and `static()`: frozen dataclasses registered with
  `jax.tree_util.register_dataclass`. A `static()` field is pytree metadata (part of the compiled
  program, must be hashable); every other field is a child. `Param`, `Prior`, `LocalGP`, every
  criterion and every search are built this way. No equinox: plain JAX does it.
- `prior.py` — `Param(init, lower, upper, gamma, learn)` and `Prior(nugget=, theta= | lengthscale=)`.
  A prior holds **exactly one** of `theta` and `lengthscale`; that choice is its `space`, and the
  bounds and Gamma prior apply in that space (`theta = 2 l^2`). `to_theta`/`from_theta` convert.
  `Prior.from_data` is the host-side numpy/scipy port of `darg`/`garg`/`getDs`/`Igamma.inv` (R's
  `quantile` type 7 == numpy's default), computed in theta and mapped to `l` for
  `space="lengthscale"`, with the Gamma prior recentered there. Its subsample of a >1000-row design
  draws with an explicit `key`; a missing key is a `ValueError`, never a hidden draw. `learn` is
  static — `fit` branches on it in Python.
- `optim.py` — `bracketed_newton` for the nugget, `implicit_argmin`, and `r_ftol`. L-BFGS-B is
  `vlse.optim.minimise`, called directly with `tol=0.0, ftol=r_ftol(dtype)` — R's `optim` default
  `factr=1e7` times *float64's* epsilon and `pgtol=0`, in every dtype (scaling by float32's epsilon
  gives ~1.2 and every fit stops after one step). `bracketed_newton` takes one `dfun` returning
  `(d, d2)`, so the nugget fit gets both derivatives from a single factorization per iteration.
  Neither solver reproduces R's iterate trajectory; they find the same KKT point.
- `fit.py` — `fit_kernel`, `fit_nugget`, `fit_both` and `fit` dispatching on the prior's `learn`
  flags; all return `(GP, FitInfo(theta, nugget, iterations, converged))`. The kernel solve runs in
  the prior's space via `_kernel_llik`; in theta space that is exactly laGP's objective. `fit_both`
  is jmleGPsep's alternation and carries `(v, nugget)` through its `while_loop` rather than a GP, so
  the half-steps never factorize a `K` nobody reads. There is no conv=102 boundary reset: a start on
  the lower bound is simply optimized.
- `criteria.py` — `Criterion` subclasses `ALC` (port of `alcGPsep`), `NN`, `IECI(fmin, w=, nugget=)`
  (`ieciGPsep`), and `Custom` wrapping a plain function. `minimize` and `cached` are class
  attributes, read directly by the greedy loop and the searches. Number-valued settings
  (`IECI.fmin`, `IECI.w`) are leaves — moving `fmin` every BO step does not recompile; flags
  (`IECI.nugget`) are static. `alc` and `nn` are instances, so `lagpax.alc(gp, Xcand, Xref)` works.
  `as_criterion` turns a name or function into one.
- `search.py` — `Search` subclasses `Exhaustive`, `Ray(numrays=, iters=)`,
  `Continuous(numstart=, maxit=)`, all-static and hashable by value; `as_search` resolves names.
  Calling one takes `(criterion, gp, Xcand, Xref, taken, step, rect)` and returns one candidate
  index. `_block` picks "the offset-th *available* candidate" the way laGP's deletion from its
  candidate list does, by cumsum + `searchsorted`; an argsort there would dominate the search.
- `model.py` — `LocalGP` (`neighbors`, `init`, `candidates`, `criterion`, `search`, `trend`,
  `maxiter`) and `Emulator`. `LocalGP` is a pytree (sizes, `trend`, `maxiter` static; criterion and
  search as children); `LocalGP.fit(X, y, prior=, key=)` resolves the prior and returns an
  `Emulator` NamedTuple of `(model, X, y, prior)`, so the whole emulator is one pytree.
  `_core(emu, Xref, theta0, full_cov=)` is jitted with only `full_cov` static. `Emulator.local` is
  the per-location primitive; `__call__` is `jax.lax.map(batch_size=)` over it and `refine` swaps
  the criterion with `model.replace(criterion=...)` between passes. `_design` carries the
  candidate/design cross-covariance across greedy steps. `_buckets` groups the greedy steps by
  padded capacity — step `t` builds its GP at the next power of two above `t`, not at
  `neighbors` — so the loop traces a logarithmic number of bodies.

The public API is flat: `__init__.py` re-exports every name and pins them in `__all__`, so a new
public name goes in two places there. **Importing lagpax changes no global state.** Computation
follows the input's floating dtype; laGP parity needs float64, so `tests/conftest.py`,
`tests/snapshot.py` and `bench/common.py` turn `jax_enable_x64` on before any data exists, and so
must a user who wants laGP's numbers. Without it the pipeline runs in float32
(`test_transforms` pins that in a subprocess).

### Why the mask exists

Greedy design selection grows the local design one point at a time, but XLA needs fixed shapes.
`GP.mask` marks the active rows, and there are **four** invariants: masked rows/columns of `K` are set
to the identity, so `ldetK` gains 0; masked `y` entries are zeroed so `phi` is unaffected; masked `F`
rows are zeroed so the GLS solve ignores them; and the identity block in `Ki` is killed by masking
`k`/`kx` in `_moments`/`_predict`/`alc`. Break any one and the padded GP silently stops matching the
unpadded one — `tests/test_gp.py::test_mask_matches_a_smaller_design`,
`tests/test_criteria.py::test_alc_ignores_masked_rows` and the trend mask test in `tests/test_trend.py`
are the guards.

The invariants are exact but the padding *width* is not neutral: `alc` differs by ~1e-16 across
widths, which flips near-tied argmaxes and cascades through the greedy loop. **Judge a change here on
the parity suite and on rmse, never on whether `neighbors` is unchanged.**

### Gradients through the fit

`optim.implicit_argmin` is what makes `jax.grad` work through `fit`. Both solvers are
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

- laGP's `d`/`g` are `theta`/`nugget`; its `Xi` (1-based) is `neighbors` (0-based).
- `laGPsep` and `predGPsep` return the raw Student-t scale `s2`; lagpax always returns the variance.
  Compare `variance` with `s2 * df / (df - 2)`. `aGPsep`'s `var` is already the variance.
- `darg`/`garg` subsample the design when `n > 1000` using R's RNG, so parity tests stay under 1000.
- `dllikGPsep` is not exported from the laGP namespace; call it as `laGP:::dllikGPsep`. Same for
  `ieciGPsep`.
- laGP's ray search draws its starting points with R's `sample()`; lagpax's are deterministic, so
  `test_search` asserts design quality, not identity.

## API

```python
jax.config.update("jax_enable_x64", True)   # the caller's choice; needed for laGP parity
key = jax.random.key(0)

model = lagpax.LocalGP(
    neighbors=50, init=6,
    criterion="alc",        # "nn" | "alc" | lagpax.IECI(fmin=...) | a Criterion | a function
    search="exhaustive",    # "exhaustive" | "ray" | "continuous" | lagpax.Ray(numrays=...) | ...
    trend="zero",           # "zero" | "constant" | "linear"; "zero" is laGP's simple kriging
    candidates=None, maxiter=100,
)
emu = model.fit(X, y, prior=None, key=key)          # -> Emulator, a pytree
out = emu(Xnew, batch=256)                          # one local GP per row (aGPsep)
out = emu.local(Xref, theta0=None, full_cov=False)  # one local GP for a reference set (laGPsep)
out = jax.vmap(lambda x: emu.local(x[None]))(Xnew)  # ... batched yourself
out = emu.refine(Xnew, passes=("nn", "alc"))
emu = emu._replace(y=ynew)                          # same model and prior, new responses
out.mean, out.variance, out.dof, out.log_likelihood, out.covariance, out.neighbors, out.fit

prior = lagpax.Prior.from_data(X, y, key=key)                       # Gamma prior on theta
prior = lagpax.Prior.from_data(X, y, key=key, space="lengthscale")  # ... or on l = sqrt(theta/2)
prior = prior.replace(nugget=lagpax.Param(learn=True))

gp = lagpax.GP(X, y, theta, nugget, mask=None, trend="zero")
gp.log_likelihood(prior=None)            # MAP objective when a prior is passed
gp.predict(Xnew, full_cov=False, nugget=True)
gp.append(Xmore, ymore); gp.replace(theta=..., nugget=...)
gp, info = lagpax.fit(gp, prior)         # info: FitInfo(theta, nugget, iterations, converged)

lagpax.alc(gp, Xcand, Xref)              # criteria: (gp, Xcand, Xref) -> Float[Array, "ncand"]
lagpax.IECI(fmin=...)(gp, Xcand, Xref)
```

Load-bearing design points:

- **`LocalGP.fit` is the seam.** Resolving the prior is the only host-side numpy/scipy step;
  putting it up front is what lets everything downstream be pure JAX. Do not let prior resolution
  leak back into the per-location path. Hyperparameters are fitted per location when the emulator
  is called, not in `fit`.
- **Static vs leaf is a design decision per field.** Anything that selects code or sets a shape
  (`neighbors`, `init`, `candidates`, `trend`, `maxiter`, `IECI.nugget`, `Ray.numrays`,
  `Param.learn`, which of `theta`/`lengthscale` a Prior holds) is static or structural; anything
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
refinement** — `("nn", "alc")` picks neighbourhoods at fitted `theta` and cuts rmse ~1.7-2x on
anisotropic problems.

## tests/

`tests/conftest.py` provides the `R` fixture (rpy2 + laGP, skipping when either is unavailable) and the
shared designs: `small` (40x3), `medium` (600x4), `wide` (200x8), `noisy` (300x3 with observation
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
- Monkeypatching a function that `model._core` closes over does nothing — `_core` is jitted and the
  closure is not part of its cache key. Call `model._core.clear_cache()` between configurations.
- vlse reads its default tolerances as arrays created at import; pass `tol`/`ftol` explicitly (as
  the call sites do) rather than relying on them under x64.
- Under an enclosing trace, `jnp.sqrt(constant)` is a tracer, so `float(jnp.sqrt(...))` raises a
  ConcretizationTypeError — but only when the caller is inside `jit`/`while_loop`, which eager tests
  and the snapshot never are. Compute host-side constants with `math`/`numpy`.
- `some_jitted_fn._cache_size()` counts top-level dispatches only. A jitted function called inside
  another trace (`_core` inside the emulator's `lax.map`) is inlined and never touches that cache, so
  a recompilation test has to call it at top level (`emu.local`, not `emu(...)`).
- There is no `gh` CLI on the cluster. Check CI with
  `curl -s https://api.github.com/repos/davidesartor/lagpax/actions/runs?per_page=3`.
