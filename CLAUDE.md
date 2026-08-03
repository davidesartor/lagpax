# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Three documents, with distinct jobs — keep them in sync, don't duplicate across them:

- **`CLAUDE.md`** (this file) — how the repo works, why things are the way they are, what the target
  API is, and the capability inventory. Loaded into context every session.
- **`ROADMAP.md`** — the execution plan: phases, acceptance criteria, transcribed laGP reference
  algorithms. Written to be handed to an agent working unsupervised.
- **`README.md`** — the user-facing pitch. Currently stale in places; see the checklist below.

## Before publishing — do these first

Full checklist with commands is in `ROADMAP.md`. The parts you must not get wrong from memory:

- **The PyPI placeholder ships as `0.0.1`, never `0.1.0`.** PyPI never lets a version number be
  reused and deleting a release does not free its filename, so burning `0.1.0` on a stub costs the
  real first release its natural number. Reserving the name matters: plain `lagp` was already
  squatted by an unrelated LLM-proxy project, so `lagpax` being free today proves nothing.
- **Land the API rename before `0.1.0`.** See "## API" for the target and `ROADMAP.md` Phase 0 for
  the procedure. The stub commits to nothing, so the rename is free until the first real release and
  expensive after it.
- **CI must fail on a nonzero skip count.** The parity suite is the credibility artifact, and without
  R + laGP in the runner it goes green having tested almost nothing. `.github/workflows/tests.yml`
  installs R + laGP and sets `LAGPAX_STRICT=1`, which turns any skip into a nonzero exit — keep that
  env var set, it is the whole point of the job.
- **Do not weaken the license to MIT/Apache.** See "License" below.

## What this is

A JAX port of the laGP R package (1.5.9), separable/`GPsep` family. Bit-parity with R is the design
goal and is enforced by `tests/`. Extracted from the LocalVorcans prototype repo, which still holds
an older, independent local-GP/Bayesian-optimization stack that does **not** implement the same math
— don't import assumptions from there.

The pitch is nonstationarity + differentiability + verified parity, not raw speed. See README.

## Commands

```bash
module load r/4.4.0 && export R_HOME=$(R RHOME)      # needed for rpy2 in tests/benchmarks

uv sync --extra tests                                 # pytest + rpy2 (rpy2 must stay <3.6)
uv run --no-sync pytest tests -q                      # parity suite vs R
LAGPAX_STRICT=1 uv run --no-sync pytest tests -q      # ... and a skip is a failure
uv run --no-sync pytest tests/test_mle.py -q          # one module
uv run --no-sync pytest tests/test_gp.py::test_mask_matches_a_smaller_design   # one test
uv run --no-sync pytest tests -q -k "alc and not mask" # by name

uv sync --extra bench                                 # + matplotlib/pandas; NOT in the tests extra
uv run --no-sync -m bench.accuracy                    # JAX vs laGP predictions -> bench/accuracy.csv
uv run --no-sync -m bench.timing                      # wall clock -> bench/timing.csv
uv run --no-sync -m bench.plot                        # -> bench/benchmark.png
```

`--no-sync` on every `uv run` is load-bearing: a bare `uv run` re-syncs the default env and prunes
whichever extra is currently installed. `tests` and `bench` are separate extras and each `uv sync`
drops the other, so switching between the suite and the benchmarks means re-syncing.

Without R + laGP reachable, the `R` fixture skips and only the property-test modules run. A green
`pytest` proves nothing unless R was loaded, so prefix with `LAGPAX_STRICT=1` and a skip becomes a
nonzero exit rather than something to notice in the summary. CI sets it. `LAGPAX_R_LIB` overrides the
`~/R/library` search path; empty means "use R's own default library", which is what CI needs.

## src/lagpax

Parameterization is laGP's throughout: correlation `exp(-sum_k (x1_k - x2_k)^2 / d_k)`, nugget `g` on
the diagonal, scale profiled out as `phi = resid' Ki resid`, Student-t predictive with `df = n - p`.
`p` is the trend basis width and is 0 under the laGP-equivalent default `trend="zero"`, where `resid`
is `Z` and `df` is `n`.

- `covar.py` — `distance` (squared euclidean), `covar`, `covar_symm`, `dcovar`.
- `gp.py` — `GP` NamedTuple caching `K`, `Ki`, `ldetK`, `KiZ`, `phi`, plus `F`, `b`, `Kfi`, `ldetF`
  for the trend. `Ki` is materialized (not just a Cholesky factor) because ALC needs it. `_llik` adds
  the Gamma log-priors when `dab`/`gab` are non-zero. Gradients come from `jax.grad`, not
  hand-written derivatives.
- `prior.py` — host-side numpy/scipy port of `darg`/`garg`/`getDs`/`Igamma.inv`, behind
  `Prior.from_design`. Deliberately *not* jitted: it runs once per call and R's `quantile` type 7 ==
  numpy's default.
- `optim.py` — `lbfgsb` (projected L-BFGS, Armijo on the projected arc, R's `factr`/`pgtol` stopping
  rules), `bracketed_newton` for the nugget, and `implicit_argmin`. Neither solver reproduces R's
  iterate trajectory; they find the same KKT point. Tests assert that, plus "our llik is no worse
  than laGP's".
- `mle.py` — `mle_d` (with laGP's `0.9*dmin + 0.1*dmax` boundary reset, conv=102), `mle_g`, `jmle`,
  and `fit` dispatching on the prior's `estimate` flags.
- `criteria.py` — `nn`, `alc` (port of `alcGPsep`) and the `ieci(fmin=, w=, nonug=)` factory. A
  `Criterion` carries `minimize`, so the greedy loop never hardcodes argmax vs argmin.
- `search.py` — `exhaustive`, `ray`, `continuous`. All three take
  `(criterion, gp, Xcand, Xref, taken, step, rect)` and return one candidate index. `_block` picks
  "the offset-th *available* candidate" the way laGP's deletion from its candidate list does, by
  cumsum + `searchsorted` on the distance-ordered candidates; an argsort there would be
  `O(ncand log ncand)` at every greedy step and would dominate the search.
- `local.py` — `Local`, `Conditioned`, `lagp`/`agp`. `_core` is jitted with a `_Cfg` NamedTuple as the
  static config; `predict` maps it over predictive locations with `jax.lax.map(batch_size=)`.

The public API is flat: `__init__.py` re-exports every name and pins them in `__all__`, so a new
public function has to be added in two places there. Importing the package has a global side effect —
`jax.config.update("jax_enable_x64", True)` runs at import time, for the whole process.

### Why the mask exists

Greedy ALC design selection grows the local design one point at a time, but XLA needs fixed shapes.
`GP.mask` marks the active rows, and there are **four** invariants, not three: masked rows/columns of
`K` are set to the identity, so `ldetK` gains 0; masked `Z` entries are zeroed so `phi` is unaffected;
masked `F` rows are zeroed so the GLS solve ignores them; and the identity block in `Ki` is killed by
masking `k`/`kx` in `_predict`/`alc`. Break any one and the padded GP silently stops matching the
unpadded one — `tests/test_gp.py::test_mask_matches_a_smaller_design`,
`tests/test_alc.py::test_alc_ignores_masked_rows` and the trend mask test in `tests/test_trend.py`
are the guards.

### Gradients through the fit

`optim.implicit_argmin` is what makes `jax.grad` work through `mle_d`/`mle_g`/`jmle`. Both solvers are
`lax.while_loop`s with no reverse-mode rule, so the answer is computed under `stop_gradient` and the
derivative attached afterwards: one Newton step from the frozen solution carries exactly the
implicit-function term, and subtracting its own `stop_gradient` leaves the **value bit-identical** —
which is why the snapshot oracle still passes. A hyperparameter on an active bound, or one produced
by the conv=102 reset, is frozen and its gradient is exactly zero.

Do not check this gradient by finite-differencing the solver. `lbfgsb` stops on R's `factr`, so it
returns ~8 significant digits and a central difference carries `|x| * 1e-8 / 2h` of noise — ~7e-5, and
tightening `factr` to 1e3 only moves it to 6.8e-5. `tests/test_diff.py` compares against an argmin
obtained by root-finding `grad llik = 0` with scipy instead, which reaches ~1e-6.

`jit` preserves the optimizer trajectory (1e-9); `vmap` does not. A batched `while_loop` shares its
trip count, so one ulp of likelihood difference flips the relative-change stopping test and the
answers separate by `sqrt(factr*eps)` ~ 5e-5 while the likelihood reached agrees to `factr*eps` =
2.2e-9. Compare the objective, not the argmin, across batching.

### laGP conventions worth remembering

- `laGPsep` returns raw `s2`; `aGPsep` returns `var = s2 * df/(df-2)`. `Prediction` carries both.
- `Xi` is 1-based in R, 0-based here.
- `darg`/`garg` subsample the design when `n > 1000` using R's RNG, so parity tests stay under 1000.
- `dllikGPsep` is not exported from the laGP namespace; call it as `laGP:::dllikGPsep`.

## API

**Shipped in `539c740`.** The names previously mirrored laGP's C entry points — `new_gpsep`,
`mle_gpsep_d`, `alc_gpsep`, `agpsep`. That was the right call while parity was the only goal, and it
did not survive the roadmap: laGP's own naming already broke under the same pressure, since
`alcGPsep`, `alcrayGPsep`, `alcoptGPsep` are one criterion crossed with three searches spelled as
three functions, and MSPE, IECI and Fisher would make that grid twelve. The rename happened while
only a `0.0.1` stub was on PyPI, so there are no aliases and no deprecation window. What follows is
the current surface.

### 1. The GP

`GP` stays a NamedTuple pytree and gains methods, instead of free functions carrying a type suffix.
`sep` disappears everywhere: separable is the only family shipping, and isotropic is `d` broadcast.

```python
gp = lagpax.GP(X, Z, d, g, mask=None)
gp.llik(prior=None)                      # MAP when a prior is passed, MLE otherwise
gp.predict(XX, lite=True, nugget=True)
gp.update(Xnew, Znew)
gp.at(d=..., g=...)                      # was newparams_gpsep
```

### 2. Priors

`darg`/`garg` are R argument-checking names. They become one object holding both, because every call
site needs both:

```python
prior = lagpax.Prior.from_design(X, Z, rng=None)          # -> Prior(d=Range(...), g=Range(...))
prior = prior.at(g=lagpax.Range(start=1e-6, estimate=False))
```

Resolving the prior is the only host-side numpy/scipy step in the whole pipeline. Making it an
explicit up-front argument is what lets everything downstream be pure JAX — it is the seam the
compilable pipeline rests on, not a cosmetic change. It is also the slot block-bootstrap subsampling
plugs into later, and the thing `bench/accuracy.py` shares between the two sides by hand.

### 3. Criteria

Uniform signature, `(gp, Xcand, Xref) -> Float[Array, "ncand"]`. Anything with that signature is a
criterion, which is the "build your own" story the roadmap promises:

```python
lagpax.alc(gp, Xcand, Xref)
lagpax.nn(gp, Xcand, Xref)
lagpax.ieci(fmin=...)(gp, Xcand, Xref)   # criteria needing config are factories
```

A `Criterion` also carries `minimize`, because IECI is minimized where ALC is maximized. The greedy
loop reads that flag; it does not hardcode a direction. Searches follow the same rule — one needing
configuration is a `functools.partial` of the function in `lagpax.search`, not a new name.

### 4. The local scheme

Configuration separate from data; criterion separate from search.

```python
scheme = lagpax.Local(
    start=6, end=50,
    criterion="alc",        # "nn" | "alc" | lagpax.ieci(fmin=...) | any callable
    search="exhaustive",    # "exhaustive" | "ray" | "continuous"
    trend="zero",           # "zero" | "constant" | "linear"; "zero" reproduces laGP
    close=None, maxit=100,
)
emu = scheme.condition(X, Z, prior=None, rng=None)
out = emu.predict(XX, batch=64)
out = emu.refine(XX, passes=("nn", "alc"))
```

Plus one-shot wrappers for the common case, thin over the above:

```python
lagpax.agp(X, Z, XX, **opts)       # whole dataset
lagpax.lagp(X, Z, Xref, **opts)    # one location — data first, matching agp
```

Why split it: `criterion x search` is a grid, not a list of methods, and `method="alcray"` is how
laGP ended up with its name explosion. `close` still bounds the candidate set for every search, but
it defaults to `10*(1000+end)` for ray and continuous and `1000+end` for exhaustive, matching laGP.
`condition` resolves the prior once, ahead of any JAX work. `refine` replaces `agpsep_seq`'s
`**kwargs` forwarding, which is exactly where the `d`-resolution bug lived; its pass loop stays a
Python loop rather than a `lax.scan`, because passes may name different criteria that trace to
different functions — it unrolls at trace time and still jits and differentiates as one graph.

### 5. Output

One `Prediction` type for both entry points, ending laGP's split where `laGPsep` returns raw `s2` and
`aGPsep` returns `s2 * df/(df-2)`:

```python
out.mean, out.s2, out.df
out.var          # property: s2 * df / (df - 2)
out.Xi           # local design indices, 0-based
out.fit          # Fit(d, g, its, conv) — a typed struct, not the current dict
```

The name-by-name rename table, the module layout it lands in, and the snapshot-first procedure for
executing it are in `ROADMAP.md` Phase 0.

### What leaves the public surface

`lbfgsb`, `bracketed_newton`, `implicit_argmin`, `covar_symm`, `dcovar`, `igamma_inv`, `get_ds`,
`dllik_d`, `dllik_g`, the `mle_*`/`jmle` entry points and the three searches. All stay importable from
their own modules — the parity suite compares values, not names, and `tests/` reaches into
`lagpax.optim` and `lagpax.search` directly — they just are not part of a 30-name `__all__` that has
to be edited in two places for every addition. The public surface is fourteen names: `GP`, `Prior`,
`Range`, `Prediction`, `Fit`, `Local`, `agp`, `lagp`, `fit`, `alc`, `ieci`, `nn`, `distance`, `covar`.

`jax.config.update("jax_enable_x64", True)` at import stays. A process-global side effect from
importing a library is rude, but every tolerance in `tests/` assumes it and laGP is float64
throughout.

## Roadmap

`ROADMAP.md` is the execution plan — phases, file-by-file design, acceptance criteria, transcribed
laGP reference algorithms — written to be handed to an agent working unsupervised. This section is
the capability inventory it derives from; keep the two in sync.

What gets ported is laGP's **capability list, not its API**. The R surface is shaped by C-handle
bookkeeping, two parallel type families and R's optimizer set, none of which is worth carrying into
JAX. Below is what laGP can actually do, and where the port stands on each.

Testing strategy is deliberately deferred — build first. Note only that parity with R stops being a
whole-package property once anything divergent ships, and becomes a property of the laGP-equivalent
configuration.

### Working with a GP directly

Everything the local scheme is assembled from is also usable on its own, which is how a user builds
a criterion or a workflow laGP does not have.

- **Build a GP on a design and query it** — covariance, log likelihood, predictive equations.
  *Shipped.*
- **Design-derived priors and ranges for the lengthscale and nugget**, so a fit on a few dozen points
  is well posed without the user inventing bounds. Usable standalone, and worth reusing across calls
  since it subsamples with an RNG. *Shipped.*
- **Score candidate points by ALC against a reference set**, outside any greedy loop. *Shipped.*
- **Derivatives of the likelihood and of the ALC criterion.** laGP ships hand-written derivative
  routines for both; here they fall out of autodiff, so they cost nothing to maintain and extend to
  anything else differentiable. *Shipped.*
- **Append observations to an existing GP.** *Shipped*, but it rebuilds rather than applying the
  rank-1 update laGP uses — correct, and a performance item rather than a capability gap.
- **Fixed-shape padded designs**, so a local design that grows one point at a time still compiles and
  batches. Not a laGP concept at all; it is what makes the greedy loop work under XLA. *Shipped.*

### Choosing the local design

The core of the method: given a predictive location, pick a few dozen points out of a huge dataset to
build a tiny exact GP on. Which points you pick is the whole ballgame.

- **Nearest neighbours** — take the closest points. Cheapest, and a surprisingly strong baseline.
  *Shipped.*
- **ALC (Active Learning Cohn)** — grow the design one point at a time, each time adding whichever
  candidate most reduces predictive variance at the location being predicted. Better designs than
  nearest-neighbour because it will reach for points that brace the prediction rather than crowd it.
  laGP's default and the reason the method works. *Shipped.*
- **MSPE** — ALC's variance reduction plus the extra error incurred from re-estimating the
  lengthscale once the point is added. Slightly better designs at appreciably more cost. *Missing;
  low priority — the accuracy gain over ALC is small.*
- **IECI (integrated expected conditional improvement)** — score a candidate by how much it improves
  the prospect of finding an optimum rather than how much it reduces variance. The criterion to use
  when the emulator serves optimization instead of prediction. *Shipped* as the `ieci(fmin=)`
  factory. It is minimized where ALC is maximized, which is why `Criterion` carries `minimize`.
- **Fisher-information criterion** — pick the point that most sharpens the *hyperparameter* estimate
  rather than the prediction. Useful when the local fit is unstable. *Missing; low priority.*
- **Ray search** — instead of scoring every candidate at every greedy step, search along rays
  emanating from the predictive location. Turns the dominant cost from linear in dataset size into
  something nearly independent of it. This is what makes laGP usable at millions of rows. *Shipped*
  as `search="ray"`. `Xstart` is deterministic rather than R-RNG sampled, so there is no exact parity
  to be had here — see `tests/test_search.py` for what is asserted instead.
- **Continuous search** — optimize the criterion over the input space directly and snap to the
  nearest candidate. Same cost saving as ray search, and it can *place* a point rather than only
  choose one, which matters for sequential design. *Shipped* as `search="continuous"`, reusing
  `optim.lbfgsb` with the criterion gradient from `jax.grad`. laGP needs a hand-written `dalcGPsep`
  for exactly this; we do not, which is the differentiability pitch in concrete form.

### Fitting the local model

- **Per-location lengthscale and nugget estimation**, with priors derived from the design so every
  tiny local fit is well posed. This is what makes the emulator nonstationary — each prediction gets
  its own lengthscales. *Shipped.*
- **Joint lengthscale/nugget estimation**, alternating until both settle. *Shipped.*
- **Choosing which hyperparameters are estimated and which are held fixed**, including the common
  case of a known noise level. *Shipped.*
- **MAP rather than MLE** — the design-derived Gamma priors act as two-sided barriers, which is what
  keeps a fit on a few dozen points from running off to a degenerate lengthscale. *Shipped.*
- **Anisotropic (separable) and isotropic correlation.** Separable is shipped; isotropic is a special
  case of it and needs no separate work.

### Predicting

- **Student-t predictive distribution**, pointwise variances or a full joint covariance over
  predictive locations. *Shipped.*
- **Prediction without the nugget**, for interpolating the underlying function rather than a noisy
  observation of it. *Shipped.*
- **Differentiable prediction** — gradients of the mean and variance with respect to the predictive
  location, and now through the local hyperparameter fit as well. Not something laGP can do, and the
  reason the emulator composes with the rest of a JAX program. *Shipped.*
- **Simple kriging** — zero mean, no trend, the laGP-equivalent default. Constant and linear trends
  are shipped alongside it; see "Beyond laGP" below.

### Applying it to a whole dataset

- **One predictive location** — build a local design, fit it, predict. *Shipped.*
- **Many predictive locations, independently** — the actual user-facing entry point, and where the
  method earns its speed, since every location is separable work. *Shipped.*
- **Iterated refinement** — run the whole thing once, then run it again with each location starting
  from the lengthscales the first pass fitted there. Design selection scores candidates at the
  *starting* lengthscale, so the second pass picks neighbourhoods that respect which dimensions
  matter instead of treating all of them alike. *Shipped* (`Conditioned.refine`). On a strongly anisotropic
  4-d problem it cuts rmse ~1.7-2x, and `("nn", "alc")` beats `("alc", "alc")` — the first pass only
  exists to supply lengthscales, so paying for ALC there buys little.
- **Cluster and thread parallelism.** *Not shipping* — XLA decides.

### Global hyperparameter estimation on big data

- **Block-bootstrap subsampling** — estimate one global lengthscale from many small subsamples of a
  huge design, as a starting point or a fixed prior for the local fits. *Missing; low priority.*
- **Random line sampling** for visual diagnostics. *Not shipping.*

### Built on top of the GP, in laGP but out of scope here

- **Constrained blackbox optimization** (augmented Lagrangian, expected improvement and expected
  feasible improvement drivers). A separate package's job; JAX users already have optimizers.
- **Computer-model calibration** (discrepancy estimation and calibration objectives). A distinct
  research workflow, not part of the emulator.

### Beyond laGP

Two things laGP cannot do, both now shipped:

- **Trend models.** laGP is simple kriging everywhere, so the user must center their own response and
  the intervals omit the uncertainty in the mean. `trend="constant"` and `trend="linear"`, off by
  default, buy better-calibrated intervals — especially at design boundaries and on one-sided local
  neighbourhoods. Sold as calibration, not as a better mean; `tests/test_trend.py` tests the coverage
  claim rather than rmse.
- **Gradients through fitting, and a compilable pipeline.** `Local(...).condition(X, Z).predict(XX)`
  is an ordinary JAX function end to end: no host callbacks after `condition`, and `jax.grad` runs
  through the local hyperparameter fit by the implicit function theorem. See "Gradients through the
  fit" above for the mechanism and the tolerance rules that go with it.

### Order

```
Phase 0  API rename                  done, 539c740
Phase 1  ray + continuous search     done, 42bf35a
Phase 2  IECI                        done, 42bf35a
Phase 3  trend models                done, 42bf35a
Phase 4  differentiable fitting      done, f86c421
Phase 5  MSPE / Fisher / blhs        remaining, low value
```

What is left: Phase 5, the CI job, and the performance levers in `ROADMAP.md`'s cross-cutting
section. `ROADMAP.md` has the per-phase design, acceptance criteria and laGP reference algorithms,
plus the departures from its own plan that were worth keeping.

The pre-publish checklist is unaffected by any of this — the PyPI placeholder should happen
regardless of progress, and a `0.0.1` stub locks in no API.

## tests/

`tests/conftest.py` provides the `R` fixture (rpy2 + laGP from `~/R/library`, skipping when either is
unavailable) and the shared designs: `small` (40x3), `medium` (600x4), `wide` (200x8), `noisy` (300x3
with observation noise). `test_covar`, `test_gp`, `test_prior`, `test_mle`, `test_criteria`,
`test_ieci` and `test_local` compare against R. `test_transforms`, `test_search`, `test_trend` and
`test_diff` are property tests — the things with no laGP equivalent to compare to.

`tests/snapshot.py` is the regression oracle, not a test module: `python -m tests.snapshot write
<path>` records 124 arrays from every entry point, `check <path>` asserts bit-identity. The durable
copy lives at `~/.cache/lagpax/snapshot.npz`. Run `check` after anything that is supposed to be
value-neutral — the rename, the trend refactor and the implicit-gradient work all leaned on it.

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
(JAX vs laGP at 1 thread and at `os.cpu_count()`), `scaling.py` cost per location as `n` grows,
`plot.py` draws them.

`scaling.py` pins `close=n` on purpose. With laGP's default cap the candidate set stops growing and
every search looks flat, which would make the plot say nothing. With it pinned, `bench/scaling.csv`
shows what Phase 1 was for: over a 64x range in `n` (1000 -> 64000) exhaustive goes 0.090 -> 0.467 s
per location while ray goes 0.088 -> 0.118 and continuous 0.149 -> 0.172. Ray is the cheaper of the
two and continuous carries a fixed L-BFGS overhead that only pays off at large `n`.

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
- `uv.lock` is in `.gitignore`, so nothing is pinned across machines and a fresh `uv sync` resolves
  current versions. The `rpy2>=3.5.17,<3.6` bound in `pyproject.toml` is the only guard — do not
  relax it while resolving something else.
- Piping a long background run through `tail` buffers everything until it exits, so the output file
  stays empty. Let it write its CSV and read that instead.
