# lagpax implementation roadmap

Execution plan for an agent working unsupervised. `CLAUDE.md` says *what* the package should be able
to do and why; this file says *how*, in what order, and what "done" means for each step.

Read `CLAUDE.md` first — it is the repo's operating manual and its "Gotchas" section will save you an
hour. Where the two disagree, `CLAUDE.md` wins on facts about the current repo and this file wins on
plans.

---

## 0. Ground rules

### Environment

```bash
module load r/4.4.0 && export R_HOME=$(R RHOME)   # required for rpy2; do this in every new shell
uv sync --extra tests
uv run --no-sync pytest tests -q
```

- `--no-sync` on every `uv run` is load-bearing. A bare `uv run` re-syncs the default env and prunes
  whichever extra is installed. `tests` and `bench` are separate extras that evict each other, so
  switching between the suite and the benchmarks means another `uv sync --extra ...`.
- Symptom of getting this wrong: `ModuleNotFoundError: No module named '_cffi_backend'`. Fix is
  `uv sync --extra tests`, not debugging rpy2.
- **A green `pytest` proves nothing unless R loaded.** Without R + laGP the `R` fixture skips and only
  `tests/test_transforms.py` runs. Check the skip count on every run; it must be 0.
- `uv` for all Python. Never bare `python`/`pip`/`poetry`/`twine`. Never hand-edit `pyproject.toml` —
  use `uv add` / `uv remove`. `uv.lock` is gitignored, so the `rpy2>=3.5.17,<3.6` bound in
  `pyproject.toml` is the only guard against a breaking rpy2; do not relax it.
- Long runs: let them write their output file and read that. Piping through `tail` buffers everything
  until exit and the file looks empty.

### Style

`~/.claude/PYTHON.md` governs. The parts that bite most often here:

- One-line docstrings. No numpy-style `Parameters`/`Returns` blocks. Types live in the signature.
- Block comments state *intent* of a block, not a narration of each statement. If a comment explains
  what a variable is, rename the variable.
- Shape hints via jaxtyping: `Float[Array, "n m"]`, `Float[Array, "... d"]` for leading batch axes.
- `einops` `rearrange`/`reduce`/`repeat` over raw `reshape`/`transpose`.

### Parity discipline

This is the package's whole credibility claim. Rules:

- Anything with a laGP equivalent gets a value-level parity test against R. Non-exported R functions
  are reachable as `laGP:::dllikGPsep` etc.
- Anything **without** a laGP equivalent (trend models, differentiability, ray-search internals) gets
  property tests instead — analytic special cases, finite-difference checks, invariances.
- Where our optimizer finds the same KKT point by a different path, assert *that*, plus "our llik is
  no worse than laGP's". This pattern already exists in `tests/test_mle.py`; copy it, don't invent a
  new one.
- Parity is a property of the **laGP-equivalent configuration**, not of the package, as soon as
  anything divergent ships. Every new flag must have a documented default that reproduces laGP, and a
  test asserting the default still matches.
- Tolerances are deliberately non-uniform. `mean`/`llik`/`phi` at ~1e-12; `s2` at `rtol=1e-8,
  atol=1e-11` because it is computed as `1 + g - k' Ki k`, a cancellation whose absolute error tracks
  the O(1) intermediate. Batch-invariance under `jit`/`vmap` at 1e-6..1e-9. Do not tighten these to
  make a point; do not loosen them to make a test pass.
- `darg`/`garg` subsample the design when `n > 1000` using an RNG, so **parity fixtures stay under
  1000 rows**.

### License

LGPL-2.0-or-later, inherited from laGP; this is a derivative work. Do not relicense. If you add
modules that are translations of laGP source, keep `NOTICE` accurate.

### When to stop and ask

Stop and report rather than guessing if: a parity test fails by more than the stated tolerance and
the cause is not obviously ours; laGP's own behaviour looks like a bug; a phase's design section
turns out to be wrong about the math. Do not silently loosen a tolerance, skip a test, or mark a
parity test `xfail` to get a green run.

---

## Phase 0 — API rename (blocking; do this first) — **done** (`539c740`)

**Goal.** Replace the laGP-C-shaped surface with the design in `CLAUDE.md` § API. Zero behaviour
change.

**Why first.** Every later phase adds a criterion, a search, or a flag. Each one landed against the
current names is another call site to migrate later. Only a `0.0.1` stub will be on PyPI, so there
are no downstream users, no aliases and no deprecation window — and this is the last moment that is
true.

### Target module layout

| module | holds |
| --- | --- |
| `covar.py` | `distance`, `covar`, `covar_symm`, `dcovar` (drop `_sep`) |
| `gp.py` | `GP` pytree with methods |
| `prior.py` | `Range`, `Prior` |
| `optim.py` | `lbfgsb`, `bracketed_newton` — internal, not exported |
| `mle.py` | `fit` (one entry point) |
| `criteria.py` | renamed from `alc.py`; `alc` now, `mspe`/`ieci`/`fisher` later |
| `search.py` | **new**; empty-ish for now, gets `exhaustive`/`ray`/`continuous` in Phase 1 |
| `local.py` | `Local`, the conditioned emulator, `agp`, `lagp` |

### Renames

| now | planned |
| --- | --- |
| `new_gpsep(X, Z, d, g, mask)` | `GP(X, Z, d, g, mask)` |
| `newparams_gpsep(gp, d, g)` | `gp.at(d=, g=)` |
| `llik_gpsep(gp, dab, gab)` | `gp.llik(prior=None)` |
| `pred_gpsep(gp, XX, lite)` | `gp.predict(XX, lite=)` |
| `update_gpsep(gp, X, Z)` | `gp.update(X, Z)` |
| `alc_gpsep(gp, Xcand, Xref)` | `alc(gp, Xcand, Xref)` |
| `darg(d, X)`, `garg(g, Z)` | `Prior.from_design(X, Z)` |
| `mle_gpsep_d`, `mle_gpsep_nug`, `jmle_gpsep`, `mle_switch` | `fit(gp, prior)` |
| `lagpsep(Xref, X, Z, ...)` | `lagp(X, Z, Xref, ...)` |
| `agpsep(X, Z, XX, method=)` | `agp(X, Z, XX, criterion=, search=)` |
| `agpsep_seq(X, Z, XX, methods=)` | `Local(...).condition(X, Z).refine(XX, passes=)` |
| `LaGP`, `AGP` | `Prediction` |
| `Darg`, `Garg` | `Prior`, `Range` |

### Design notes

- `GP` stays a NamedTuple so it remains a pytree; methods on a NamedTuple are jit-safe. `sep`
  disappears — separable is the only family and isotropic is `d` broadcast.
- `Prior` holds both `d` and `g` as `Range(start, min, max, ab, estimate)`. Every call site needs
  both. `Prior.from_design(X, Z, rng=None)` is the old `darg` + `garg`, and stays host-side
  numpy/scipy — R's `quantile` type 7 equals numpy's default, which is why it is not jitted.
- `fit(gp, prior)` dispatches on `prior.d.estimate` / `prior.g.estimate`, which is what `mle_switch`
  already does. Keep `mle_gpsep_d`'s laGP boundary reset (`0.9*dmin + 0.1*dmax`, conv=102).
- `Local` is a NamedTuple of settings. `Local.condition(X, Z, prior=None, rng=None)` resolves the
  prior once and returns an object holding `(settings, X, Z, prior)`. `.predict(XX, batch=64)` and
  `.refine(XX, passes=(...))` are methods on it. `agp`/`lagp` are three-line wrappers.
- **`condition` is not cosmetic.** It is the seam that puts the one host-side step ahead of all JAX
  work, which Phase 4 depends on. Do not let prior resolution leak back into the per-location path.
- `refine` replaces `agpsep_seq`'s `**kwargs` forwarding. That forwarding is where a `d`-resolution
  bug lived: each pass re-ran `darg`, which re-subsamples with the RNG, so passes silently disagreed
  on the prior. `refine` resolves once at `condition` time and reuses it.
- One `Prediction` for both entry points, with `mean`, `s2`, `df`, `Xi` (0-based), `fit`
  (`Fit(d, g, its, conv)` — a typed struct, not the current dict), and `var` as a property equal to
  `s2 * df / (df - 2)`. This ends laGP's split where `laGPsep` returns raw `s2` and `aGPsep` returns
  the scaled `var`.
- Demote from `__all__`: `lbfgsb`, `bracketed_newton`, `covar_symm`, `dcovar`, `igamma_inv`,
  `get_ds`, `dllik_gpsep_d`, `dllik_gpsep_g`, and the four `mle_*` names. They stay importable from
  their modules — `tests/` reaches into `lagpax.optim` directly and the parity suite compares values,
  not names. Target public surface ~14: `GP`, `Prior`, `Range`, `Prediction`, `Fit`, `Local`, `agp`,
  `lagp`, `fit`, `alc`, `mspe`, `ieci`, `distance`, `covar`.
- Keep `jax.config.update("jax_enable_x64", True)` at import. Process-global side effects from a
  library import are rude, but every tolerance in `tests/` assumes it and laGP is float64 throughout.

### Procedure

1. **Snapshot first.** Before touching anything, write a script that runs the current `agpsep`,
   `agpsep_seq` and `lagpsep` on the `tests/conftest.py` designs with a fixed seed and saves
   `mean`/`var`/`llik`/`Xi`/`mle` to `.npz` in the scratchpad. This is the regression oracle.
2. Rename module by module, bottom up: `covar` → `gp` → `prior` → `mle` → `criteria` → `local` →
   `__init__`. Run the suite after each module.
3. Migrate `tests/` and `bench/` call sites. Test *names* can keep their laGP references — they
   describe what is being compared against.
4. Re-run the snapshot script against the new API and assert bit-identical output. Any difference is
   a bug in the rename, not an improvement.
5. Update `README.md`. Its "Scope" section currently lists the old laGP-shaped names and claims
   `aGP.seq` is "not ported", which has been false since `agpsep_seq` landed.

### Acceptance

- `uv run --no-sync pytest tests -q` green with **0 skips**.
- Snapshot outputs bit-identical before/after.
- `len(lagpax.__all__) <= 15`.
- No name containing `gpsep` or `sep` in `__all__`.
- `README.md` quickstart runs as written.

---

## Phase 1 — Ray and continuous search — **done** (`42bf35a`)

Landed in `search.py` as `exhaustive`/`ray`/`continuous`, selected by `Local(search=)`. `Xstart` is
deterministic (the first `numrays` of the `offset` nearest candidates) rather than R-RNG sampled, as
the design notes below allow. Scaling evidence is `bench/scaling.py` -> `bench/scaling.csv`.

**Goal.** `search="ray"` and `search="continuous"`, ports of laGP's `alcray`/`alcopt`.

**Why.** The largest scalability item. Exhaustive ALC scores every one of `close` candidates at every
greedy step, so cost is linear in dataset size. Ray and continuous search make it nearly independent
of it, which is what makes laGP usable at millions of rows.

### laGP reference

Get the R-level source with:

```bash
Rscript -e '.libPaths("~/R/library"); library(laGP); print(laGP:::lalcrayGPsep.R)'
Rscript -e '.libPaths("~/R/library"); library(laGP); print(laGP:::lalcoptGPsep.R)'
Rscript -e '.libPaths("~/R/library"); library(laGP); print(laGP:::ray.end)'
Rscript -e '.libPaths("~/R/library"); library(laGP); print(laGPsep.R)'
```

`laGPsep.R` is the pure-R reference for the whole greedy loop and shows exactly how the two searches
slot in. Transcribed here so you can work without R up:

**Ray search**, per greedy step `t`:

```
offset  = ((t - start) %% floor(sqrt(t - start))) + 1
numrays = ncol(X)                                   # default
Xstart  = numrays candidates sampled from the first `offset` candidates
Xend    = ray.end(...): Xend = 10*(Xstart - Xref) + Xstart, then repeatedly
          clip: for the first coordinate outside `rect`, rescale the whole
          ray so that coordinate lands exactly on the violated face
(r, s)  = argmax over rays r and s in [0,1] of ALC at (1-s)*Xstart[r] + s*Xend[r]
Xstar   = (1 - s)*Xstart[r] + s*Xend[r]
w       = nearest candidate to Xstar
```

**Continuous search**, per greedy step `t`:

```
offset = t - start
Xstart = Xcand[offset : offset + numstart]
for each start: maximize ALC over x in rect (box-constrained), snap to nearest
                candidate, score that candidate with plain ALC
w      = candidate with the best score
```

Also: `rect` defaults to the coordinatewise range of the candidate set; `close` defaults to
`10 * (1000 + end)` for both searches rather than `1000 + end`; `numstart` defaults to `ncol(X)` for
ray and 1 for continuous; ray search requires `nrow(Xref) == 1`, which is always true on the `agp`
path.

### Design notes

- Both go in `search.py` behind a common interface: given `(gp, Xcand, Xref, rect, step)`, return the
  index of the chosen candidate. The greedy loop in `local.py` calls it and stays otherwise unchanged.
- **Everything must be fixed-shape.** The greedy loop is a `lax.fori_loop`. A ray search with a
  data-dependent iteration count will not compile. Use a fixed number of golden-section iterations
  for the 1-D search along a ray (~30 is well past float64 resolution on a unit-length ray) rather
  than a convergence loop.
- **Continuous search should reuse `optim.lbfgsb`.** It is already a projected box-constrained
  L-BFGS, and `jax.grad` gives the ALC gradient w.r.t. `x` for free. laGP needs a hand-written
  `dalcGPsep` for exactly this; we do not. This is a concrete instance of the differentiability pitch
  — say so in the docstring and in `README.md`.
- The ray `Xstart` sampling uses R's RNG. Do not chase RNG parity. Make `Xstart` deterministic
  (take the first `numrays` of the `offset` nearest candidates) and document the divergence.
- `rect` should be exposed as an optional argument, defaulting to the candidate bounding box.

### Testing

Exact parity is **not** achievable here and you should not try: the 1-D search inside
`alcrayGPsep` is C-only with no R-level reference, and the `Xstart` sampling is RNG-dependent. Test:

1. **Criterion parity.** For a fixed `gp`, `Xcand`, `Xref`, our ALC scores match `laGP:::alcGPsep` to
   1e-12. Already covered; keep it as the foundation.
2. **No worse than laGP.** For the same greedy problem, the ALC value of *our* chosen candidate is
   `>=` the ALC value of laGP's chosen candidate, minus tolerance. Same pattern as the optimizer
   tests.
3. **Degenerate agreement.** With `numrays` large and the candidate set small, ray search should
   converge on the same pick as exhaustive ALC. Assert equality on a small design.
4. **Accuracy floor.** On the `medium` fixture, rmse of `search="ray"` within ~1.5x of
   `search="exhaustive"`, and both far better than `criterion="nn"`.
5. **Scaling.** A timing check (in `bench/`, not `tests/`) showing ray-search cost is roughly flat in
   `n` where exhaustive is linear. This is the entire justification for the phase; if it does not
   show up, something is wrong.

### Acceptance

- Suite green, 0 skips.
- `agp(..., search="ray")` and `search="continuous"` run end to end under `jit`.
- Default `search="exhaustive"` output bit-identical to Phase 0 snapshot.
- `bench/` gains a plot or CSV column showing the flat-in-`n` scaling.

---

## Phase 2 — IECI — **done** (`42bf35a`)

`criteria.ieci(fmin=, w=, nonug=)` is a factory returning the standard criterion signature, and
`Criterion.minimize` carries the argmin/argmax choice so the greedy loop never hardcodes a direction.
Parity in `tests/test_ieci.py`.

**Goal.** `criterion="ieci"`, port of `ieciGPsep`. Turns the emulator into a BO surrogate.

**laGP signature.** `ieciGPsep(gpsepi, Xcand, fmin, Xref = Xcand, w = NULL, nonug = FALSE)`.

**Definition.** Expected improvement at a reference location, conditional on having added a candidate
point, integrated (weighted-averaged) over the reference set:

```
EI(x)          = E[max(fmin - Y(x), 0)]        under the Student-t predictive at x
ECI(x* | xc)   = EI(x*) under the GP updated with xc   (variance-only update: adding a
                 candidate changes the predictive variance, not the mean, since the
                 response there is unknown)
IECI(xc)       = sum_j w_j * ECI(xref_j | xc)
```

Lower IECI is better — laGP returns it as a quantity to **minimize**, unlike ALC. Check the sign
against `laGP:::ieciGPsep` before wiring it into the greedy loop, and make the greedy loop's
`argmax`/`argmin` a property of the criterion rather than hardcoded.

### Design notes

- Criterion signature stays `(gp, Xcand, Xref) -> Float[Array, "ncand"]`. IECI needs `fmin` and
  optional weights `w`, so it is a **factory**: `ieci(fmin=..., w=None, nonug=False)` returns a
  function with the standard signature. This is the pattern for any configurable criterion.
- `fmin` in a BO loop is the best observed value so far, so it changes every iteration. Keep it an
  argument, never derive it internally from `Z`.
- Student-t EI, not Gaussian EI — the predictive is Student-t with `df = n`. Getting this wrong is
  the most likely error in this phase and it will look almost right.
- The conditional variance update is the same rank-1 quantity ALC already computes. Reuse it rather
  than re-deriving; `alc.py`/`criteria.py` already has the pieces.

### Testing

- Value parity against `laGP:::ieciGPsep` on the `small` and `noisy` fixtures, 1e-10.
- Sign/direction test: on a 1-d toy problem, IECI prefers candidates near the incumbent minimum,
  while ALC prefers candidates that reduce global variance. If those two agree everywhere, the
  implementation is wrong.
- `jax.grad` of IECI w.r.t. a candidate location is finite and matches finite differences.

### Acceptance

- Parity test green.
- `agp(..., criterion=lagpax.ieci(fmin=...))` runs.
- Greedy loop handles minimize-vs-maximize criteria generically.

---

## Phase 3 — Trend models — **done** (`42bf35a`)

`GP` carries `F`, `b`, `Kfi`, `ldetF` and `df = n - p`; `trend="zero"` keeps `p == 0` and every
snapshot array bit-identical. Tests in `tests/test_trend.py`, coverage claim included.

**Goal.** `trend="constant"` and `trend="linear"` alongside the default `trend="zero"`. Not in laGP.

**Why.** laGP is simple kriging everywhere, so users must center their own response and the intervals
omit the uncertainty in the mean. **Sell this as calibration, not as a better mean** — the gain shows
up in interval coverage at design boundaries and on one-sided local neighbourhoods, not in rmse.

### Math

With trend basis `F` (n x p; `F = 1` for constant, `F = [1, X]` for linear), profiling out both the
trend coefficients and the scale:

```
b    = (F' Ki F)^-1 F' Ki Z                       generalized least squares coefficients
phi  = (Z - F b)' Ki (Z - F b)
df   = n - p                                      was n
mean(x)  = f(x)' b + k(x)' Ki (Z - F b)
s2(x)    = (phi / df) * [ 1 + g - k' Ki k + u' (F' Ki F)^-1 u ]
           where u = f(x) - F' Ki k
```

The `u' (F' Ki F)^-1 u` term is the trend-uncertainty contribution — it is the entire point, and it
is what simple kriging drops. `p = 0` must reduce exactly to the current code path.

### Design notes

- `trend` is a static field on `Local` and on `GP` construction; it changes shapes, so it cannot be a
  traced value.
- `F' Ki F` is `p x p` with `p <= m + 1` and `m` small in the local setting, so a dense solve is fine.
  Guard against singularity: a local design of `end=50` points is always well past `p`, but a padded
  masked design must exclude masked rows from `F` too. **This is a mask-invariant.** The existing
  three (masked `K` rows/cols set to identity, masked `Z` zeroed, masked `k`/`kx` zeroed) become
  four. Add a test mirroring `test_mask_matches_a_smaller_design` for the trend path.
- `df = n - p` propagates into the `var = s2 * df/(df-2)` scaling. Check the `Prediction.var`
  property still does the right thing.

### Testing

No laGP equivalent, so property tests:

- `trend="zero"` output bit-identical to Phase 0 snapshot. Non-negotiable — this is the parity
  default.
- Analytic check: with `d -> inf` (flat correlation) and `trend="constant"`, the predictive mean must
  approach the GLS mean of `Z`.
- Coverage check: on a design with a strong linear trend, simulate, and confirm empirical coverage of
  90% intervals is closer to 0.90 with `trend="linear"` than with `trend="zero"`. This is the claim;
  test the claim.
- Mask invariance for the trend path, as above.
- Finite-difference check of `jax.grad` through the trend-model predictive mean and variance.

### Acceptance

- Suite green, 0 skips, `trend="zero"` bit-identical.
- Coverage test demonstrates the calibration gain.
- `README.md`'s "Ordinary and universal kriging ... Not implemented yet" paragraph updated.

---

## Phase 4 — Differentiable fitting and a compilable pipeline — **done**

Landed as `optim.implicit_argmin`, applied in `mle.mle_d`, `mle.mle_g` and `mle.jmle`. Tests in
`tests/test_diff.py`. Three deliberate departures from the design below, each for a reason worth
keeping:

- **A straight-through Newton step, not `jax.custom_vjp`.** `x - (step - stop_gradient(step))` from a
  frozen `x` carries exactly the implicit-function derivative while returning the solver's value
  bit-identically — which is what keeps the 124-array snapshot intact. `custom_vjp` would have needed
  `nondiff_argnums` over a closure holding tracers (an `UnexpectedTracerError` waiting to happen) and
  float0 plumbing for `GP`'s boolean `mask` leaf.
- **`refine` keeps its Python loop.** `lax.scan` is impossible, not merely unnecessary: passes may
  name *different* criteria, which trace to different functions. The loop unrolls at trace time and
  the whole thing still jits and differentiates as one graph.
- **The 1e-5 finite-difference acceptance is not reachable by differencing the solver,** and chasing
  it is a trap. `lbfgsb` stops on R's `factr=1e7`, so it returns ~8 significant digits and a central
  difference of its output carries `|x| * 1e-8 / 2h` ~ 7e-5 of noise; truncation error forbids raising
  `h`. Tightening `factr` to 1e3 only moves the floor to 6.8e-5. The tests instead compare against an
  argmin obtained by root-finding `grad llik = 0` with scipy, which removes the floor entirely and
  agrees to ~1e-6. Reuse that pattern for any future gradient test through an optimizer.

Also established: `jit` preserves the optimizer trajectory (1e-9), `vmap` does not. A batched
`while_loop` shares its trip count, so one ulp of likelihood difference flips `lbfgsb`'s
relative-change test and the answers separate by the solver's own resolution `sqrt(factr*eps)` ~ 5e-5
while the likelihood reached still agrees to `factr*eps` = 2.2e-9. Assert the objective, not the
argmin, when comparing batched against unbatched.

**Goal.** Make the whole emulator an ordinary JAX function: `jit`-able end to end, and
differentiable through the hyperparameter fit, not just with hyperparameters held fixed.

**Why.** `README.md` already sells differentiability. Today that is true of prediction only; the
fitting stage escapes to the host mid-call. Closing this is what lets the emulator nest inside a
larger JAX model.

### Blockers, in order

1. **Host-side prior resolution.** `prior.py` is numpy/scipy (`igamma_inv`, quantiles). *Already
   solved by Phase 0's `condition` seam* — resolve once up front, treat the result as a constant.
   Nothing further needed unless someone wants gradients w.r.t. the prior itself, which is not a
   goal. Do not re-derive `igamma_inv` in JAX for this.
2. **`optim.lbfgsb` uses `lax.while_loop`**, which has no reverse-mode rule. Same for
   `bracketed_newton`.
3. **`local.py`'s per-pass Python loop** in `refine`, and the host-side branching in `fit`'s dispatch.

### Approach

Do **not** unroll the optimizer to differentiate through its iterates. Memory blows up and the
gradient is dominated by optimizer transients. Use **implicit differentiation**: at the optimum the
gradient of the objective vanishes, so by the implicit function theorem the derivative of the
argmin w.r.t. upstream inputs follows from the Hessian and the mixed partial, both of which
autodiff gives you. Wrap the solver in `jax.custom_vjp`.

Caveats to handle explicitly:

- The implicit-function argument assumes an **interior** optimum. `mle_gpsep_d`'s boundary reset
  (conv=102) and the nugget pinning at its lower bound on noiseless designs are exactly the cases
  where it fails. Define the gradient as zero w.r.t. an active bound constraint and document it. Do
  not pretend it is smooth there.
- Test on `noisy` (interior optimum) for gradient correctness and on the noiseless fixtures for the
  boundary behaviour.
- `refine`'s pass loop can become `lax.scan` once the per-pass work is jittable, since every pass has
  identical shapes. `fit`'s dispatch is on static booleans, so it stays a Python `if`.

### Testing

- `jax.jit(agp)` compiles and matches the uncompiled result to batch-invariance tolerance
  (1e-6..1e-9, not exact — reassociation under `jit` costs a few ulps).
- Finite-difference check of `d(prediction)/d(X)` and `d(prediction)/d(Z)` through the full fit on
  `noisy`, where the optimum is interior.
- Gradient is exactly zero w.r.t. a hyperparameter pinned at a bound.
- `jax.vmap` over multiple datasets works, not just over predictive locations.

### Acceptance

- The full `Local(...).condition(X, Z).predict(XX)` path runs under `jit` with no host callbacks.
- Finite-difference agreement on `noisy` at 1e-5 relative.
- `README.md`'s differentiability claim is true without qualification.

---

## Phase 5 — Low-value leftovers

Do these last or not at all. Each is small; none changes what the package is for.

- **MSPE** (`criterion="mspe"`, port of `mspeGP`). ALC's variance reduction plus the extra error from
  re-estimating the lengthscale once the point is added. Signature
  `mspeGP(gpi, Xcand, Xref = Xcand, fi = TRUE)`; `fi` toggles the Fisher-information term. Parity
  against `laGP:::mspeGP`. Accuracy gain over ALC is small at appreciably more cost.
- **Fisher criterion** (`criterion="fisher"`, port of `fishGP(gpi, Xcand)`). Picks the point that
  most sharpens the *hyperparameter* estimate rather than the prediction. Useful when the local fit
  is unstable. Parity against `laGP:::fishGP`.
- **Block-bootstrap subsampling** (port of `blhs(y, X, m)` / `blhs.loop`). Estimates one global
  lengthscale from many small subsamples of a huge design. It slots in as a producer of a `Prior`,
  which is why Phase 0 made `Prior` a first-class argument — `condition(X, Z, prior=blhs_prior)`
  should just work with no further plumbing. Parity against `laGP::blhs`.

**Explicitly not shipping**, do not implement even if it looks easy: cluster/thread parallelism (XLA
decides), random line sampling (a visual diagnostic), constrained blackbox optimization
(`optim.auglag`/`optim.efi` — a separate package's job, and JAX users already have optimizers), and
computer-model calibration (`discrep.est`/`fcalib` — a distinct research workflow, not part of the
emulator).

---

## Cross-cutting: performance

Not a phase; pick these up when a phase makes one of them natural. Current state is that a 25x
fp64-throughput range across GTX 1080 / L40S / V100 produces only a 2.9x spread in wall clock, so the
workload is bound by kernel-launch latency and the sequential ALC greedy loop, not arithmetic. Buying
a better GPU does not fix it. Open levers:

- Rank-1 Cholesky update inside the ALC greedy loop instead of a rebuild. `GP.update` also rebuilds
  rather than applying laGP's rank-1 update — correct, but a performance item.
- `lax.scan` over the greedy steps.
- fp32 for the ALC *selection* stage with fp64 for the final fit. Selection only needs an argmax.
- Batch-size sweep for `lax.map(batch_size=)`.

**Benchmark hygiene**, from `CLAUDE.md` § bench: `bench/accuracy.py` must keep computing `darg` once
and handing the same prior to both sides — with independent priors the median relative error at
n=2000 is ~1e-4 versus ~1e-8 shared, i.e. it measures the RNG gap rather than the port.
`OPENBLAS_NUM_THREADS=1` is set at the top of `bench/common.py` before jax or R load; keep it there.
The `vs16c` ratio is **not comparable across CSV files** — the GPU runs used SLURM nodes whose host
CPUs differ by ~3x. Fix that before publishing any cross-chip comparison: pin the same host CPU, or
report a laGP baseline per node.

---

## Pre-publish checklist

Independent of how far the phases have progressed, except where noted.

- [ ] **Reserve `lagpax` on PyPI with a placeholder upload.** Plain `lagp` was already squatted by an
      unrelated LLM-proxy project, so the name is not safe just because it is free today.
      `uv build && uv publish` (needs `UV_PUBLISH_TOKEN`, or `--token`). Ship it as version **`0.0.1`,
      not `0.1.0`** — PyPI never lets a version number be reused, and deleting a release does not
      free its filename, so burning `0.1.0` on a stub costs the real first release its natural
      number. A stub `__init__.py` with a version string plus a README pointing at the GitHub repo is
      enough; it reserves the name without shipping something you have to support.
- [ ] **Phase 0 lands before `0.1.0`.** The stub commits to nothing, so the rename is free until the
      first real release and expensive after it.
- [x] **CI running the parity suite.** `.github/workflows/tests.yml`: `r-lib/actions/setup-r`, laGP
      from CRAN with a fallback to the pinned `laGP_1.5.9.tar.gz` archive tarball, then the suite
      under `LAGPAX_STRICT=1`. That env var makes `tests/conftest.py` return a nonzero exit on any
      skip, which is what stops the job going green without ever loading R. It has not run on GitHub
      yet — the repo has no remote — so treat the first push as the real check of the R install step.
- [ ] **Do not weaken the license.** LGPL-2.0-or-later, inherited. Keep `NOTICE` accurate as modules
      are added.
- [ ] `README.md` accurate: scope list, the `aGP.seq` claim, the differentiability claim, and the
      performance section.

---

## Order, restated

```
Phase 0  API rename                       done, 539c740
Phase 1  ray + continuous search          done, 42bf35a
Phase 2  IECI                             done, 42bf35a
Phase 3  trend models                     done, 42bf35a
Phase 4  differentiable fitting           done
Phase 5  MSPE / Fisher / blhs             remaining
```

What is left: Phase 5, the CI job, and the performance levers above.
