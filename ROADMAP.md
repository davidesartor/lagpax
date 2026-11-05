# lagpax roadmap

Status, rules, lessons, and what is next. `CLAUDE.md` is the operating manual and wins on any fact
about how the current code works; this file owns every measurement and every decision record.

## Status

The laGP port is **complete** and now lives under lagpax's own API (phase 8): the core laGP does —
the separable GP, default priors, MAP fitting, ALC/IECI with exhaustive/ray/continuous search, and
the local schemes — is verified against laGP 1.5.9; a few laGP features were dropped on purpose
(below). Box-constrained optimization is delegated to `vlse`, and parity also covers real
optimization benchmark surfaces. What is left before a first release is in § Next.

| phase | | commit |
| --- | --- | --- |
| 0 | API rename — laGP's C-shaped names to the surface in `CLAUDE.md` § API | `539c740` |
| 1 | ray + continuous search (`search.py`) | `42bf35a` |
| 2 | IECI (`criteria.ieci`) | `42bf35a` |
| 3 | trend models (`trend="constant"`/`"linear"`) | `42bf35a` |
| 4 | differentiable fitting (`optim.implicit_argmin`) | `f86c421` |
| 5 | MSPE, Fisher, block-bootstrap subsampling | `f6b4579` |
| 6 | L-BFGS-B from `vlse.optim.minimise`; parity on vlse's SFU surfaces (`tests/test_vlse.py`) | branch `ccr-7f9aed99-rlk3k1` |
| 7 | JAX-native API: no import side effect, explicit keys, pytree criteria/searches/emulator | `332e6cf` |
| 8 | Hard redesign: lagpax's own API (`LocalGP`/`Emulator`/`Prior`/`Param`), theta-or-lengthscale priors, laGP-isms dropped | `18691de` |
| 9 | From-scratch rewrite of the internals: pytree dataclasses throughout, self-documenting names, full type hints | branch `ccr-7f9aed99-rlk3k1` |

**Explicitly not shipping**, do not implement even if it looks easy: `aGP.parallel` (XLA decides
parallelism), `randLine` (a visual diagnostic), `optim.auglag`/`optim.efi` (constrained blackbox
optimization — a separate package's job), `discrep.est`/`fcalib` (computer-model calibration — a
distinct research workflow). **Removed in phase 8**, also not to come back without a user asking:
MSPE and Fisher-information criteria (rarely used, isotropic-only in laGP), `blhs` lengthscale
bootstrapping, laGP's conv=102 refusal to optimize from the lower bound, and the raw-scale `s2`
output.

## Rules that govern new work

- **Parity is the credibility claim.** Anything with a laGP equivalent gets a value-level parity test
  against R; non-exported R functions are reachable as `laGP:::alcGPsep` etc. Anything *without* one
  gets property tests — analytic special cases, finite differences, invariances.
- **Parity is a property of the laGP-equivalent configuration**, not of the package. Every divergent
  flag needs a documented default that reproduces laGP and a test asserting it still does. The API
  does not follow laGP (phase 8), but the numbers behind the defaults do: a theta-space `Prior` from
  `Prior.from_data`, `trend="zero"`, `criterion="alc"`, `search="exhaustive"`.
- **Where our optimizer finds the same KKT point by a different path, assert that** — plus "our llik
  is no worse than laGP's". On multimodal likelihoods only the second half holds; say so in the test.
- **Tolerances are not a dial.** Never loosen one to get a green run, never skip or `xfail` a parity
  test. When a deliberate change (a new solver, say) legitimately moves a tolerance, measure the new
  floor, measure the old code on the same check, set the tolerance from both, and record it below.
- **Stop and report** rather than guess if a parity test fails beyond tolerance for a reason that is
  not obviously ours, laGP's own behaviour looks like a bug, or the math turns out to be wrong.
- **Run the strict suite against R** (`LAGPAX_STRICT=1`) before calling a numerical change done. A
  cloud container can install R; `CLAUDE.md` § Commands has the recipe.

## What we learned

### Bit-identity is not the invariant; parity is

Several correct changes move results in the last ulps. `tests/snapshot.py` is a *change detector* —
it answers "did this move anything", and when it says yes the gate is the parity suite plus rmse.

- **Padding width is not neutral.** The mask invariants hold exactly but ALC's reduction reassociates
  by ~1.4e-16 across widths, which flips near-tied argmaxes and cascades through the greedy loop. At
  `n=4000` two padding schemes agree on 25 of 100 local designs. laGP is a third such point.
- **`jit` preserves the optimizer trajectory (1e-9); `vmap` does not.** A batched `while_loop` shares
  its trip count, so one ulp flips L-BFGS-B's relative-change test and the iterates separate by
  roughly `sqrt(factr*eps)` ~ 4.7e-5 in `d`. Compare the objective, not the argmin.
- **Do not finite-difference the solver.** It stops on R's `factr=1e7`, so it returns ~8 significant
  digits. `tests/test_diff.py` compares against an argmin from root-finding `grad llik = 0` with scipy.
  Reuse that for any gradient test through an optimizer.

### The solver swap (phase 6)

The hand-rolled projected L-BFGS (two-loop recursion, 21 Armijo probes evaluated in one `vmap`) was
replaced by vlse's L-BFGS-B (generalized Cauchy point, subspace minimisation, scipy's `dcsrch`), called
directly with R's stopping rule. What moved, each measured against the old solver on the same check:

- **Stationarity at `factr=1e7` is worse.** On `noisy[:60]`, `mle_d` stops after 8 iterations at
  max |grad| 2.7e-5, where the old solver took 11 and reached 1.7e-6. Both satisfy R's rule; the
  objective agrees to 5e-11. vlse reaches 2e-9 at `factr=1e3`.
- **So the implicit gradient is less exact**: 2.9e-5 relative against the root-found argmin, vs <1e-5
  before. `test_lengthscale_gradient_matches_an_exactly_solved_argmin` moved from `rtol=1e-5` to `1e-4`.
- **And the fitted map is rougher.** A pipeline finite difference at `h=1e-4` straddled a
  stopping-iterate jump on one coordinate (14% off); the old solver straddled them only from `h=3e-4`
  up. At `h<=3e-5` both agree with `jax.grad` to ~2e-3.
  `test_gradients_reach_the_data_through_the_whole_scheme` moved to `h=1e-5`, `rel=5e-3`.
- **Snapshot drift**: 83/124 arrays identical; `d` moves up to 1e-3 on `wide`, predictive means up to
  1.1e-6. Every pre-existing parity test passes unchanged against laGP 1.5.9.
- **Speed**: see the fit timing in § Performance.

### Real surfaces (vlse's SFU functions)

`tests/test_vlse.py`, eight surfaces, full-GP `jmle` at n=150 and `agp` at n=800, 40 locations:

- **Joint MLE is never worse than laGP's** (worst case −9e-9 in log-likelihood). Where the
  lengthscales differ wholesale — Branin, Michalewicz, Rosenbrock, Goldstein-Price — our likelihood
  is *higher*, by 3 to 14 units.
- **aGP agrees to 1e-10..1e-6 at the median everywhere**, and to <3e-3 at the max on smooth
  surfaces. On Michalewicz and Ackley a few locations land in a different local-likelihood basin from
  laGP's and the prediction moves by O(1); rmse against the truth is a wash (ours better on two of
  four cases). The old solver showed the same outliers, so they are a property of the surfaces, not
  of vlse.

### Design departures worth keeping

- **A straight-through Newton step, not `jax.custom_vjp`**, for implicit gradients: it returns the
  solver's value bit-identically and avoids `nondiff_argnums` over a closure holding tracers.
- **`refine` keeps a Python loop.** Passes may name different criteria, which trace to different
  functions, so `lax.scan` is impossible. It still jits and differentiates as one graph.
- **Criteria and searches are pytree dataclasses (phase 7), not factories or partials.** The old
  factories existed because `functools.partial` dropped the `minimize` attribute, which searches
  read with `getattr(..., False)` and would have silently maximized a minimized criterion. Class
  attributes on a `Criterion` base remove the failure mode instead of documenting it.
- **The parity targets for MSPE/Fisher are isotropic** (`newGP`); a separable GP with equal `d_k` is
  that GP exactly.
- **laGP contradicts itself and the port copies it** (`efiGP`'s `sq(dmu)/s2` vs `mspeGP`'s
  `sq(dmu/s2)`).
- **`Xstart` for ray search is deterministic**, not R-RNG sampled, so `test_search` asserts a property.

### Performance: the workload is latency-bound

A 25x fp64-throughput range across GTX 1080 / L40S / V100 produces a 2.9x spread in wall clock. The
binding constraints are kernel-launch latency and the sequential greedy loop, not arithmetic.

Profile on 16 CPU cores at `n=4000`, `end=50`, 100 locations (old solver): 0.38s neighbours + build +
predict, 0.78s design selection, **4.31s fit**. The fit is the cost.

**Fit with vlse vs the old solver**, CPU only (a 4-core cloud container), `agp` at `n=4000`, `end=50`,
100 locations, best of 3 after warm-up: `nn` 1.80s vs 1.91s, `alc` 3.16s vs 3.06s; mean lengthscale
iterations 10.1 vs 11.4. Neutral within noise on CPU. The GPU, where batched trip counts matter, is
unmeasured (§ Next).

**The location sweep.** Both engines are affine in the predictive-location count; quote
`fixed + per_location * nn`. `bench/timing-a100-loc{,-big}.csv`, `uri-gpu002` (A100-SXM4-80GB, Xeon
8352Y, 16 cores), `n=8000`, 250 -> 256,000 locations, `batch=64`, single seed, old solver:

| criterion | jax A100 | laGP 16-core | break-even | asymptotic |
| --- | --- | --- | --- | --- |
| `alc` | 3.15s + 0.850ms/loc | 6.218ms/loc | nn ~ 590 | 7.3x |
| `nn` | 2.45s + 0.473ms/loc | 1.608ms/loc | nn ~ 2160 | 3.4x |

**The batch sweep.** `bench/timing-a100-loc-b{64,256,1024,4096}.csv`, `ece-gpu002` (A100-PCIE-40GB),
`n=8000`, 16k and 64k locations:

| `batch` | alc ms/loc | alc fixed | nn ms/loc | nn fixed |
| --- | --- | --- | --- | --- |
| 64 | 0.819 | 3.02s | 0.454 | 2.50s |
| **256** | **0.568** | 3.01s | **0.316** | 2.00s |
| 1024 | 0.529 | 3.93s | 0.303 | 2.30s |
| 4096 | 0.521 | 5.77s | 0.446 | 0.88s |

Hence the default of 256: 1.44x off the per-location rate, lowest fixed term, smallest peak memory of
the competitive three.

**`batch=256` on the SXM card**, `bench/timing-a100-loc{,-big}-b256-sxm.csv` (JAX only, single seed,
old solver), fitted over 16k -> 256k: `alc` ~5.1s + 0.515 ms/loc, `nn` ~3.1s + 0.31 ms/loc. The
per-location rates match the PCIe sweep; the fixed term is higher. At 250 locations — under one
batch — the whole run is ~0.2s, and it jumps to 3-5s at 1000, so the "fixed" term appears with the
second batch: consistent with launch-latency-bound `lax.map` iterations, not a constant setup cost.

What worked:

| | |
| --- | --- |
| stop inverting `K` in the MLE inner loop (`gp._profile`) | ~2x fewer flops in a fitted local GP (`b52be87`) |
| cache the candidate/design cross-covariance across greedy steps | design selection no longer the hot stage (`8520f42`) |
| `lax.map` batch default 64 -> 256 | 1.44x off the per-location rate on an A100 (`b3d4995`) |
| fuse `bracketed_newton`'s two derivative callables into one `jvp` | -7% on the g-solve, -3.5% on `jmle` (`fe39244`) |
| pad the greedy design to a power of two | neutral at `end=50`, 1.16-1.20x at `end=200` (`d62668c`) |

What did not, and should not be retried without new evidence:

- **A gemm-based `covar`.** The kernel is 1.4-2.5x faster and the pipeline got **19% slower**: XLA's
  fusion of the broadcast-reduce form beats BLAS3 here.
- **fp32 for ALC selection.** Selection is not arithmetic-bound; halving precision costs the argmax
  its ties for nothing.
- **Fewer Armijo probes in the old solver** (21 -> 4): 0% faster and wrecked the answer (3.3e-2).
- **Looser `factr`** (1e7 -> 1e13, old solver): 2.1x fewer iterations for only -12% wall clock. The
  sweep fit 0.039 s/iteration with a **1.06s intercept** — 63% of the d-solve was fixed cost no
  iteration count touched. `implicit_argmin`'s Hessian was ruled out. Unexplained; re-measure with
  vlse before chasing it.
- **A lower `maxit`** (100 -> 8): slower than 30.
- **Warm-starting `d0`** from a previous pass: 75% slower, `its` 15.8 -> 18.3.
- **Rank-1 Cholesky update in the greedy loop**: ~3% of the design stage, not the 30% flop counts
  suggested. `GP.update` rebuilds for the same reason.

Measurement traps: monkeypatching into the jitted `local._core` does nothing without
`local._core.clear_cache()`; `from lagpax import covar` gives the function, not the module. The
`vs16c` ratio is **not comparable across CSV files** (host CPUs differ ~3x across SLURM nodes, so every
`bench/*.sbatch` runs `lscpu` and `bench/cpu.sbatch` pins `--constraint=amd7763`). On `ece-gpu002`
laGP's 16-core ALC ran 8.5x slower than on `uri-gpu002` while `nn` was unaffected — `omp.threads` had
not taken. Keep the 1-core row; it is the only way such a node announces itself.

## Next

### Before a first release

- [ ] **Finish the vlse license split.** Done in vlse (`vlse.optim` Apache-2.0, `vlse.functions`
      lazy and GPL-2.0-only, version 0.1.4) but not yet merged or on PyPI. Once it is, raise
      `pyproject.toml` to `jaxvlse>=0.1.4`; up to 0.1.3 the whole package is GPL-2.0-only.
- [ ] **Reserve `lagpax` on PyPI with a `0.0.1` placeholder** (not `0.1.0` — see `CLAUDE.md`).
      `pyproject.toml` declares `0.1.0`; the upload has to override it or the declaration walked back.
- [ ] **Do not weaken lagpax's own license.** It is a derivative of laGP.
- [ ] Re-check `README.md` on scope, the speed claim and the solver whenever a criterion, search or
      dependency changes.
- [ ] Decide the version for the first real release now that the API has broken twice since the
      plan was written; `0.1.0` is still unused.
- [x] CI runs the parity suite under `LAGPAX_STRICT=1`. Green on `8d8e4b4` and `3a180d3`.
- [x] CI pins laGP to 1.5-9 from the CRAN archive and fails on any other version; it used to take
      CRAN's newest (1.5-10) and fall back to the archive only if that failed.

### Measurement debt

- [ ] Re-run the GPU location sweep with the vlse solver: every GPU number above predates it, and
      vlse's line search is sequential where the old one was one `vmap`, so batched fits pay the
      slowest member's trip count.
- [ ] Re-measure the fit intercept (the 1.06s) with vlse.
- [ ] Multi-seed reruns: every GPU number is single-seed.
- [ ] A first submission of `bench/cpu.sbatch`, committed but never run.

### Phase 9: the from-scratch rewrite

Every module was rewritten rather than renamed, around three rules: JAX's own idioms (pure functions
over pytree dataclasses, no NamedTuple `__new__` overrides, no hand-built static configs), names
that say what a thing is so the code needs no comments, and type hints on everything.

| phase 8 | phase 9 |
| --- | --- |
| `GP(X, y, theta, nugget)` NamedTuple with a `__new__` override | `GaussianProcess.from_data(inputs, outputs, Hyperparameters(squared_lengthscales, nugget))`, a pytree dataclass |
| `Ki`, `ldetK`, `phi`, `b`, `Kfi`, `mask`, `dof` | `inverse_correlation`, `log_det_correlation`, `residual_sum_of_squares`, `trend_coefficients`, `inverse_trend_information`, `active`, `degrees_of_freedom` |
| `Param(init, ..., gamma=(a, b))` | `Param(initial, ..., prior=GammaPrior(shape, rate))` |
| `Prior(theta=...)`, `space="lengthscale"` | `Prior(squared_lengthscale=...)`, `kernel_space="lengthscale"` |
| `ALC`, `NN`, `IECI(fmin, w, nugget)`, `Custom` | `ActiveLearningCohn`, `NearestNeighbor`, `IntegratedExpectedConditionalImprovement(best_observed, weights, include_nugget)`, `CustomCriterion` |
| `Exhaustive`, `Ray(numrays, iters)`, `Continuous(numstart, maxit)` | `ExhaustiveSearch`, `RaySearch(rays, golden_section_steps)`, `ContinuousSearch(starts, max_iterations)` |
| `LocalGP(init, maxiter)`, `emu(Xnew, batch=)`, `emu.local(Xref, full_cov=)` | `LocalGP(initial_neighbors, max_iterations)`, `emulator(new_inputs, batch_size=)`, `emulator.predict_local(references, full_covariance=)` |
| `FitInfo(theta, ...)`, `fit_kernel`/`fit_both` | `FitResult(squared_lengthscales, ...)`, `fit_squared_lengthscales`/`fit_jointly` |
| `implicit_argmin`, `bracketed_newton`, `r_ftol` | `with_implicit_gradient`, `newton_maximize_on_interval`, `R_OPTIM_RELATIVE_TOLERANCE` |

- **Numerically the same program.** Against a snapshot written by the phase-8 code in the same
  container, 80/124 arrays are bit-identical and the rest move by at most 2e-11. The trail starts
  at the default Gamma rate, now `gammaincinv(a, mass)` instead of laGP's
  `Igamma.inv(a, mass * gamma(a))` — one ulp apart (`prior.*.d.ab`, 1e-15), which nudges the MAP
  objective and so the fitted squared lengthscales and everything downstream.
- **Tests did not move a tolerance.** Every module was ported to the new names and passes as
  before; the `Igamma.inv` test became a parity test of `GammaPrior.with_mass_below`, and the
  analytic-derivative test of the kernel went with `dcorrelation` (autodiff covers it).

### Phase 8: the hard redesign

Equivalent core functionality, lagpax's own API. Names follow what things are, not what laGP's C
entry points were called:

| laGP-shaped (phase 7) | phase 8 |
| --- | --- |
| `Local(start, end, close).condition(X, Z)` | `LocalGP(init, neighbors, candidates).fit(X, y)` |
| `Conditioned.predict(XX)`, `agp`, `lagp` | `Emulator.__call__(Xnew)`, `Emulator.local(Xref)`; no free functions |
| `Prediction(mean, s2, df, llik, Xi, fit)` + `.var` | `Prediction(mean, variance, dof, log_likelihood, covariance, neighbors, fit)` |
| `Fit(d, g, its, conv)` | `FitInfo(theta, nugget, iterations, converged)` + `.lengthscale` |
| `Prior(d=Range(start, min, max, ab, estimate), g=...)` | `Prior(theta= or lengthscale=Param(init, lower, upper, gamma, learn), nugget=...)` |
| `GP(X, Z, d, g)`, `.llik`, `.update`, `.at`, `predict(lite=)` | `GP(X, y, theta, nugget)`, `.log_likelihood`, `.append`, `.replace`, `predict(full_cov=)` |
| `covar(d, X1, X2)`, `distance` | `correlation(X1, X2, theta)`, `sqdist` |
| `mle_d`, `mle_g`, `jmle` | `fit_kernel`, `fit_nugget`, `fit_both` (in `lagpax.fit`) |

- **A prior on the conventional lengthscale is the one new capability.** `Prior(lengthscale=...)`
  puts the bounds and Gamma prior on `l = sqrt(theta / 2)`; the solve runs in `l`. With no Gamma
  prior both spaces reach the same optimum (`test_fit` checks it to 2e-3); with one they are
  different MAP estimates by construction. `from_data(space="lengthscale")` maps laGP's defaults to
  `l` and recenters the Gamma(1.5, b) prior so 95% of its mass lies below the upper bound in `l`.
- **The theta-space path is numerically the old one.** Against a baseline written by the phase-7
  code in the same container, every surviving snapshot value is bit-identical except the aGP
  variances, which move by one ulp (2e-19 absolute): the variance is now formed inside the jitted
  `_core`, where XLA may fuse `scale * dof / (dof - 2)`, instead of as an eager property afterwards.
- **A snapshot does not travel between machines.** The phase-7 code itself differs from its own
  baseline by up to 1e-11 after a container restart (different core count, different XLA CPU
  reduction order). Compare against a baseline written by the old code in the current environment.
- **Removing the conv=102 reset changed no tested behaviour.** laGP jumps to `0.9*min + 0.1*max`
  when the start sits on the lower bound; lagpax optimizes from there and reaches the interior
  start's optimum (`test_a_start_on_the_bound_is_simply_optimized`). darg's starts are interior
  anyway.

### Phase 7: what the JAX-native redesign decided

Shipped in four stages, each checked against the strict parity suite and the 124-array snapshot.
Every stage was **bit-identical** under float64; nothing numerical moved.

| was | now |
| --- | --- |
| `import lagpax` flipped `jax_enable_x64` process-wide | no global state; the input dtype rules, and float32 runs end to end |
| `rng: np.random.Generator`, defaulting to an unseeded one | `key: jax.Array`; required exactly when the call is random |
| string criteria/searches, `_Cfg`, `getattr(..., "minimize")`, factories, partials | `ALC()`, `IECI(fmin=)`, `Ray(numrays=)`: pytree dataclasses, static flags, numeric leaves |
| `Conditioned` held a NamedTuple scheme passed to `_core` via a hand-built static `_Cfg` | `Local`, `Range`, `Conditioned` are pytrees; `_core(emu, Xref, d0, lite=)` |
| `predict` owned the batching | `local` is the primitive; `predict` is `lax.map` over it, `vmap` works too |

Decisions worth keeping:

- **Strings stay as sugar.** `Local(criterion="alc", search="ray")` normalizes to objects on
  construction, so nothing downstream sees a string. Removing them would have bought purity and
  cost every laGP user the vocabulary they already have.
- **No equinox.** `jax.tree_util.register_dataclass` with a `static()` field marker covers it.
- **A missing key is an error only when a draw happens.** darg on <=1000 rows is deterministic and
  takes no key; making the key mandatory everywhere would have been ceremony.
- **float32 needs R's absolute `ftol`, not a rescaled one.** `factr * eps(float32)` is ~1.2, so
  every fit stopped after one iteration and predictions were off by O(1); `factr * eps(float64)`
  lets vlse stop on stalled progress instead, and float32 aGP then lands within 2e-3 of the truth.
  `s2` can come out negative in float32 at `g=1e-4` (the `1 + g - k'Ki k` cancellation) — that is
  float32, not a bug, and the float32 test does not promise its sign.
- **A latent bug the redesign exposed:** `float(jnp.sqrt(eps))` in `bracketed_newton` is a tracer
  under an enclosing `jit`, so the joint MLE failed inside `_core` while every eager test and the
  snapshot passed. Only the strict suite caught it. Host-side constants now use `math`.
