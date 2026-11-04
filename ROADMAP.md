# lagpax roadmap

Status, rules, lessons, and what is next. `CLAUDE.md` is the operating manual and wins on any fact
about how the current code works; this file owns every measurement and every decision record.

## Status

The laGP port is **complete**: audited against laGP 1.5.9's 51 namespace exports, nothing in scope is
unported (re-run with `.libPaths("~/R/library"); print(sort(getNamespaceExports("laGP")))`).
Box-constrained optimization is delegated to `vlse`, and parity now also covers real optimization
benchmark surfaces. What is left before a first release is in § Next.

| phase | | commit |
| --- | --- | --- |
| 0 | API rename — laGP's C-shaped names to the surface in `CLAUDE.md` § API | `539c740` |
| 1 | ray + continuous search (`search.py`) | `42bf35a` |
| 2 | IECI (`criteria.ieci`) | `42bf35a` |
| 3 | trend models (`trend="constant"`/`"linear"`) | `42bf35a` |
| 4 | differentiable fitting (`optim.implicit_argmin`) | `f86c421` |
| 5 | MSPE, Fisher, block-bootstrap subsampling | `f6b4579` |
| 6 | L-BFGS-B from `vlse.optim.minimise`; parity on vlse's SFU surfaces (`tests/test_vlse.py`) | branch `ccr-7f9aed99-rlk3k1` |

**Explicitly not shipping**, do not implement even if it looks easy: `aGP.parallel` (XLA decides
parallelism), `randLine` (a visual diagnostic), `optim.auglag`/`optim.efi` (constrained blackbox
optimization — a separate package's job), `discrep.est`/`fcalib` (computer-model calibration — a
distinct research workflow).

## Rules that govern new work

- **Parity is the credibility claim.** Anything with a laGP equivalent gets a value-level parity test
  against R; non-exported R functions are reachable as `laGP:::alcGPsep` etc. Anything *without* one
  gets property tests — analytic special cases, finite differences, invariances.
- **Parity is a property of the laGP-equivalent configuration**, not of the package. Every divergent
  flag needs a documented default that reproduces laGP and a test asserting it still does.
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
- **Configurable criteria are factories, never `functools.partial`** — a partial does not forward
  `minimize`, so it would silently maximize a criterion meant to be minimized.
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

### Candidate phase 7: a JAX-native API (proposed, not decided)

The core is already pytrees and pure functions after `condition`. The boundaries are not:

| today | proposed |
| --- | --- |
| `import lagpax` flips `jax_enable_x64` process-wide | compute in the input dtype; warn when x64 is off |
| `rng: np.random.Generator` in `Prior.from_design`, `bootstrap`, `agp` | explicit `key: jax.Array`; the R-exact numpy path stays for parity |
| `emu.predict(XX, batch=)` owns the `lax.map` | expose the per-location function; the user picks `vmap`/`lax.map`/`shard_map` |
| string criteria/searches + `_Cfg` + `getattr(..., "minimize")` | small modules with static fields (`ALC()`, `IECI(fmin=)`, `Ray()`) |
| `Conditioned` mixes data, prior and config | a pytree state plus a hashable scheme |

Breaking changes are cheap only until `0.1.0` ships. Parity tests and the snapshot must hold through
it value-for-value.
