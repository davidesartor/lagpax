# lagpax roadmap — closed

Every phase has landed and the capability list is closed, so this is no longer an execution plan. It
is the record: what shipped, what was learned, and what is still open. `CLAUDE.md` is the operating
manual and wins on any fact about the current repo.

## What shipped

| phase | | commit |
| --- | --- | --- |
| 0 | API rename — laGP's C-shaped names to the surface in `CLAUDE.md` § API | `539c740` |
| 1 | ray + continuous search (`search.py`) | `42bf35a` |
| 2 | IECI (`criteria.ieci`) | `42bf35a` |
| 3 | trend models (`trend="constant"`/`"linear"`) | `42bf35a` |
| 4 | differentiable fitting (`optim.implicit_argmin`) | `f86c421` |
| 5 | MSPE, Fisher, block-bootstrap subsampling | `f6b4579` |

Audited against laGP 1.5.9's 51 namespace exports; nothing in scope is unported. Re-run the audit
with `.libPaths("~/R/library"); print(sort(getNamespaceExports("laGP")))`.

**Explicitly not shipping**, do not implement even if it looks easy: `aGP.parallel` (cluster/thread
parallelism — XLA decides), `randLine` (a visual diagnostic), `optim.auglag`/`optim.efi` (constrained
blackbox optimization — a separate package's job, and JAX users already have optimizers),
`discrep.est`/`fcalib` (computer-model calibration — a distinct research workflow, not part of the
emulator).

## Rules that still govern new work

- **Parity is the credibility claim.** Anything with a laGP equivalent gets a value-level parity test
  against R; non-exported R functions are reachable as `laGP:::alcGPsep` etc. Anything *without* one
  gets property tests instead — analytic special cases, finite differences, invariances.
- **Parity is a property of the laGP-equivalent configuration**, not of the package. Every divergent
  flag needs a documented default that reproduces laGP and a test asserting it still does.
- **Where our optimizer finds the same KKT point by a different path, assert that** — plus "our llik
  is no worse than laGP's". The pattern is in `tests/test_mle.py`; copy it, do not invent one.
- **Tolerances are non-uniform on purpose** and are not a dial. Do not loosen one to pass, do not
  tighten one to make a point. See `CLAUDE.md` § tests.
- **Stop and report** rather than guessing if a parity test fails beyond tolerance for a reason that
  is not obviously ours, laGP's own behaviour looks like a bug, or the math turns out to be wrong.
  Never silently loosen a tolerance, skip a test, or `xfail` a parity test to get a green run.

## What we learned

### Bit-identity is not the invariant; parity is

Several correct changes move results in the last ulps, and treating that as a regression would have
blocked all of them. `tests/snapshot.py` is a *change detector*, not a spec — it answers "did this
move anything", and when it says yes the gate is the parity suite plus rmse.

- **Padding width is not neutral.** The mask invariants hold exactly (`ldetK` and `phi` differ by 0
  across widths) but ALC's reduction reassociates by ~1.4e-16, which flips near-tied argmaxes and
  cascades through the greedy loop. At `n=4000` two padding schemes agree on 25 of 100 local designs.
  laGP, growing real `k x k` matrices, is a third such point. None is more correct.
- **`jit` preserves the optimizer trajectory (1e-9); `vmap` does not.** A batched `while_loop` shares
  its trip count, so one ulp flips `lbfgsb`'s relative-change test and the iterates separate by the
  solver's own resolution `sqrt(factr*eps)` ~ 4.7e-5 in `d`. Compare the objective, not the argmin —
  at ~1e-5, not at `factr*eps`, since the likelihood is stationary at both stopping points and
  inherits a second-order gap (~2e-6 measured). `factr*eps` bounds only the last accepted step.
- **Do not finite-difference the solver.** `lbfgsb` stops on R's `factr=1e7`, so it returns ~8
  significant digits and a central difference carries ~7e-5 of noise; tightening to `factr=1e3` only
  moves the floor to 6.8e-5. `tests/test_diff.py` compares against an argmin from root-finding
  `grad llik = 0` with scipy instead, which reaches ~1e-6. Reuse that for any gradient test through
  an optimizer.

### Design departures worth keeping

- **A straight-through Newton step, not `jax.custom_vjp`**, for implicit gradients:
  `x - (step - stop_gradient(step))` from a frozen `x` carries exactly the implicit-function
  derivative while returning the solver's value bit-identically. `custom_vjp` would have needed
  `nondiff_argnums` over a closure holding tracers and float0 plumbing for `GP`'s boolean `mask`.
- **`refine` keeps a Python loop.** `lax.scan` is impossible, not merely unnecessary: passes may name
  different criteria, which trace to different functions. It unrolls at trace time and still jits and
  differentiates as one graph.
- **Configurable criteria are factories, never `functools.partial`.** A partial does not forward
  `minimize`, and `search._pick` reads it with `getattr(..., False)` — so a partial would silently
  maximize a criterion meant to be minimized. Configurable *searches* can still be partials.
- **The parity targets for MSPE/Fisher are isotropic** (`newGP`, not `newGPsep`). A separable GP with
  every `d_k` equal is that GP exactly, and the port's lengthscale derivative is the directional one
  along the all-ones direction.
- **laGP contradicts itself and the port copies it**: `efiGP` computes `sq(dmu)/s2` where `mspeGP`
  computes `sq(dmu/s2)` in the otherwise identical Fisher denominator. Both reproduced as written.
- **`Xstart` for ray search is deterministic**, not R-RNG sampled, so no exact parity exists there;
  `tests/test_search.py` asserts a property instead.

### Performance: the workload is latency-bound, and that is the whole story

A 25x fp64-throughput range across GTX 1080 / L40S / V100 produces a 2.9x spread in wall clock. The
binding constraints are kernel-launch latency and the sequential greedy loop, not arithmetic. Every
subsequent measurement agreed, including the ones that were supposed to disagree.

**Measure with `bench.timing --suite locations`, never at one location count.** Both engines are
affine in the predictive-location count — laGP with no fixed term, lagpax with a few seconds of one —
so a ratio is a statement about where on the curve you stood. Quote `fixed + per_location * nn`. The
numbers and break-evens are in `CLAUDE.md` § The location sweep.

Profile on 16 CPU cores at `n=4000`, `end=50`, 100 locations, after the cross-covariance cache:
0.38s neighbours + build + predict, 0.78s design selection, **4.31s fit**. The fit is the cost now.

What worked:

| | |
| --- | --- |
| stop inverting `K` in the MLE inner loop (`gp._profile`) | ~2x fewer flops in a fitted local GP (`b52be87`) |
| cache the candidate/design cross-covariance across greedy steps | design selection no longer the hot stage (`8520f42`) |
| `lax.map` batch default 64 -> 256 | 1.44x off the per-location rate on an A100 (`b3d4995`) |
| fuse `bracketed_newton`'s two derivative callables into one `jvp` | -7% on the g-solve, -3.5% on `jmle` |
| pad the greedy design to a power of two | neutral at `end=50`, 1.16-1.20x at `end=200` (`d62668c`) |

What did not, and should not be retried without new evidence:

- **A gemm-based `covar`.** The kernel itself is 1.4-2.5x faster (max rel err 7.89e-15) and the
  pipeline got **19% slower**. XLA's fusion of the broadcast-reduce form beats BLAS3 here. This is
  the clearest single statement of the latency diagnosis: making the hottest kernel faster made the
  program slower.
- **fp32 for the ALC selection stage.** Follows from the above — selection is not arithmetic-bound,
  so halving the precision buys nothing and costs the argmax its ties. Not separately measured;
  listed as a lever for a long time and refuted by the gemm result.
- **Fewer Armijo probes in `lbfgsb`** (21 -> 4): 0% faster, and it destroys the answer (3.3e-2) as
  `its` collapses to 1.2. A blocked line search was tried and reverted earlier for the same reason:
  the probes go out as one `vmap` and XLA spreads them, so saving flops saves no time.
- **Looser `factr`** (1e7 -> 1e13): cuts iterations 2.1x for only -12% wall clock, at 1.6e-4 of
  accuracy. The sweep fits 0.039 s/iteration with a **1.06s intercept** — 63% of the d-solve is fixed
  cost no iteration count touches. `implicit_argmin`'s Hessian was the named suspect and removing it
  changed nothing; the intercept is still unexplained.
- **A lower `maxit`** (100 -> 8): slower than 30.
- **Warm-starting `d0`** from a previous pass: 75% slower, `its` 15.8 -> 18.3. Note the structural
  obstacle anyway — `local._core` takes one `d0` for *both* design selection and the fit start, so
  warm-starting the fit alone would need a separate `d_fit_start`.
- **Rank-1 Cholesky update in the greedy loop** instead of a rebuild: worth ~3% of the design stage,
  not the 30% a GPU-era reading of the flop counts assumed. `GP.update` rebuilds for the same reason.

Two measurement traps that cost real time:

- **Monkeypatching into a jitted function does nothing.** `local._core` caches its trace and a
  closure is not part of the cache key; call `local._core.clear_cache()` between configurations or
  every row of the sweep is the same run.
- **`lagpax/__init__.py` does `from .covar import covar`,** which rebinds the package attribute and
  shadows the submodule. Both `from lagpax import covar as m` and `import lagpax.covar as m` give you
  the function. Use `sys.modules["lagpax.covar"]`.

**Benchmark hygiene.** `bench/accuracy.py` must keep computing `darg` once and handing the same prior
to both sides — with independent priors the median relative error at n=2000 is ~1e-4 against ~1e-8
shared, i.e. it measures the RNG gap rather than the port. `OPENBLAS_NUM_THREADS=1` is set at the top
of `bench/common.py` before jax or R load; keep it there. The `vs16c` ratio is **not comparable
across CSV files** — GPU runs used SLURM nodes whose host CPUs differ by ~3x, which is why every
`bench/*.sbatch` runs `lscpu` and `bench/cpu.sbatch` pins `--constraint=amd7763`. Keep the 1-core row
(`--r-threads 1 all`): a 16-core-to-1-core ratio near 1x instead of ~6x is the only way a node that
never gave laGP its threads announces itself, and it invalidates the whole file.

## What is open

Not features. The capability list is closed and CI runs green on `origin/main`.

- [ ] **Reserve `lagpax` on PyPI with a placeholder upload.** Plain `lagp` was already squatted by an
      unrelated LLM-proxy project, so the name is not safe just because it is free today.
      `uv build && uv publish` (needs `UV_PUBLISH_TOKEN`, or `--token`). Ship it as version **`0.0.1`,
      not `0.1.0`** — PyPI never lets a version number be reused and deleting a release does not free
      its filename, so burning `0.1.0` on a stub costs the real first release its natural number. A
      stub `__init__.py` plus a README pointing at the repo is enough. Note `pyproject.toml` declares
      `version = "0.1.0"`; the upload has to override it or the declaration has to be walked back.
- [ ] **Do not weaken the license.** LGPL-2.0-or-later, inherited from laGP — this is a derivative
      work, so relicensing to MIT/Apache is not available without Gramacy's permission. Keep `NOTICE`
      accurate as modules are added.
- [ ] Multi-seed reruns: every GPU number reported so far is single-seed.
- [ ] An actual submission of `bench/cpu.sbatch`, which is committed but has never run.
- [x] Phase 0 landed before `0.1.0` (`539c740`), while only a stub could have been published.
- [x] CI runs the parity suite under `LAGPAX_STRICT=1`, so a skip is a nonzero exit. Green on
      `8d8e4b4` and `3a180d3`. No `gh` CLI on this cluster — check with
      `curl -s https://api.github.com/repos/davidesartor/lagpax/actions/runs?per_page=3`.
- [x] `README.md` accurate on scope, `aGP.seq` -> `emu.refine`, differentiability and the speed claim.
      Re-check it whenever a criterion or search is added.
