# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repo. Loaded every session, so it is the
operating manual: how the repo works, why things are the way they are, and what will bite you.

`ROADMAP.md` is the closed record — what shipped, what was learned, what is still open. `README.md`
is the user-facing pitch. Keep all three in sync; don't duplicate across them.

## What this is

A JAX port of the laGP R package (1.5.9), separable/`GPsep` family. Bit-parity with R is the design
goal and is enforced by `tests/`. The pitch is nonstationarity + differentiability + verified parity,
not raw speed. Extracted from the LocalVorcans prototype repo, which still holds an older,
independent local-GP stack that does **not** implement the same math — don't import assumptions from
there.

Every phase of the plan has landed and the capability list is closed; `ROADMAP.md` has the audit.
What is left is the pre-publish checklist, multi-seed reruns, and a first run of `bench/cpu.sbatch`.

### Before publishing

Full checklist in `ROADMAP.md`. The parts you must not get wrong from memory:

- **The PyPI placeholder ships as `0.0.1`, never `0.1.0`.** PyPI never lets a version number be
  reused and deleting a release does not free its filename, so burning `0.1.0` on a stub costs the
  real first release its natural number. Reserving the name matters: plain `lagp` was already
  squatted by an unrelated LLM-proxy project, so `lagpax` being free today proves nothing.
- **CI must fail on a nonzero skip count.** The parity suite is the credibility artifact, and without
  R + laGP in the runner it goes green having tested almost nothing. `.github/workflows/tests.yml`
  installs R + laGP and sets `LAGPAX_STRICT=1`, which turns any skip into a nonzero exit — keep that
  env var set, it is the whole point of the job.
- **Do not weaken the license to MIT/Apache.** See "License" below.

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
uv run --no-sync -m bench.timing --suite locations    # wall clock vs predictive-location count
uv run --no-sync -m bench.plot                        # -> bench/benchmark.png
uv run --no-sync -m bench.plot --locations timing-a100-loc.csv timing-a100-loc-big.csv
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
  the Gamma log-priors when `dab`/`gab` are non-zero. `_profile` is the MLE inner-loop objective —
  one Cholesky, two triangular solves, no explicit inverse. Gradients come from `jax.grad`, not
  hand-written derivatives.
- `prior.py` — host-side numpy/scipy port of `darg`/`garg`/`getDs`/`Igamma.inv`, behind
  `Prior.from_design`. Deliberately *not* jitted: it runs once per call and R's `quantile` type 7 ==
  numpy's default.
- `optim.py` — `lbfgsb` (projected L-BFGS, Armijo on the projected arc, R's `factr`/`pgtol` stopping
  rules), `bracketed_newton` for the nugget, and `implicit_argmin`. `bracketed_newton` takes one
  `dfun` returning `(d, d2)` rather than two callables, so the nugget MLE gets both derivatives from
  a single factorization per iteration. Neither solver reproduces R's iterate trajectory; they find
  the same KKT point. Tests assert that, plus "our llik is no worse than laGP's".
- `mle.py` — `mle_d` (with laGP's `0.9*dmin + 0.1*dmax` boundary reset, conv=102), `mle_g`, `jmle`,
  and `fit` dispatching on the prior's `estimate` flags. `jmle` carries `(d, g)` through its
  `while_loop` rather than a GP, so the alternating half-steps stop factorizing a `K` nobody reads.
- `criteria.py` — `nn`, `alc` (port of `alcGPsep`), `fisher` (port of `efiGP`/`fishGP`), and the
  `ieci(fmin=, w=, nonug=)` and `mspe(fi=)` factories. A `Criterion` carries `minimize`, so the
  greedy loop never hardcodes argmax vs argmin — and configurable criteria have to be factories
  rather than `functools.partial`, because a partial does not forward that attribute and
  `search._pick` reads it with `getattr(..., False)`. `fisher_info` and `_dpred` are laGP's
  `fishinfoGP` and `dmus2GP`, both of which laGP hand-differentiates and neither of which needs
  derivative code here.
- `blhs.py` — block Latin hypercube subsampling, port of `blhs`/`blhs.loop`. Host-side numpy like
  `prior.py`: every draw has a different row count, so the `K` fits are different shapes and there
  is nothing to batch. `bootstrap` returns a `Prior`.
- `search.py` — `exhaustive`, `ray`, `continuous`. All three take
  `(criterion, gp, Xcand, Xref, taken, step, rect)` and return one candidate index. `_block` picks
  "the offset-th *available* candidate" the way laGP's deletion from its candidate list does, by
  cumsum + `searchsorted` on the distance-ordered candidates; an argsort there would be
  `O(ncand log ncand)` at every greedy step and would dominate the search.
- `local.py` — `Local`, `Conditioned`, `lagp`/`agp`. `_core` is jitted with a `_Cfg` NamedTuple as the
  static config; `predict` maps it over predictive locations with `jax.lax.map(batch_size=)`.
  `_design` carries the candidate/design cross-covariance across greedy steps rather than recomputing
  it. `_buckets` groups the greedy steps by padded capacity — step `t` builds its GP at the next
  power of two above `t`, not at `end` — so the loop traces a logarithmic number of bodies.

The public API is flat: `__init__.py` re-exports every name and pins them in `__all__`, so a new
public function has to be added in two places there. Importing the package has a global side effect —
`jax.config.update("jax_enable_x64", True)` runs at import time, for the whole process. It stays:
every tolerance in `tests/` assumes it and laGP is float64 throughout.

### Why the mask exists

Greedy design selection grows the local design one point at a time, but XLA needs fixed shapes.
`GP.mask` marks the active rows, and there are **four** invariants, not three: masked rows/columns of
`K` are set to the identity, so `ldetK` gains 0; masked `Z` entries are zeroed so `phi` is unaffected;
masked `F` rows are zeroed so the GLS solve ignores them; and the identity block in `Ki` is killed by
masking `k`/`kx` in `_predict`/`alc`. Break any one and the padded GP silently stops matching the
unpadded one — `tests/test_gp.py::test_mask_matches_a_smaller_design`,
`tests/test_alc.py::test_alc_ignores_masked_rows` and the trend mask test in `tests/test_trend.py`
are the guards.

The invariants are exact but the *padding width* is not neutral. `ldetK` and `phi` agree to the bit
across widths, while `alc` differs by ~1.4e-16 because XLA tiles a reduction over 8 rows differently
from one over 50. That is enough to flip a near-tied argmax, and the greedy loop cascades it: at
`n=4000` two padding schemes agree on only 25 of 100 local designs. Neither is more correct, and laGP
(which grows real `k x k` matrices) is a third such point. **Judge a change here on the parity suite
and on rmse, never on whether `Xi` is unchanged.**

### Gradients through the fit

`optim.implicit_argmin` is what makes `jax.grad` work through `mle_d`/`mle_g`/`jmle`. Both solvers are
`lax.while_loop`s with no reverse-mode rule, so the answer is computed under `stop_gradient` and the
derivative attached afterwards: one Newton step from the frozen solution carries exactly the
implicit-function term, and subtracting its own `stop_gradient` leaves the **value bit-identical**.
A hyperparameter on an active bound, or one produced by the conv=102 reset, is frozen and its
gradient is exactly zero.

Do not check this gradient by finite-differencing the solver — `lbfgsb` stops on R's `factr` and
returns ~8 significant digits, so a central difference carries ~7e-5 of noise. `tests/test_diff.py`
compares against an argmin obtained by root-finding `grad llik = 0` with scipy instead, which reaches
~1e-6. See `ROADMAP.md` § "Bit-identity is not the invariant" for that and for the `jit`-vs-`vmap`
tolerance rules.

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

Shipped in `539c740`. The names previously mirrored laGP's C entry points; that grid — criterion
crossed with search — is what produced `alcGPsep`/`alcrayGPsep`/`alcoptGPsep` and would have produced
twelve names once MSPE, IECI and Fisher landed. The rename happened while only a `0.0.1` stub was on
PyPI, so there are no aliases and no deprecation window.

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

- **`condition` is the seam.** Resolving the prior is the only host-side numpy/scipy step in the
  pipeline; putting it up front is what lets everything downstream be pure JAX. Do not let prior
  resolution leak back into the per-location path. It is also where a `bootstrap(...)` prior plugs in.
- **`refine` replaces `agpsep_seq`'s kwargs forwarding**, which is exactly where a `d`-resolution bug
  lived: each pass re-ran `darg`, re-subsampling with the RNG, so passes silently disagreed on the
  prior. Its pass loop stays a Python loop — passes may name different criteria that trace to
  different functions, so it unrolls at trace time and still jits and differentiates as one graph.
- **One `Prediction` for both entry points**, ending laGP's raw-`s2`-vs-scaled-`var` split.
- **`close`** bounds the candidate set for every search, defaulting to `10*(1000+end)` for ray and
  continuous and `1000+end` for exhaustive, matching laGP.

The public surface is seventeen names: `GP`, `Prior`, `Range`, `Prediction`, `Fit`, `Local`, `agp`,
`lagp`, `fit`, `alc`, `fisher`, `ieci`, `mspe`, `nn`, `bootstrap`, `distance`, `covar`. Everything
else — `lbfgsb`, `bracketed_newton`, `implicit_argmin`, `covar_symm`, `dcovar`, `igamma_inv`,
`get_ds`, `dllik_d`, `dllik_g`, the `mle_*`/`jmle` entry points, the three searches — stays importable
from its own module, and `tests/` reaches into `lagpax.optim` and `lagpax.search` directly.

### Beyond laGP

Two things laGP cannot do, both shipped:

- **Trend models.** laGP is simple kriging everywhere, so the user must center their own response and
  the intervals omit the uncertainty in the mean. `trend="constant"` and `trend="linear"`, off by
  default, buy better-calibrated intervals — especially at design boundaries and on one-sided local
  neighbourhoods. Sold as calibration, not as a better mean; `tests/test_trend.py` tests the coverage
  claim rather than rmse.
- **Gradients through fitting, and a compilable pipeline.** `Local(...).condition(X, Z).predict(XX)`
  is an ordinary JAX function end to end: no host callbacks after `condition`, and `jax.grad` runs
  through the local hyperparameter fit by the implicit function theorem.

One capability worth knowing about because it is easy to under-use: **iterated refinement**
(`Conditioned.refine`). Design selection scores candidates at the *starting* lengthscale, so a second
pass picks neighbourhoods that respect which dimensions matter. On a strongly anisotropic 4-d problem
it cuts rmse ~1.7-2x, and `("nn", "alc")` beats `("alc", "alc")` — the first pass only exists to
supply lengthscales, so paying for ALC there buys little.

## tests/

`tests/conftest.py` provides the `R` fixture (rpy2 + laGP from `~/R/library`, skipping when either is
unavailable) and the shared designs: `small` (40x3), `medium` (600x4), `wide` (200x8), `noisy` (300x3
with observation noise). `test_covar`, `test_gp`, `test_prior`, `test_mle`, `test_criteria`,
`test_ieci` and `test_local` compare against R. `test_transforms`, `test_search`, `test_trend` and
`test_diff` are property tests — the things with no laGP equivalent to compare to.

`tests/snapshot.py` is the regression oracle, not a test module: `python -m tests.snapshot write
<path>` records 124 arrays from every entry point, `check <path>` asserts bit-identity. The durable
copy lives at `~/.cache/lagpax/snapshot.npz`. Run `check` after anything that is supposed to be
value-neutral. It is a **change detector, not a spec** — when it reports a difference the gate is the
parity suite plus rmse, not the snapshot itself.

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
every search looks flat. With it pinned, `bench/scaling.csv` shows what ray search was for: over a
64x range in `n` (1000 -> 64000) exhaustive goes 0.090 -> 0.467 s per location while ray goes
0.088 -> 0.118 and continuous 0.149 -> 0.172.

`accuracy.py` computes `darg` once and hands the same prior to both sides. Without that it measures
the RNG gap, not the port: at n=2000 the median relative error is ~1e-4 with independent priors and
~1e-8 with a shared one.

### The location sweep

**Quote `fixed + per_location * nn`, never a ratio at one `nn`** — both implementations are affine in
the predictive-location count, which is how the earlier "4.3x" claims came to be true and useless at
the same time. `bench/timing-a100-loc{,-big}.csv`, on `uri-gpu002` (A100-SXM4-80GB, Xeon 8352Y, 16
cores), `n=8000`, 250 -> 256,000 locations:

| criterion | jax A100 | laGP 16-core | break-even | asymptotic |
| --- | --- | --- | --- | --- |
| `alc` | 3.15s + 0.850ms/loc | 6.218ms/loc, no fixed term | nn ~ 590 | 7.3x |
| `nn` | 2.45s + 0.473ms/loc | 1.608ms/loc, no fixed term | nn ~ 2160 | 3.4x |

Max residual 0.09s over a 1000x range. Single seed. Below break-even laGP wins outright and the
README says so. Both rows were measured at `batch=64` — the old default — so they are a floor on the
port, not its best. The ~3s fixed term is **not** compile (`timeit` absorbs that) and not host-side
(`Prior.from_design` is 70ms, `condition` ~0ms); it is the GPU running launch-latency bound while the
`lax.map` batch count is small.

### The batch sweep

`bench/timing-a100-loc-b{64,256,1024,4096}.csv`, on `ece-gpu002` (A100-**PCIE-40GB**, a different
card from the location sweep), `n=8000`, `nn` of 16,000 and 64,000:

| `batch` | alc ms/loc | alc fixed | nn ms/loc | nn fixed |
| --- | --- | --- | --- | --- |
| 64 | 0.819 | 3.02s | 0.454 | 2.50s |
| **256** | **0.568** | 3.01s | **0.316** | 2.00s |
| 1024 | 0.529 | 3.93s | 0.303 | 2.30s |
| 4096 | 0.521 | 5.77s | 0.446 | 0.88s |

So the default is 256, in `Conditioned.predict`, `Conditioned.refine` and `agp`: 1.44x off the
per-location rate for both criteria, at the lowest fixed term and the smallest peak memory of the
three that are competitive. 4096 regresses `nn` outright and is the memory ceiling announcing itself
— at `close=1050, end=50` a 4096-wide batch is ~1.7GB of candidate correlations against ~107MB at 256.

### Reading the timing CSVs

The `vs16c` ratio is **not comparable across CSV files** — the GPU runs were done on different SLURM
nodes whose host CPUs differ by ~3x. Only GPU-to-GPU wall clock is clean.

Worse than a slow CPU: a node where laGP's OpenMP silently does not get the cores. On `ece-gpu002`,
with `-c 16` granted, laGP's 16-core ALC ran at 53.2 ms/location against 6.27 on `uri-gpu002` — 8.5x,
far more than any CPU generation gap. The tell is that `nn` was *unaffected* on the same two runs
(1.47 vs 1.61 ms/location), so it was thread count and not clock speed. **A 16-core laGP number is
only trustworthy next to a 1-core number from the same node**: on a healthy node the ratio is ~6x, and
anything near 1x means `omp.threads` did not take and the whole file is scrap.

`OPENBLAS_NUM_THREADS=1` is set at the top of `bench/common.py`, before jax or R load, so laGP's
`omp.threads` is the only thing that varies between the 1-core and N-core rows.

## Performance

**Read `ROADMAP.md` § "Performance: the workload is latency-bound" before optimizing anything.** Short
version: a 25x fp64-throughput range across GTX 1080 / L40S / V100 produces only a 2.9x spread in wall
clock, so the workload is bound by kernel-launch latency and the sequential greedy loop, not
arithmetic. Buying a better GPU does not fix that, and neither does shrinking the arithmetic — a
gemm-based `covar` with a 1.4-2.5x faster kernel made the pipeline 19% *slower*. That file lists what
worked, what did not, and two measurement traps that will waste an afternoon each.

After the cross-covariance cache the profile at `n=4000`, `end=50`, 100 locations on 16 CPU cores is
0.38s neighbours + build + predict, 0.78s design selection, 4.31s fit. The fit is the cost now, and
63% of its lengthscale solve is a fixed intercept no iteration count touches.

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
- `lagpax/__init__.py` does `from .covar import covar`, which shadows the submodule: both
  `from lagpax import covar as m` and `import lagpax.covar as m` give you the *function*. Use
  `sys.modules["lagpax.covar"]`.
- Monkeypatching a function that `local._core` closes over does nothing — `_core` is jitted and the
  closure is not part of its cache key. Call `local._core.clear_cache()` between configurations.
- There is no `gh` CLI on this cluster. Check CI with
  `curl -s https://api.github.com/repos/davidesartor/lagpax/actions/runs?per_page=3`.
