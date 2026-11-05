# lagpax

Local approximate Gaussian processes in JAX — jittable, vmappable, differentiable, and
verified value-for-value against the [laGP](https://cran.r-project.org/package=laGP) R package.

`lagpax` builds a separate GP for every predictive location from a small local design chosen out of
the full dataset. That makes it a **nonstationary** emulator without a nonstationary kernel: each
prediction gets its own lengthscales, estimated locally. Predictions are independent, so the whole
thing is one batched XLA call.

## Why this exists

- **No local-GP option in JAX.** GPJax and tinygp cover exact and inducing-point GPs; neither does
  local approximation. If your model already lives in JAX, this composes with it.
- **Differentiable.** `jax.grad` works through the whole emulator, the local hyperparameter fit
  included — the optimizers are differentiated implicitly rather than by taping their iterates. laGP's
  C/OpenMP implementation cannot do this, and it is what makes local GPs usable inside a larger JAX
  pipeline — BO loops, gradient-based design, hybrid NN-surrogate models.
- **Parity, not approximation.** The `tests/` suite compares against R's laGP call by call. This is a
  port you can trust to reproduce published laGP results, not a re-derivation that drifts.

## Install

```bash
uv add lagpax                       # runtime
uv add "lagpax[gpu]"                # + CUDA 12 wheels
uv sync --extra tests               # + pytest and rpy2 for the parity suite
```

## Quickstart

```python
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)   # laGP is float64; lagpax follows your dtype
import lagpax

X = np.random.uniform(size=(2000, 4))
Z = np.sum(np.sin(4 * X), axis=1)
XX = np.random.uniform(size=(500, 4))

key = jax.random.key(0)    # laGP's darg subsamples designs over 1000 rows; no hidden RNG here
out = lagpax.agp(X, Z, XX, key=key, start=6, end=50, criterion="alc")

out.mean   # (500,) predictive mean
out.var    # (500,) predictive variance
out.Xi     # (500, 50) local design indices, 0-based
```

`criterion` and `search` are a grid, not a list of methods. `criterion="nn"` takes nearest
neighbours, `criterion="alc"` grows the design greedily by Active Learning Cohn (Gramacy & Apley,
2015), and `criterion=lagpax.IECI(fmin=...)` scores candidates by expected improvement instead of
variance, for when the emulator serves optimization. `criterion=lagpax.MSPE(fi=True)` adds the error
from re-estimating the lengthscale once the point is added, and `criterion="fisher"` picks the point
that most sharpens the hyperparameter estimate rather than the prediction — useful when the local fit
is unstable, and the one criterion that needs no reference set. `search="exhaustive"` scores every
candidate; `search="ray"` and `search="continuous"` (or `lagpax.Ray(numrays=...)`,
`lagpax.Continuous(numstart=...)`) cost a fixed number of criterion evaluations per greedy step
regardless of how many candidates there are, which is what makes the scheme usable on large datasets.
A plain function `(gp, Xcand, Xref) -> scores` drops in as a criterion, or subclass
`lagpax.Criterion`.

Criteria and searches are small JAX pytrees. Settings that change the program (`MSPE.fi`,
`Ray.numrays`) are static; settings that are numbers are leaves, so a Bayesian-optimization loop
that moves `IECI.fmin` every step reuses one compiled program.

Reuse the settings and the design-derived prior across calls by conditioning once:

```python
emu = lagpax.Local(start=6, end=50, criterion="alc").condition(X, Z)
out = emu.predict(XX, batch=256)
out = emu.refine(XX, passes=("nn", "alc"))   # warm-start each location from its own fit
one = emu.local(XX[:1])                      # a single reference set, as laGPsep
```

`emu.local` is the per-location primitive, and `predict` is `jax.lax.map` over it. To batch
differently, map it yourself: `jax.vmap(lambda x: emu.local(x[None]))(XX)`.

On a design too large to fit globally, `bootstrap` estimates one lengthscale from many small block
Latin hypercube subsamples (`blhs.loop`) and hands it back as a prior, either as a starting point or
— as laGP uses it — held fixed:

```python
prior = lagpax.bootstrap(X, Z, m=2, K=32, key=key)         # -> Prior, d.start from the subsamples
prior = prior.at(d=lagpax.Range(estimate=False))           # laGP's use: one global, fixed d
out = lagpax.agp(X, Z, XX, prior=prior)
```

`condition` resolves the prior — the one host-side step — so everything after it is pure JAX.
The emulator it returns is itself a pytree (the data and prior are leaves, the scheme's sizes are
static), so it can be passed into `jax.jit`, rebuilt on new data with `emu._replace(Z=...)`, and
differentiated through as a whole:

```python
prior = lagpax.Prior.from_design(X, Z, key=key)        # resolved once, on the host
scheme = lagpax.Local(start=6, end=50, search="ray")

def total(Z):
    return scheme.condition(X, Z, prior=prior).predict(XX).mean.sum()

grad = jax.grad(total)(Z)                              # jax.jit(total) works too
```

Gradients run through the fitted hyperparameters by the implicit function theorem at the optimum, so
they cost one small Hessian solve rather than a tape over the optimizer's iterates. A hyperparameter
resting on a bound — a nugget pinned at its floor on noiseless data, or laGP's `conv=102` lengthscale
reset — is an active constraint, and its gradient is reported as exactly zero rather than smoothed
over.

## Scope

The separable (`GPsep`) family of laGP 1.5.9: GP construction, likelihood and predictive equations,
design-derived priors (`darg`/`garg`), lengthscale and nugget MLE/MAP (`mleGPsep`, `jmleGPsep`),
ALC (`alcGPsep`) with ray and continuous search (`alcrayGPsep`, `alcoptGPsep`), IECI (`ieciGPsep`),
and the local schemes `laGPsep`, `aGPsep` and `aGP.seq`. Isotropic `GP` routines
are a special case of the separable ones — pass a scalar `d` — which is also where laGP keeps MSPE
(`mspeGP`) and the Fisher-information criterion (`fishGP`), both ported here, along with block Latin
hypercube subsampling (`blhs`, `blhs.loop`) as `lagpax.bootstrap`. laGP's constrained-optimization
(`optim.auglag`) and calibration (`fcalib`) layers are out of scope.

The public surface is the GP and its fit (`GP`, `Prior`, `Range`, `Prediction`, `Fit`, `fit`,
`bootstrap`, `distance`, `covar`), the local scheme (`Local`, `Conditioned`, `agp`, `lagp`), the
criteria (`Criterion`, `ALC`, `NN`, `MSPE`, `IECI`, `Fisher`, `Custom`, and the instances `alc`,
`nn`, `fisher`) and the searches (`Search`, `Exhaustive`, `Ray`, `Continuous`). Everything else
stays importable from its own module (`lagpax.optim.implicit_argmin`, `lagpax.mle.jmle`).

Conventions follow laGP throughout — correlation `exp(-sum_k (x1_k - x2_k)^2 / d_k)`, nugget `g` on
the diagonal, scale profiled out as `phi = Z' Ki Z`, Student-t predictive. Two deliberate
differences: `Xi` indices are 0-based (R is 1-based), and randomness takes an explicit
`jax.random` key where R uses its global RNG. Computation follows the input dtype; laGP parity
needs `jax_enable_x64`, which lagpax does not switch on for you. One API difference:
laGP's `laGPsep` returns a raw `s2` while `aGPsep` returns `s2 * df/(df - 2)`; here both return a
`Prediction` carrying `s2`, `df` and a `var` property, so the conversion is explicit.

The default `trend="zero"` is **simple kriging**, matching laGP: the process is zero-mean with no
trend basis, so you should center `Z` yourself — an uncentered response inflates `phi`, which makes
`s2` wrong and not just the mean. `trend="constant"` and `trend="linear"` profile out a mean
alongside the scale (ordinary and universal kriging). They are not in laGP, and they are sold as
calibration rather than accuracy: simple kriging omits the variance contribution from not knowing
the trend coefficients, so its intervals are too narrow, especially at design boundaries.

## Parity

```bash
module load r/4.4.0 && export R_HOME=$(R RHOME)
uv sync --extra tests
uv run --no-sync pytest tests -q
```

Every module is compared against R. Tolerances are not uniform, on purpose: `mean`, `llik` and `phi`
match to ~1e-12, while `s2` is computed as `1 + g - k' Ki k`, a cancellation whose absolute error
tracks the O(1) intermediate rather than the ~1e-3 result, so it is checked at `rtol=1e-8`.

The optimizers (L-BFGS-B from [vlse](https://github.com/davidesartor/vlse) for lengthscales,
`bracketed_newton` for the nugget) do not reproduce R's iterate trajectory — they find the
same KKT point. Tests assert that, plus that our likelihood is never worse than laGP's. On noiseless
designs the nugget MLE pins to its lower bound in a badly conditioned regime where the two disagree
in the 8th digit and ours is *higher*; those cases are tested as "no worse", and interior-optimum
parity is tested on a separately generated noisy design.

Beyond random designs, `tests/test_vlse.py` runs the comparison on eight benchmark surfaces from the
SFU library (Branin, Hartmann, Michalewicz, Ackley, Rosenbrock, Goldstein-Price, Shekel). The joint
MLE is never worse than laGP's, and is often several log-likelihood units better. aGP predictions
agree to 1e-6 or better at the median on every surface. On the multimodal ones (Michalewicz, Ackley),
a few locations' local fits settle in a different likelihood basin than laGP's, without a systematic
difference in rmse.

## Performance

A single speedup number would be dishonest here, because the answer depends almost entirely on how
many locations you predict. Both implementations are affine in that count, so the useful form is a
fixed cost plus a per-location cost. On one A100 node (`n=8000`, 4-d, `end=50`, laGP on 16 cores of
the same node's Xeon 8352Y), sweeping 250 to 256,000 locations:

| criterion | lagpax, A100 | laGP, 16 cores | break-even | large-`nn` speedup |
| --- | --- | --- | --- | --- |
| `alc` | 3.15 s + **0.850 ms**/loc | 6.218 ms/loc | ~590 locations | **7.3x** |
| `nn` | 2.45 s + **0.473 ms**/loc | 1.608 ms/loc | ~2,200 locations | **3.4x** |

Those fits hold to within 0.09 s across a 1000x range in the location count, and predict the
256,000-location times from data taken below 16,000. laGP has no fixed term; we have a few seconds
of one, so below the break-even laGP wins outright — at 250 locations it is 2x ahead on ALC and 5x on
nearest-neighbour. Above it the per-location rate decides, and the advantage is larger for the more
expensive criterion, which is the case for shipping MSPE, IECI and Fisher rather than only ALC.

That table is conservative: it was taken at `batch=64`, and a later sweep on a second A100 found the
per-location rate falls **1.44x** going to `batch=256` for both criteria, which is now the default.
The break-evens move down with it. We have not re-run the headline table on its original node, so
rather than multiply two numbers from two machines together, the table stands as measured.

Read the ratios with care. laGP's 16-core time for one fixed case ranged over 8.5x across the host
CPUs we have benchmarked on, and on the worst of them the `nn` criterion was unaffected while ALC
was not — that node was not giving laGP its threads, and the run was discarded rather than published.
A speedup quoted against a different machine than it was measured on means nothing. `bench/*.csv` are
per-node, every `bench/*.sbatch` records the host CPU, and the numbers above are single-seed.

Three GPUs spanning a 25x range in fp64 throughput produce only a 2.9x spread in wall clock, which
says the workload is bound by kernel-launch latency and the sequential greedy loop, not by
arithmetic. That also shows up on CPU: padding the greedy design to a power of two rather than to
`end` cuts its average matrix from 50 rows to 34 and changes nothing at `end=50`, while paying off
1.2x at `end=200`.

The arithmetic itself is roughly half what it was: the MLE inner loop evaluates the likelihood
through a Cholesky alone rather than rebuilding the GP, whose explicit `K` inverse only ALC needs.
By XLA's own cost model one local GP at `end=50` went from 3.21e7 to 1.68e7 flops with ALC design
selection, and 2.58e7 to 1.04e7 with nearest-neighbour. Design selection, untouched, is now the
larger half of what remains.

Reproduce with `uv run -m bench.timing` and `uv run -m bench.accuracy`. Note that `bench/common.py`
pins `OPENBLAS_NUM_THREADS=1` before jax or R load, and that `accuracy.py` hands both sides the same
`darg` prior — without that it measures the RNG gap rather than the port.

## Related work

- [laGP](https://cran.r-project.org/package=laGP) — the original, in R and C/OpenMP.
- [Scaled Vecchia](https://github.com/katzfuss-group/scaledVecchia) — Katzfuss, Guinness & Lawrence
  (2020). A valid joint model rather than a per-location predictor, and generally the better choice
  when you need a likelihood, joint predictive draws, or a global fit. `lagpax` is preferable where
  nonstationarity dominates, or where you want per-location UQ without a global fitting stage.
- [GPJax](https://github.com/JaxGaussianProcesses/GPJax), [tinygp](https://tinygp.readthedocs.io/) —
  exact and sparse variational GPs in JAX.

## License and citation

Same license as laGP, inherited: this is a translation of laGP's source, hence a derivative work.
laGP declares a bare `License: LGPL`, which CRAN reads as version 2 or later, so `lagpax` is
**LGPL-2.0-or-later**. See `LICENSE` and `NOTICE`.

If you use this, cite the methods papers as well as the software:

> Gramacy, R. B. and Apley, D. W. (2015). Local Gaussian Process Approximation for Large Computer
> Experiments. *Journal of Computational and Graphical Statistics*, 24(2), 561-578.

`lagpax` is not affiliated with or endorsed by the laGP authors.
