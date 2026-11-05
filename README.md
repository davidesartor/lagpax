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

jax.config.update("jax_enable_x64", True)   # float64, as laGP; lagpax follows your dtype
import lagpax

inputs = np.random.uniform(size=(2000, 4))
outputs = np.sum(np.sin(4 * inputs), axis=1)
new_inputs = np.random.uniform(size=(500, 4))

key = jax.random.key(0)    # the default prior subsamples designs over 1000 rows; no hidden RNG
model = lagpax.LocalGP(neighbors=50, criterion="alc")
emulator = model.fit(inputs, outputs, key=key)
prediction = emulator(new_inputs)

prediction.mean                      # (500,)
prediction.variance                  # (500,) Student-t predictive variance
prediction.degrees_of_freedom        # (500,)
prediction.neighbors                 # (500, 50) training rows each local GP used
prediction.fit.squared_lengthscales  # (500, 4) each local GP's fitted kernel parameter
```

Each row of `new_inputs` gets its own GP on `neighbors` training points. The first
`initial_neighbors` (default 6) are its nearest neighbours; the rest are chosen one at a time by
the `criterion`:

- `"nn"` / `lagpax.NearestNeighbor()` — nearest neighbours only.
- `"alc"` / `lagpax.ActiveLearningCohn()` — the point that most reduces predictive variance at the
  target (Gramacy & Apley, 2015). The default.
- `lagpax.IntegratedExpectedConditionalImprovement(best_observed=...)` — for when the emulator
  serves optimization.
- any function `(gp, candidates, references) -> scores`, or a `lagpax.Criterion` subclass.

and looked for by the `search`: `"exhaustive"` scores every candidate, while `"ray"` and
`"continuous"` (`lagpax.RaySearch(rays=...)`, `lagpax.ContinuousSearch(starts=...)`) cost a fixed
number of evaluations per step however many candidates there are, which keeps large datasets
tractable.

### The kernel and its prior

The kernel is `exp(-sum_k (x_k - x'_k)^2 / s_k)` plus a `nugget` on the diagonal, with the scale
profiled out; `s` are squared lengthscales, and `FitResult.lengthscales` gives the conventional
`l = sqrt(s / 2)`. Each local GP fits `s` (and optionally the nugget) by MAP under a `Prior`, whose
defaults come from the data:

```python
prior = lagpax.Prior.from_data(inputs, outputs, key=key)        # Gamma prior on s, laGP's default
prior = lagpax.Prior.from_data(inputs, outputs, key=key, kernel_space="lengthscale")  # ... on l
prior = prior.replace(nugget=lagpax.Param(learn=True))          # also learn the nugget
emulator = model.fit(inputs, outputs, prior=prior)
```

The two kernel spaces have the same likelihood but different MAP estimates, since a Gamma density
on `l` is not one on `s`. The default, squared lengthscales, reproduces laGP exactly.

### Composing with JAX

`fit` resolves the prior — the one host-side step — and returns an `Emulator`, which is a pytree:
the data and prior are leaves, the model's sizes are static. So it crosses `jax.jit`, takes new
data with `emulator._replace(outputs=...)`, and differentiates as a whole:

```python
prior = lagpax.Prior.from_data(inputs, outputs, key=key)
model = lagpax.LocalGP(neighbors=50, search="ray")

def total(outputs):
    return model.fit(inputs, outputs, prior=prior)(new_inputs).mean.sum()

gradient = jax.grad(total)(outputs)                 # jax.jit(total) works too
```

Gradients run through each local hyperparameter fit by the implicit function theorem at the
optimum, so they cost one small Hessian solve rather than a tape over the optimizer's iterates. A
hyperparameter resting on a bound — a nugget pinned at its floor on noiseless data — is an active
constraint, and its gradient is reported as exactly zero rather than smoothed over.

`emulator.predict_local(references)` is the per-point primitive (one GP for a whole reference set,
`full_covariance=True` for a joint covariance); `emulator(new_inputs, batch_size=256)` is
`jax.lax.map` over it, and `jax.vmap(lambda x: emulator.predict_local(x[None]))(new_inputs)` batches
it with vmap instead. `emulator.refine(new_inputs, passes=("nn", "alc"))` predicts in passes, each
starting every point from the last pass's fit — designs are chosen at the *starting* squared
lengthscales, so this picks neighbourhoods that respect which inputs matter, cutting rmse ~1.7-2x
on anisotropic problems.

Criteria and searches are small pytrees too: settings that change the program
(`include_nugget`, `rays`) are static, settings that are numbers are leaves, so a
Bayesian-optimization loop that moves `best_observed` every step reuses one compiled program.
Nothing in lagpax touches global JAX state, and there is no hidden randomness: anything random
takes a `jax.random` key.

The lower level is public too: `lagpax.GaussianProcess.from_data(inputs, outputs,
lagpax.Hyperparameters(squared_lengthscales, nugget))` is an exact GP with `log_likelihood`,
`predict`, `with_hyperparameters` and `with_observations`, and `lagpax.fit_hyperparameters(gp,
prior)` fits it.

## Scope

Equivalent to the separable (`GPsep`) core of laGP 1.5.9: GP likelihood and predictive equations,
data-driven default priors (`darg`/`garg`), MAP estimation of the kernel parameter and nugget
(`mleGPsep`, `jmleGPsep`), the ALC and IECI criteria with exhaustive, ray and continuous search
(`alcGPsep`, `alcrayGPsep`, `alcoptGPsep`, `ieciGPsep`), and the local schemes (`laGPsep`,
`aGPsep`, `aGP.seq`) — under its own API, not laGP's. Beyond laGP: `trend="constant"`/`"linear"`
for better-calibrated intervals, gradients through fitting, and a prior on the conventional
lengthscale.

Deliberately left out: laGP's MSPE and Fisher-information criteria, block Latin hypercube
lengthscale bootstrapping (`blhs`), its refusal to optimize from a start on the lower bound
(`conv=102`), the raw-scale-vs-variance split between `laGPsep` and `aGPsep` (lagpax always returns
the variance), and the constrained-optimization (`optim.auglag`) and calibration (`fcalib`) layers.

The default `trend="zero"` is **simple kriging**, matching laGP: center `y` yourself — an
uncentered response inflates the profiled scale, which makes the variance wrong and not just the
mean. `trend="constant"` and `trend="linear"` profile out a mean too, and are sold as interval
calibration rather than accuracy: simple kriging omits the variance from not knowing the trend, so
its intervals are too narrow, especially at design boundaries.

## Parity

```bash
export R_HOME=$(R RHOME)
uv sync --extra tests
LAGPAX_STRICT=1 uv run --no-sync pytest tests -q
```

Everything with a laGP counterpart is compared against R. Tolerances are not uniform, on purpose:
means and log-likelihoods match to ~1e-12, while the variance is `1 + g - k' Ki k`, a cancellation
whose absolute error tracks the O(1) intermediate rather than the ~1e-3 result, so it is checked at
`rtol=1e-8`.

The optimizers (L-BFGS-B from [vlse](https://github.com/davidesartor/vlse) for the kernel, a
safeguarded Newton iteration for the nugget) do not reproduce R's iterate trajectory — they find the
same KKT point. Tests assert that, plus that our likelihood is never worse than laGP's. On noiseless
designs the nugget pins to its lower bound in a badly conditioned regime where the two disagree in
the 8th digit and ours is *higher*; those cases are tested as "no worse", and interior-optimum
parity is tested on a separately generated noisy design.

Beyond random designs, `tests/test_vlse.py` runs the comparison on eight benchmark surfaces from the
SFU library (Branin, Hartmann, Michalewicz, Ackley, Rosenbrock, Goldstein-Price, Shekel). The joint
MAP fit is never worse than laGP's, and is often several log-likelihood units better. Predictions
agree to 1e-6 or better at the median on every surface. On the multimodal ones (Michalewicz,
Ackley), a few locations' local fits settle in a different likelihood basin than laGP's, without a
systematic difference in rmse.

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
expensive criterion.

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

Reproduce with `uv run --no-sync -m bench.timing` and `uv run --no-sync -m bench.accuracy`. Note that `bench/common.py`
pins `OPENBLAS_NUM_THREADS=1` before jax or R load, and that `accuracy.py` hands both sides the same
default prior — without that it measures the RNG gap rather than the port.

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
