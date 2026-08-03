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
import numpy as np
import lagpax

X = np.random.uniform(size=(2000, 4))
Z = np.sum(np.sin(4 * X), axis=1)
XX = np.random.uniform(size=(500, 4))

out = lagpax.agp(X, Z, XX, start=6, end=50, criterion="alc")

out.mean   # (500,) predictive mean
out.var    # (500,) predictive variance
out.Xi     # (500, 50) local design indices, 0-based
```

`criterion` and `search` are a grid, not a list of methods. `criterion="nn"` takes nearest
neighbours, `criterion="alc"` grows the design greedily by Active Learning Cohn (Gramacy & Apley,
2015), and `criterion=lagpax.ieci(fmin=...)` scores candidates by expected improvement instead of
variance, for when the emulator serves optimization. `search="exhaustive"` scores every candidate;
`search="ray"` and `search="continuous"` cost a fixed number of criterion evaluations per greedy
step regardless of how many candidates there are, which is what makes the scheme usable on large
datasets. Any callable with the criterion signature `(gp, Xcand, Xref) -> scores` drops in.

Reuse the settings and the design-derived prior across calls by conditioning once:

```python
emu = lagpax.Local(start=6, end=50, criterion="alc").condition(X, Z)
out = emu.predict(XX, batch=64)
out = emu.refine(XX, passes=("nn", "alc"))   # warm-start each location from its own fit
one = emu.local(XX[:1])                      # a single reference set, as laGPsep
```

`condition` resolves the prior — the one host-side step — so everything after it is pure JAX:
`jax.jit`, `jax.vmap` over datasets and `jax.grad` all apply to `emu.predict` as a whole.

```python
prior = lagpax.Prior.from_design(X, Z)                 # resolved once, on the host
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

The public surface is seventeen names: `GP`, `Prior`, `Range`, `Prediction`, `Fit`, `Local`, `agp`,
`lagp`, `fit`, `alc`, `ieci`, `mspe`, `fisher`, `nn`, `bootstrap`, `distance`, `covar`. Everything
else stays importable from its own module (`lagpax.optim.lbfgsb`, `lagpax.search.ray`,
`lagpax.mle.jmle`).

Conventions follow laGP throughout — correlation `exp(-sum_k (x1_k - x2_k)^2 / d_k)`, nugget `g` on
the diagonal, scale profiled out as `phi = Z' Ki Z`, Student-t predictive. Two deliberate
differences: `Xi` indices are 0-based (R is 1-based), and everything is float64. One API difference:
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

The optimizers (`lbfgsb`, `bracketed_newton`) do not reproduce R's iterate trajectory — they find the
same KKT point. Tests assert that, plus that our likelihood is never worse than laGP's. On noiseless
designs the nugget MLE pins to its lower bound in a badly conditioned regime where the two disagree
in the 8th digit and ours is *higher*; those cases are tested as "no worse", and interior-optimum
parity is tested on a separately generated noisy design.

## Performance

Work in progress, and stated plainly: on current benchmarks the GPU implementation beats
single-core laGP by 2-14x but is **not yet consistently faster than 16-core laGP**. Profiling
across three GPUs spanning a 25x range in fp64 throughput produces only a 2.9x spread in wall clock,
which says the workload is bound by kernel-launch latency and the sequential ALC greedy loop, not by
arithmetic. Optimization is ongoing; see `bench/`.

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
