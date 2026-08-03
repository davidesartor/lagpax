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
- **Differentiable.** `jax.grad` works through prediction. laGP's C/OpenMP implementation cannot do
  this, and it is what makes local GPs usable inside a larger JAX pipeline — BO loops, gradient-based
  design, hybrid NN-surrogate models.
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

d = lagpax.darg(None, X)                      # design-derived lengthscale prior
out = lagpax.agpsep(X, Z, XX, start=6, end=50, method="alc", d=d)

out.mean   # (500,) predictive mean
out.var    # (500,) predictive variance
```

`method="nn"` uses nearest neighbours; `method="alc"` grows the local design greedily by Active
Learning Cohn, as in Gramacy & Apley (2015).

## Scope

The separable (`GPsep`) family: `newGPsep`, `mleGPsep`, `jmleGPsep`, `alcGPsep`, `laGPsep`,
`aGPsep`, plus `darg`/`garg`. Isotropic `GP` routines and the joint `aGP.seq`/`fish` extensions are
not ported.

Conventions follow laGP throughout — correlation `exp(-sum_k (x1_k - x2_k)^2 / d_k)`, nugget `g` on
the diagonal, zero mean, scale profiled out as `phi = Z' Ki Z`, Student-t predictive with `df = n`.
Two deliberate differences: `Xi` indices are 0-based (R is 1-based), and everything is float64.

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
