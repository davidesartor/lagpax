# lagpax

Local approximate Gaussian processes in JAX: jittable, vmappable, differentiable, and verified
value-for-value against the [laGP](https://cran.r-project.org/package=laGP) R package.

Every predictive location gets its own small GP, built on a local design picked from the full
dataset and fitted with its own lengthscales. The result is a nonstationary emulator that runs as one
batched XLA call, with `jax.grad` working through it, local hyperparameter fits included.

## Install

```bash
uv add lagpax            # or "lagpax[gpu]" for CUDA 12
```

## Quickstart

```python
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)   # float64, as laGP
import lagpax

inputs = np.random.uniform(size=(2000, 4))
outputs = np.sum(np.sin(4 * inputs), axis=1)
new_inputs = np.random.uniform(size=(500, 4))

model = lagpax.LocalGP(neighbors=50, criterion="alc", search="exhaustive")
emulator = model.fit(inputs, outputs, key=jax.random.key(0))
prediction = emulator(new_inputs)

prediction.mean, prediction.variance        # (500,), Student-t predictive
prediction.neighbors                        # (500, 50) training rows each local GP used
prediction.fit.squared_lengthscales         # (500, 4)
```

- **criterion** picks each next design point: `"nn"`, `"alc"` (default),
  `IntegratedExpectedConditionalImprovement`, or any `(gp, candidates, references) -> scores`.
- **search** finds it: `"exhaustive"`, or `"ray"` / `"continuous"`, whose cost doesn't grow with
  the dataset.
- **prior**: `Prior.from_data(inputs, outputs, key=key)` is laGP's default; pass
  `kernel_space="lengthscale"` to put it on `l` instead of `s = 2 l^2`.
- **refine**: `emulator.refine(new_inputs, passes=("nn", "alc"))` re-picks designs with fitted
  lengthscales, cutting rmse ~1.7-2x on anisotropic problems.

`Emulator` is a pytree, so it crosses `jax.jit` and differentiates as a whole:

```python
prior = lagpax.Prior.from_data(inputs, outputs, key=jax.random.key(0))
jax.grad(lambda y: model.fit(inputs, y, prior=prior)(new_inputs).mean.sum())(outputs)
```

Importing lagpax changes no global JAX state, and anything random takes a key.

## Scope

lagpax covers the separable (`GPsep`) core of laGP 1.5.9 under its own API: likelihood and
prediction, default priors, MAP fits of lengthscales and nugget, ALC/IECI with all three searches,
and `laGPsep` / `aGPsep` / `aGP.seq`. It adds `trend="constant"`/`"linear"` (better-calibrated
intervals; the default `"zero"` is simple kriging, so center `y`), gradients, and a lengthscale-space
prior. It leaves out MSPE/Fisher criteria, `blhs`, `optim.auglag` and `fcalib`.

## Parity

```bash
export R_HOME=$(R RHOME) && uv sync --extra tests
LAGPAX_STRICT=1 uv run --no-sync pytest tests -q
```

Every function with a laGP counterpart is checked against R. Means and log-likelihoods agree to
~1e-12, variances to `rtol=1e-8`. The optimizers reach the same optimum by a different path, and
their likelihood is never lower than laGP's, including on eight SFU benchmark surfaces.

## Performance

Both implementations scale linearly in the number of locations. A100 against laGP on 16 Xeon 8352Y
cores of the same node, with `n=8000`, 4 inputs and 50 neighbors:

| criterion | lagpax | laGP | break-even | asymptotic speedup |
| --- | --- | --- | --- | --- |
| `alc` | 3.15 s + 0.850 ms/loc | 6.218 ms/loc | ~590 locations | 7.3x |
| `nn` | 2.45 s + 0.473 ms/loc | 1.608 ms/loc | ~2,200 locations | 3.4x |

Below the break-even, laGP is faster. These numbers are from a single seed at `batch_size=64`; the
current default of 256 is ~1.4x faster per location. Reproduce them with
`uv run --no-sync -m bench.timing`.

## Related work

- [laGP](https://cran.r-project.org/package=laGP): the original, in R and C/OpenMP.
- [Scaled Vecchia](https://github.com/katzfuss-group/scaledVecchia): a valid joint model, the better
  choice when you need a global likelihood or joint draws.
- [GPJax](https://github.com/JaxGaussianProcesses/GPJax), [tinygp](https://tinygp.readthedocs.io/):
  exact and sparse GPs in JAX.

## License and citation

lagpax is **LGPL-2.0-or-later**, the license inherited from laGP, since lagpax is a translation of
its source. See `LICENSE` and `NOTICE`. Please cite:

> Gramacy, R. B. and Apley, D. W. (2015). Local Gaussian Process Approximation for Large Computer
> Experiments. *Journal of Computational and Graphical Statistics*, 24(2), 561-578.

lagpax is not affiliated with or endorsed by the laGP authors.
