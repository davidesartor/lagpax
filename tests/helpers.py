"""Small constructors shared by the test modules."""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from lagpax import GammaPrior, GaussianProcess, Hyperparameters, Param, Prior
from lagpax.gaussian_process import Trend, log_likelihood_at


def make_gp(
    inputs,
    outputs,
    squared_lengthscales,
    nugget,
    active: Optional[np.ndarray] = None,
    trend: Trend = "zero",
) -> GaussianProcess:
    return GaussianProcess.from_data(
        jnp.asarray(inputs),
        jnp.asarray(outputs),
        Hyperparameters(jnp.asarray(squared_lengthscales), nugget),
        None if active is None else jnp.asarray(active),
        trend,
    )


def gamma(shape_rate) -> Optional[GammaPrior]:
    """laGP's (a, b) pair as a GammaPrior; (0, 0) means no prior, as in laGP."""
    shape, rate = (float(v) for v in shape_rate)
    return None if shape <= 0 or rate <= 0 else GammaPrior(shape, rate)


def lagp_prior(
    squared_lengthscale_ab=(0.0, 0.0),
    nugget_ab=(0.0, 0.0),
    squared_lengthscale_range=(1e-3, 10.0),
    nugget_range=(float(np.sqrt(np.finfo(float).eps)), 1.0),
    learn=(True, True),
) -> Prior:
    """A squared-lengthscale Prior from laGP's (range, ab) arguments."""
    return Prior(
        squared_lengthscale=Param(
            lower=squared_lengthscale_range[0],
            upper=squared_lengthscale_range[1],
            prior=gamma(squared_lengthscale_ab),
            learn=learn[0],
        ),
        nugget=Param(
            lower=nugget_range[0],
            upper=nugget_range[1],
            prior=gamma(nugget_ab),
            learn=learn[1],
        ),
    )


def log_likelihood_gradient(gp: GaussianProcess, prior: Optional[Prior] = None):
    """d log-likelihood / d (squared lengthscales, nugget) at the process's hyperparameters."""
    return jax.grad(lambda h: log_likelihood_at(gp, h, prior))(gp.hyperparameters)
