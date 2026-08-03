"""JAX port of the laGP R package (separable / GPsep family).

Everything here is float64 and follows laGP's parameterization: the correlation is
`exp(-sum_k (x1_k - x2_k)^2 / d_k)` with nugget `g` on the diagonal, the process is
zero-mean unless a trend is asked for, and the scale is profiled out (`phi = Z' Ki Z`).

The public surface is deliberately small. Everything else stays importable from its own
module -- `lagpax.optim.lbfgsb`, `lagpax.covar.dcovar`, `lagpax.mle.jmle`, `lagpax.search.ray`.
"""

import jax

jax.config.update("jax_enable_x64", True)

from .covar import covar, distance
from .criteria import alc, ieci, nn
from .gp import GP, Prediction
from .local import Local, agp, lagp
from .mle import Fit, fit
from .prior import Prior, Range

__all__ = [
    "GP",
    "Fit",
    "Local",
    "Prediction",
    "Prior",
    "Range",
    "agp",
    "alc",
    "covar",
    "distance",
    "fit",
    "ieci",
    "lagp",
    "nn",
]
