"""JAX port of the laGP R package (separable / GPsep family).

Computation runs in the dtype of the input. laGP is float64 throughout, so parity with it
needs `jax.config.update("jax_enable_x64", True)` *before* the data is created -- lagpax
does not flip that switch for you. It follows laGP's parameterization: the correlation is
`exp(-sum_k (x1_k - x2_k)^2 / d_k)` with nugget `g` on the diagonal, the process is
zero-mean unless a trend is asked for, and the scale is profiled out (`phi = Z' Ki Z`).

The public surface is deliberately small. Everything else stays importable from its own
module -- `lagpax.optim.implicit_argmin`, `lagpax.covar.dcovar`, `lagpax.mle.jmle`, `lagpax.search.ray`.
"""

from .blhs import bootstrap
from .covar import covar, distance
from .criteria import ALC, IECI, MSPE, NN, Criterion, Custom, Fisher, alc, fisher, nn
from .gp import GP, Prediction
from .local import Conditioned, Local, agp, lagp
from .mle import Fit, fit
from .prior import Prior, Range
from .search import Continuous, Exhaustive, Ray, Search

__all__ = [
    "ALC",
    "Conditioned",
    "Continuous",
    "Criterion",
    "Custom",
    "Exhaustive",
    "Fisher",
    "Fit",
    "GP",
    "IECI",
    "Local",
    "MSPE",
    "NN",
    "Prediction",
    "Prior",
    "Range",
    "Ray",
    "Search",
    "agp",
    "alc",
    "bootstrap",
    "covar",
    "distance",
    "fisher",
    "fit",
    "lagp",
    "nn",
]
