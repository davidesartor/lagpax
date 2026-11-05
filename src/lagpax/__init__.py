"""Local approximate Gaussian processes in JAX.

    model = lagpax.LocalGP(neighbors=50, criterion="alc")
    emulator = model.fit(X, y, key=jax.random.key(0))
    prediction = emulator(Xnew)            # Prediction(mean, variance, dof, ...)

Each prediction location gets its own small GP on a design grown greedily from the training
set, with its own fitted hyperparameters -- which is what makes the model nonstationary. The
kernel is `exp(-sum (x - x')^2 / theta)` plus a nugget; the numerics are verified against the
laGP R package, which this began as a port of.

Computation runs in the dtype of the input; nothing here changes global JAX state. Results
matching laGP's need `jax.config.update("jax_enable_x64", True)` before the data is created.
"""

from .criteria import ALC, IECI, NN, Criterion, Custom, alc, nn
from .fit import FitInfo, fit
from .gp import GP, Prediction
from .kernel import correlation, sqdist
from .model import Emulator, LocalGP
from .prior import Param, Prior
from .search import Continuous, Exhaustive, Ray, Search

__all__ = [
    "ALC",
    "Continuous",
    "Criterion",
    "Custom",
    "Emulator",
    "Exhaustive",
    "FitInfo",
    "GP",
    "IECI",
    "LocalGP",
    "NN",
    "Param",
    "Prediction",
    "Prior",
    "Ray",
    "Search",
    "alc",
    "correlation",
    "fit",
    "nn",
    "sqdist",
]
