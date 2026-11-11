"""Local approximate Gaussian processes in JAX.

    model = lagpax.LocalGP(neighbors=50, criterion="alc")
    emulator = model.fit(inputs, outputs, key=jax.random.key(0))
    prediction = emulator(new_inputs)      # Prediction(mean, variance, degrees_of_freedom, ...)

Every prediction point gets its own small GP on a design grown greedily from the training set,
with its own fitted hyperparameters, which makes the model nonstationary. The numerics are
verified against the laGP R package, which this began as a port of.

Computation follows the input dtype and nothing here changes global JAX state. Results that
match laGP's need `jax.config.update("jax_enable_x64", True)` before the data is created.
"""

from .criteria import (
    ActiveLearningCohn,
    Criterion,
    CustomCriterion,
    IntegratedExpectedConditionalImprovement,
    NearestNeighbor,
)
from .fitting import FitResult, fit_hyperparameters
from .gaussian_process import GaussianProcess, Hyperparameters, Prediction
from .kernel import correlation, squared_distances
from .local import Emulator, LocalGP
from .priors import GammaPrior, Param, Prior
from .search import ContinuousSearch, ExhaustiveSearch, RaySearch, Search

__all__ = [
    "ActiveLearningCohn",
    "ContinuousSearch",
    "Criterion",
    "CustomCriterion",
    "Emulator",
    "ExhaustiveSearch",
    "FitResult",
    "GammaPrior",
    "GaussianProcess",
    "Hyperparameters",
    "IntegratedExpectedConditionalImprovement",
    "LocalGP",
    "NearestNeighbor",
    "Param",
    "Prediction",
    "Prior",
    "RaySearch",
    "Search",
    "correlation",
    "fit_hyperparameters",
    "squared_distances",
]
