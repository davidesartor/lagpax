"""Design criteria: how much each candidate point would improve a local GP's design.

A criterion is a pytree dataclass called as `criterion(gp, candidates, references)` and returning
one score per candidate. `lower_is_better` gives the score's direction. A criterion with
`reuses_candidate_correlation` also accepts `candidate_correlation =
correlation(candidates, gp.inputs, squared_lengthscales)`, which the greedy design loop carries
forward one column per step instead of rebuilding.

Settings that are numbers (`IntegratedExpectedConditionalImprovement.best_observed`) are pytree
leaves, so changing them does not recompile; settings that choose code are static.
"""

from __future__ import annotations

from typing import Callable, ClassVar, Optional

import jax.numpy as jnp
from jax.scipy.special import betainc, gammaln
from jaxtyping import Array, Float

from ._pytree import pytree_dataclass, static
from .gaussian_process import GaussianProcess, predictive_mean_and_scale
from .kernel import correlation, squared_distances
from .priors import ROOT_MACHINE_EPSILON

CandidateCorrelation = Optional[Float[Array, "c n"]]


class Criterion:
    lower_is_better: ClassVar[bool] = False
    reuses_candidate_correlation: ClassVar[bool] = False

    def __call__(
        self,
        gp: GaussianProcess,
        candidates: Float[Array, "c d"],
        references: Float[Array, "r d"],
        candidate_correlation: CandidateCorrelation = None,
    ) -> Float[Array, "c"]:
        raise NotImplementedError


@pytree_dataclass
class NearestNeighbor(Criterion):
    """Prefers candidates closest to the nearest reference point."""

    def __call__(self, gp, candidates, references, candidate_correlation=None):
        return -jnp.min(squared_distances(candidates, references), axis=-1)


@pytree_dataclass
class ActiveLearningCohn(Criterion):
    """Average reduction in predictive variance over the references from adding each candidate."""

    reuses_candidate_correlation: ClassVar[bool] = True

    def __call__(self, gp, candidates, references, candidate_correlation=None):
        reduction, conditional_variance = variance_reduction(
            gp, candidates, references, candidate_correlation
        )
        dof = gp.degrees_of_freedom
        expected = jnp.mean(
            gp.residual_sum_of_squares * reduction / dof * (dof / (dof - 2.0)), axis=-1
        )
        return jnp.where(
            conditional_variance <= ROOT_MACHINE_EPSILON, -jnp.inf, expected
        )


@pytree_dataclass
class IntegratedExpectedConditionalImprovement(Criterion):
    """Average expected improvement below `best_observed` over the references, once each candidate
    has sharpened the predictive variance there. Lower is better.

    `weights` scale the average rather than renormalize it, as laGP's ieciGPsep does.
    """

    best_observed: Float[Array, ""]
    weights: Optional[Float[Array, "r"]] = None
    include_nugget: bool = static(default=True)
    lower_is_better: ClassVar[bool] = True
    reuses_candidate_correlation: ClassVar[bool] = True

    def __call__(self, gp, candidates, references, candidate_correlation=None):
        reduction, conditional_variance = variance_reduction(
            gp, candidates, references, candidate_correlation
        )
        reference_mean, reference_scale = predictive_mean_and_scale(
            gp, references, self.include_nugget
        )
        dof = gp.degrees_of_freedom
        sharpened_scale = reference_scale - gp.residual_sum_of_squares * reduction / dof
        scale = jnp.sqrt(jnp.maximum(sharpened_scale, 0.0))
        scale = jnp.where(scale > 0, scale, 1.0)
        standardized = (self.best_observed - reference_mean) / scale
        student_t_log_density = (
            gammaln(0.5 * (dof + 1.0))
            - gammaln(0.5 * dof)
            - 0.5 * jnp.log(jnp.pi * dof)
            - 0.5 * (dof + 1.0) * jnp.log1p(jnp.square(standardized) / dof)
        )
        tail = 0.5 * betainc(0.5 * dof, 0.5, dof / (dof + jnp.square(standardized)))
        student_t_cdf = jnp.where(standardized <= 0, tail, 1.0 - tail)
        improvement = scale * (
            standardized * student_t_cdf
            + (dof + jnp.square(standardized))
            / (dof - 1.0)
            * jnp.exp(student_t_log_density)
        )
        if self.weights is not None:
            improvement = self.weights * improvement
        average = jnp.mean(improvement, axis=-1)
        return jnp.where(conditional_variance <= ROOT_MACHINE_EPSILON, jnp.inf, average)


@pytree_dataclass
class CustomCriterion(Criterion):
    """Any `(gp, candidates, references) -> scores` function, as a criterion.

    `score` is static, so it is part of the compiled program, keyed by identity.
    """

    score: Callable = static()
    lower_is_better: bool = static(default=False)
    reuses_candidate_correlation: bool = static(default=False)

    def __call__(self, gp, candidates, references, candidate_correlation=None):
        if self.reuses_candidate_correlation and candidate_correlation is not None:
            return self.score(gp, candidates, references, candidate_correlation)
        return self.score(gp, candidates, references)


def variance_reduction(
    gp: GaussianProcess,
    candidates: Float[Array, "c d"],
    references: Float[Array, "r d"],
    candidate_correlation: CandidateCorrelation = None,
) -> tuple[Float[Array, "c r"], Float[Array, "c"]]:
    """For each candidate: the drop in unscaled predictive variance at each reference if the
    candidate joined the design, and the candidate's own conditional variance."""
    squared_lengthscales = gp.hyperparameters.squared_lengthscales
    reference_correlation = jnp.where(
        gp.active, correlation(references, gp.inputs, squared_lengthscales), 0.0
    )
    if candidate_correlation is None:
        candidate_correlation = correlation(candidates, gp.inputs, squared_lengthscales)
    candidate_correlation = jnp.where(gp.active, candidate_correlation, 0.0)
    candidate_to_reference = correlation(candidates, references, squared_lengthscales)

    weights = candidate_correlation @ gp.inverse_correlation
    conditional_variance = (
        1.0
        + gp.hyperparameters.nugget
        - jnp.sum(candidate_correlation * weights, axis=-1)
    )
    weights = -weights / conditional_variance[:, None]

    projected = weights @ reference_correlation.T
    reduction = (
        jnp.square(projected) * conditional_variance[:, None]
        + 2.0 * projected * candidate_to_reference
        + jnp.square(candidate_to_reference) / conditional_variance[:, None]
    )
    return reduction, conditional_variance
