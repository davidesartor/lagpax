"""How the greedy design loop looks for its next point, given a criterion to score with.

Any criterion works with any search. A search is a pytree dataclass of static settings, called as
`search(criterion, gp, candidates, references, taken, step, bounding_box)` and returning the
index of one untaken candidate. `candidates` arrive sorted by distance to the references.
`RaySearch` and `ContinuousSearch` cost a fixed number of criterion evaluations per step however
many candidates there are.
"""

from __future__ import annotations

from typing import Callable, ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Scalar
from vlse.optim import minimise

from ._pytree import pytree_dataclass, static
from .criteria import CandidateCorrelation, Criterion
from .gaussian_process import GaussianProcess
from .kernel import squared_distances
from .optimize import R_OPTIM_RELATIVE_TOLERANCE

GOLDEN_RATIO_CONJUGATE = (5.0**0.5 - 1.0) / 2.0


class Search:
    reuses_candidate_correlation: ClassVar[bool] = False

    def __call__(
        self,
        criterion: Criterion,
        gp: GaussianProcess,
        candidates: Float[Array, "c d"],
        references: Float[Array, "r d"],
        taken: Bool[Array, "c"],
        step: Int[Array, ""],
        bounding_box: Float[Array, "2 d"],
        candidate_correlation: CandidateCorrelation = None,
    ) -> Int[Array, ""]:
        raise NotImplementedError


@pytree_dataclass
class ExhaustiveSearch(Search):
    """Score every candidate and take the best. laGP's default."""

    reuses_candidate_correlation: ClassVar[bool] = True

    def __call__(
        self,
        criterion,
        gp,
        candidates,
        references,
        taken,
        step,
        bounding_box,
        candidate_correlation=None,
    ):
        if candidate_correlation is not None and criterion.reuses_candidate_correlation:
            scores = criterion(gp, candidates, references, candidate_correlation)
        else:
            scores = criterion(gp, candidates, references)
        return best_untaken(scores, taken, criterion.lower_is_better)


@pytree_dataclass
class RaySearch(Search):
    """Golden-section search along rays leaving the reference point, snapped to a candidate.

    `rays=0` means one ray per input dimension. The rays start from untaken candidates near the
    reference on laGP's offset schedule, chosen deterministically where laGP samples them with R's
    RNG.
    """

    rays: int = static(default=0)
    golden_section_steps: int = static(default=30)

    def __call__(
        self,
        criterion,
        gp,
        candidates,
        references,
        taken,
        step,
        bounding_box,
        candidate_correlation=None,
    ):
        candidate_count, dimension = candidates.shape
        rays = min(self.rays or dimension, candidate_count)
        sign = -1.0 if criterion.lower_is_better else 1.0

        greedy_step = step + 1
        offset = jnp.mod(
            greedy_step,
            jnp.maximum(jnp.floor(jnp.sqrt(greedy_step)), 1.0).astype(step.dtype),
        )
        ray_starts = candidates[untaken_by_rank(taken, offset, rays)]
        ray_ends = _ray_ends(references[:1], ray_starts, bounding_box)

        def score_along_rays(fractions: Float[Array, "k"]) -> Float[Array, "k"]:
            points = ray_starts + fractions[:, None] * (ray_ends - ray_starts)
            return sign * criterion(gp, points, references)

        zeros = jnp.zeros(rays, candidates.dtype)
        fractions, values = _golden_section_maximize(
            score_along_rays, zeros, zeros + 1.0, self.golden_section_steps
        )
        best_ray = jnp.argmax(jnp.where(jnp.isfinite(values), values, -jnp.inf))
        best_point = (ray_starts + fractions[:, None] * (ray_ends - ray_starts))[
            best_ray
        ]
        return nearest_untaken(best_point[None], candidates, taken)


@pytree_dataclass
class ContinuousSearch(Search):
    """L-BFGS-B on the criterion over the bounding box from `starts` candidates, then snapped
    to the nearest untaken candidate. Gradients come from `jax.grad`, so any differentiable
    criterion works."""

    starts: int = static(default=1)
    max_iterations: int = static(default=100)

    def __call__(
        self,
        criterion,
        gp,
        candidates,
        references,
        taken,
        step,
        bounding_box,
        candidate_correlation=None,
    ):
        starts = min(self.starts, candidates.shape[0])
        sign = 1.0 if criterion.lower_is_better else -1.0

        def objective(point: Float[Array, "d"]) -> Scalar:
            return sign * criterion(gp, point[None], references)[0]

        def snapped_optimum(start: Float[Array, "d"]) -> Int[Array, ""]:
            state = minimise(
                objective,
                start,
                (bounding_box[0], bounding_box[1]),
                tol=0.0,
                ftol=R_OPTIM_RELATIVE_TOLERANCE,
                max_iterations=self.max_iterations,
            )
            return nearest_untaken(state.x[None], candidates, taken)

        start_points = candidates[untaken_by_rank(taken, step, starts)]
        chosen = jnp.stack([snapped_optimum(start_points[i]) for i in range(starts)])
        if starts == 1:
            return chosen[0]
        scores = criterion(gp, candidates[chosen], references)
        return chosen[best_untaken(scores, taken[chosen], criterion.lower_is_better)]


def best_untaken(
    scores: Float[Array, "c"], taken: Bool[Array, "c"], lower_is_better: bool
) -> Int[Array, ""]:
    unusable = jnp.inf if lower_is_better else -jnp.inf
    scores = jnp.where(taken | ~jnp.isfinite(scores), unusable, scores)
    return jnp.argmin(scores) if lower_is_better else jnp.argmax(scores)


def nearest_untaken(
    point: Float[Array, "1 d"], candidates: Float[Array, "c d"], taken: Bool[Array, "c"]
) -> Int[Array, ""]:
    distances = squared_distances(point, candidates)[0]
    return jnp.argmin(jnp.where(taken, jnp.inf, distances))


def untaken_by_rank(
    taken: Bool[Array, "c"], first_rank: Scalar, count: int
) -> Int[Array, "count"]:
    """Indices of the untaken candidates ranked `first_rank, first_rank + 1, ...` by distance.

    laGP removes chosen points from its candidate list, so "the k-th candidate" means the k-th
    still available. Candidates arrive distance-sorted, so the running count of untaken ones is
    each one's rank, and `searchsorted` inverts it in linear time where an argsort would not be.
    """
    available_so_far = jnp.cumsum(~taken)
    ranks = jnp.mod(
        first_rank + jnp.arange(count), jnp.maximum(available_so_far[-1], 1)
    )
    return jnp.searchsorted(available_so_far, ranks + 1, side="left")


def _golden_section_maximize(
    function: Callable[[Float[Array, "k"]], Float[Array, "k"]],
    lower: Float[Array, "k"],
    upper: Float[Array, "k"],
    steps: int,
) -> tuple[Float[Array, "k"], Float[Array, "k"]]:
    """Golden-section maximization on k intervals at once, one evaluation per step."""
    inner_low = upper - GOLDEN_RATIO_CONJUGATE * (upper - lower)
    inner_high = lower + GOLDEN_RATIO_CONJUGATE * (upper - lower)

    def narrow(_, state):
        low, high, probe_low, probe_high, value_low, value_high = state
        keep_left = value_low > value_high
        low, high = (
            jnp.where(keep_left, low, probe_low),
            jnp.where(keep_left, probe_high, high),
        )
        probe_low, probe_high = (
            jnp.where(
                keep_left, high - GOLDEN_RATIO_CONJUGATE * (high - low), probe_high
            ),
            jnp.where(
                keep_left, probe_low, low + GOLDEN_RATIO_CONJUGATE * (high - low)
            ),
        )
        value_new = function(jnp.where(keep_left, probe_low, probe_high))
        return (
            low,
            high,
            probe_low,
            probe_high,
            jnp.where(keep_left, value_new, value_high),
            jnp.where(keep_left, value_low, value_new),
        )

    initial = (
        lower,
        upper,
        inner_low,
        inner_high,
        function(inner_low),
        function(inner_high),
    )
    _, _, probe_low, probe_high, value_low, value_high = jax.lax.fori_loop(
        0, steps, narrow, initial
    )
    return (
        jnp.where(value_low > value_high, probe_low, probe_high),
        jnp.maximum(value_low, value_high),
    )


def _ray_ends(
    reference: Float[Array, "1 d"],
    ray_starts: Float[Array, "k d"],
    bounding_box: Float[Array, "2 d"],
) -> Float[Array, "k d"]:
    """Cast each ray ten times its start's distance from the reference, then pull it back to
    where it first leaves the bounding box (laGP's ray.end)."""
    direction = 10.0 * (ray_starts - reference)
    room = jnp.where(
        direction > 0, bounding_box[1] - ray_starts, bounding_box[0] - ray_starts
    )
    nonzero = jnp.where(direction != 0.0, direction, 1.0)
    fraction = jnp.min(jnp.where(direction != 0.0, room / nonzero, jnp.inf), axis=-1)
    return ray_starts + jnp.clip(fraction, 0.0, 1.0)[:, None] * direction
