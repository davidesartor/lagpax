"""Local approximate GPs: one small GP per prediction point, on a greedily grown design.

`LocalGP` holds the settings. `LocalGP.fit(inputs, outputs)` resolves the prior -- the one
host-side step -- and returns an `Emulator`. Calling the emulator builds, for every row of the
new inputs, a GP on `neighbors` training points: the `initial_neighbors` nearest, then the rest
chosen one at a time by `criterion` and `search` from the `candidates` nearest. Each local GP
fits its own hyperparameters, which is what makes the model nonstationary.

Both classes are pytrees, so an emulator crosses `jax.jit`, is differentiable with respect to
its data, and its per-point primitive `predict_local` maps under `vmap` as readily as
`__call__` maps it under `lax.map`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from functools import partial
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, Key, Scalar

from ._pytree import pytree_dataclass, static
from .criteria import ActiveLearningCohn, Criterion, CustomCriterion, NearestNeighbor
from .fitting import fit_hyperparameters
from .gaussian_process import (
    GaussianProcess,
    Hyperparameters,
    Prediction,
    Trend,
    as_floating,
)
from .kernel import correlation, squared_distances
from .priors import Prior
from .search import ContinuousSearch, ExhaustiveSearch, RaySearch


@pytree_dataclass
class LocalGP:
    """Settings for local approximate GP prediction.

    neighbors: training points in each local design. initial_neighbors: how many of them are
    simply the nearest. candidates: how many nearest points the greedy search may pick from;
    the default, laGP's, is 1000 + neighbors, ten times that for ray and continuous search.
    trend: "zero" (laGP's simple kriging), "constant" or "linear". max_iterations: the cap for
    each hyperparameter solve.

    `criterion` accepts "nn", "alc", a `Criterion`, or a scoring function; `search` accepts
    "exhaustive", "ray", "continuous", or a `Search`. Both are normalized on construction.
    """

    neighbors: int = static(default=50)
    initial_neighbors: int = static(default=6)
    candidates: Optional[int] = static(default=None)
    criterion: Any = "alc"
    search: Any = "exhaustive"
    trend: Trend = static(default="zero")
    max_iterations: int = static(default=100)

    def __post_init__(self) -> None:
        criterion, search = self.criterion, self.search
        if criterion == "ieci":
            raise ValueError(
                "ieci needs the best observed value: "
                "pass IntegratedExpectedConditionalImprovement(best_observed=...)"
            )
        if isinstance(criterion, str):
            criteria = {"nn": NearestNeighbor, "alc": ActiveLearningCohn}
            if criterion not in criteria:
                raise ValueError(f"unknown criterion {criterion!r}")
            criterion = criteria[criterion]()
        elif not isinstance(criterion, Criterion):
            criterion = CustomCriterion(criterion)
        if isinstance(search, str):
            searches = {
                "exhaustive": ExhaustiveSearch,
                "ray": RaySearch,
                "continuous": ContinuousSearch,
            }
            if search not in searches:
                raise ValueError(f"unknown search {search!r}")
            search = searches[search]()
        object.__setattr__(self, "criterion", criterion)
        object.__setattr__(self, "search", search)

    def replace(self, **changes: Any) -> LocalGP:
        return dataclasses.replace(self, **changes)

    def fit(
        self,
        inputs: Float[Array, "n d"],
        outputs: Float[Array, "n"],
        prior: Optional[Prior] = None,
        key: Optional[Key[Array, ""]] = None,
    ) -> Emulator:
        """Bind training data and resolve the prior.

        Without a prior, `Prior.from_data(inputs, outputs, key=key)` is used, which needs `key`
        for more than 1000 rows. Hyperparameters are fitted per point when the emulator is
        called, not here.
        """
        inputs = as_floating(inputs)
        outputs = jnp.asarray(outputs, inputs.dtype)
        count = inputs.shape[0]
        if outputs.shape != (count,):
            raise ValueError("outputs must be a vector with one entry per row of inputs")
        if not 6 <= self.initial_neighbors < self.neighbors:
            raise ValueError("need 6 <= initial_neighbors < neighbors")
        if count <= self.neighbors:
            raise ValueError("need more training points than neighbors")

        scores_every_candidate = not isinstance(self.search, (RaySearch, ContinuousSearch))
        default_candidates = (1000 + self.neighbors) * (1 if scores_every_candidate else 10)
        candidates = (
            min(default_candidates, count) if self.candidates is None else int(self.candidates)
        )
        if not self.neighbors < candidates <= count:
            raise ValueError("need neighbors < candidates <= number of training points")

        if prior is None:
            prior = Prior.from_data(np.asarray(inputs), np.asarray(outputs), key=key)
        return Emulator(self.replace(candidates=candidates), inputs, outputs, prior)


class Emulator(NamedTuple):
    """A `LocalGP` bound to training data and a resolved prior. Call it to predict.

    A pytree: `emulator._replace(outputs=new_outputs)` is the same model on new responses.
    """

    model: LocalGP
    inputs: Float[Array, "n d"]
    outputs: Float[Array, "n"]
    prior: Prior

    def predict_local(
        self,
        references: Float[Array, "r d"],
        initial_squared_lengthscales: Optional[Float[Array, "d"]] = None,
        full_covariance: bool = False,
    ) -> Prediction:
        """One local GP for a whole reference set, and its prediction there.

        `initial_squared_lengthscales` overrides where design selection and the fit start
        (clipped into the prior's bounds).
        """
        references = self._as_points(references)
        start = self._starting_squared_lengthscales(
            initial_squared_lengthscales, (references.shape[-1],)
        )
        return _predict_local(self, references, start, full_covariance=full_covariance)

    def __call__(
        self,
        new_inputs: Float[Array, "m d"],
        batch_size: Optional[int] = 256,
        initial_squared_lengthscales: Optional[Float[Array, "m d"]] = None,
    ) -> Prediction:
        """Predict at every row of `new_inputs`, each from its own local GP.

        `batch_size` bounds how many points are in flight at once (`lax.map`'s); it trades
        memory for throughput and does not change the results.
        """
        new_inputs = self._as_points(new_inputs)
        starts = self._starting_squared_lengthscales(initial_squared_lengthscales, new_inputs.shape)
        predictions = jax.lax.map(
            lambda point_and_start: _predict_local(
                self, point_and_start[0][None], point_and_start[1], full_covariance=False
            ),
            (new_inputs, starts),
            batch_size=batch_size,
        )
        return predictions._replace(
            mean=predictions.mean[:, 0], variance=predictions.variance[:, 0]
        )

    def refine(
        self,
        new_inputs: Float[Array, "m d"],
        passes: Sequence[str | Criterion] = ("nn", "alc"),
        batch_size: Optional[int] = 256,
    ) -> Prediction:
        """Predict in passes, each starting every point from the previous pass's fit.

        Designs are chosen at the *starting* squared lengthscales, so a pass that inherits each
        point's fit picks neighbourhoods that respect which inputs matter. The first pass only
        supplies those fits, so a cheap "nn" serves as well as "alc" there.
        """
        if not passes:
            raise ValueError("passes must name at least one pass")
        new_inputs = self._as_points(new_inputs)
        starts = self._starting_squared_lengthscales(None, new_inputs.shape)
        for criterion in passes:
            emulator = self._replace(model=self.model.replace(criterion=criterion))
            prediction = emulator(new_inputs, batch_size, starts)
            fitted = prediction.fit.squared_lengthscales
            starts = jnp.where(jnp.isfinite(fitted), fitted, starts)
        return prediction

    def _as_points(self, points: Float[Array, "m d"]) -> Float[Array, "m d"]:
        points = jnp.atleast_2d(jnp.asarray(points, self.inputs.dtype))
        if points.shape[-1] != self.inputs.shape[-1]:
            raise ValueError("points and training inputs have different numbers of columns")
        return points

    def _starting_squared_lengthscales(
        self, requested: Optional[Float[Array, "..."]], shape: tuple[int, ...]
    ) -> Float[Array, "..."]:
        dtype, prior = self.inputs.dtype, self.prior
        if requested is None:
            initial = jnp.asarray(prior.kernel.initial, dtype)
            return jnp.broadcast_to(prior.to_squared_lengthscales(initial), shape)
        in_prior_space = prior.from_squared_lengthscales(
            jnp.broadcast_to(jnp.asarray(requested, dtype), shape)
        )
        clipped = jnp.clip(
            in_prior_space,
            jnp.asarray(prior.kernel.lower, dtype),
            jnp.asarray(prior.kernel.upper, dtype),
        )
        return prior.to_squared_lengthscales(clipped)


@partial(jax.jit, static_argnames=("full_covariance",))
def _predict_local(
    emulator: Emulator,
    references: Float[Array, "r d"],
    starting_squared_lengthscales: Float[Array, "d"],
    *,
    full_covariance: bool,
) -> Prediction:
    model = emulator.model
    starting = Hyperparameters(
        starting_squared_lengthscales,
        jnp.asarray(emulator.prior.nugget.initial, emulator.inputs.dtype),
    )
    distance_to_references = jnp.min(squared_distances(references, emulator.inputs), axis=0)
    pool = jax.lax.top_k(-distance_to_references, model.candidates)[1]
    pool_inputs, pool_outputs = emulator.inputs[pool], emulator.outputs[pool]

    initial, neighbors = model.initial_neighbors, model.neighbors
    criterion, search = model.criterion, model.search
    if isinstance(criterion, NearestNeighbor):
        design = jnp.arange(neighbors)
    else:
        bounding_box = jnp.stack(
            [
                jnp.min(pool_inputs[initial:], axis=0),
                jnp.max(pool_inputs[initial:], axis=0),
            ]
        )
        candidate_correlation = None
        if criterion.reuses_candidate_correlation and search.reuses_candidate_correlation:
            candidate_correlation = (
                jnp.zeros((model.candidates, neighbors), pool_inputs.dtype)
                .at[:, :initial]
                .set(
                    correlation(
                        pool_inputs,
                        pool_inputs[:initial],
                        starting.squared_lengthscales,
                    )
                )
            )
        state = DesignState(
            chosen=jnp.concatenate([jnp.arange(initial), jnp.zeros(neighbors - initial, int)]),
            active=jnp.arange(neighbors) < initial,
            taken=jnp.arange(model.candidates) < initial,
            candidate_correlation=candidate_correlation,
        )

        def add_one_point(capacity: int):
            def body(step: Scalar, state: DesignState) -> DesignState:
                gp = GaussianProcess.from_data(
                    pool_inputs[state.chosen[:capacity]],
                    pool_outputs[state.chosen[:capacity]],
                    starting,
                    state.active[:capacity],
                    model.trend,
                )
                cached = (
                    None
                    if state.candidate_correlation is None
                    else state.candidate_correlation[:, :capacity]
                )
                pick = search(
                    criterion,
                    gp,
                    pool_inputs,
                    references,
                    state.taken,
                    step - initial,
                    bounding_box,
                    cached,
                )
                updated_correlation = state.candidate_correlation
                if updated_correlation is not None:
                    new_column = correlation(
                        pool_inputs,
                        pool_inputs[pick][None],
                        starting.squared_lengthscales,
                    )[:, 0]
                    updated_correlation = updated_correlation.at[:, step].set(new_column)
                return DesignState(
                    state.chosen.at[step].set(pick),
                    state.active.at[step].set(True),
                    state.taken.at[pick].set(True),
                    updated_correlation,
                )

            return body

        for first_step, end_step, capacity in design_buckets(initial, neighbors):
            state = jax.lax.fori_loop(first_step, end_step, add_one_point(capacity), state)
        design = state.chosen

    gp = GaussianProcess.from_data(
        pool_inputs[design], pool_outputs[design], starting, trend=model.trend
    )
    gp, fit = fit_hyperparameters(gp, emulator.prior, model.max_iterations)
    prediction = gp.predict(references, full_covariance=full_covariance)
    return prediction._replace(neighbors=pool[design], fit=fit)


class DesignState(NamedTuple):
    chosen: Int[Array, "neighbors"]
    active: Bool[Array, "neighbors"]
    taken: Bool[Array, "candidates"]
    candidate_correlation: Optional[Float[Array, "candidates neighbors"]]


def design_buckets(initial: int, neighbors: int) -> list[tuple[int, int, int]]:
    """Greedy steps grouped by the padded GP size they build: (first_step, end_step, capacity).

    Step t has t active rows; padding it to the next power of two rather than to `neighbors`
    halves the work and still traces only logarithmically many loop bodies.
    """
    buckets, step = [], initial
    while step < neighbors:
        capacity = min(1 << max(step - 1, 0).bit_length(), neighbors)
        end_step = min(capacity + 1, neighbors)
        buckets.append((step, end_step, capacity))
        step = end_step
    return buckets
