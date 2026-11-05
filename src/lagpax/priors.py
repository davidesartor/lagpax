"""Hyperparameter priors: starting values, bounds, optional Gamma densities, and data defaults.

The kernel's parameter can be described in one of two spaces. `squared_lengthscale` is the
kernel's own parameter s (laGP's `d`), and a prior on it reproduces laGP's fits exactly.
`lengthscale` is the conventional l = sqrt(s / 2). Both spaces share one likelihood, but a Gamma
density on l is not a Gamma density on s, so their MAP estimates differ.

`Prior.from_data` sets every default from the data the way laGP's darg/garg do. It runs once, on
the host, in numpy: R's `quantile` type 7 is numpy's default.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special
from jaxtyping import Array, Float, Key, Scalar

from ._pytree import pytree_dataclass, static

ROOT_MACHINE_EPSILON = float(np.sqrt(np.finfo(float).eps))

KernelSpace = Literal["squared_lengthscale", "lengthscale"]


@pytree_dataclass
class GammaPrior:
    shape: float
    rate: float

    @classmethod
    def with_mass_below(
        cls, upper: float, shape: float = 1.5, mass: float = 0.95
    ) -> GammaPrior:
        """The Gamma(shape, rate) putting `mass` of its probability below `upper`."""
        return cls(shape, float(scipy.special.gammaincinv(shape, mass)) / upper)

    def log_density(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return (
            (self.shape - 1.0) * jnp.log(x)
            - self.rate * x
            + self.shape * jnp.log(self.rate)
            - jax.scipy.special.gammaln(self.shape)
        )


@pytree_dataclass
class Param:
    """One hyperparameter: where its fit starts, its bounds, its prior, and whether it is fitted.

    Fields left as `None` mean "use the default" in `Prior.from_data` and `Prior.replace`, so
    `Param(initial=1e-6, learn=False)` is a usable partial specification.
    """

    initial: Union[float, np.ndarray, None] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    prior: Optional[GammaPrior] = None
    learn: Optional[bool] = static(default=None)

    def overridden_by(self, other: Param) -> Param:
        changes = {
            field.name: getattr(other, field.name)
            for field in dataclasses.fields(Param)
            if getattr(other, field.name) is not None
        }
        return dataclasses.replace(self, **changes)

    def validate(self) -> None:
        if self.upper <= 0:
            raise ValueError("upper must be positive")
        if self.lower < 0 or self.lower > self.upper:
            raise ValueError("lower must lie in [0, upper]")
        initial = np.asarray(self.initial)
        if np.any(initial < self.lower) or np.any(initial > self.upper):
            raise ValueError("initial must lie in [lower, upper]")


ParamSpec = Union[float, np.ndarray, Param, None]


def as_param(spec: ParamSpec) -> Param:
    return spec if isinstance(spec, Param) else Param(initial=spec)


@pytree_dataclass
class Prior:
    """The nugget's `Param` and the kernel's, given as exactly one of the two kernel spaces."""

    nugget: Param
    squared_lengthscale: Optional[Param] = None
    lengthscale: Optional[Param] = None

    def __post_init__(self) -> None:
        if (self.squared_lengthscale is None) == (self.lengthscale is None):
            raise ValueError(
                "give exactly one of squared_lengthscale= and lengthscale="
            )

    @property
    def kernel_space(self) -> KernelSpace:
        return (
            "squared_lengthscale"
            if self.squared_lengthscale is not None
            else "lengthscale"
        )

    @property
    def kernel(self) -> Param:
        return (
            self.squared_lengthscale
            if self.squared_lengthscale is not None
            else self.lengthscale
        )

    def to_squared_lengthscales(
        self, kernel_values: Float[Array, "d"]
    ) -> Float[Array, "d"]:
        if self.kernel_space == "squared_lengthscale":
            return kernel_values
        return 2.0 * jnp.square(kernel_values)

    def from_squared_lengthscales(
        self, squared: Float[Array, "d"]
    ) -> Float[Array, "d"]:
        if self.kernel_space == "squared_lengthscale":
            return squared
        return jnp.sqrt(0.5 * squared)

    def nugget_log_density(self, nugget: Scalar) -> Scalar:
        if self.nugget.prior is None:
            return jnp.zeros((), jnp.result_type(nugget))
        return jnp.where(nugget > 0, self.nugget.prior.log_density(nugget), 0.0)

    def replace(
        self,
        squared_lengthscale: ParamSpec = None,
        lengthscale: ParamSpec = None,
        nugget: ParamSpec = None,
    ) -> Prior:
        """Override the given fields of this prior's Params, keeping everything left unset."""
        updated = self
        if nugget is not None:
            updated = dataclasses.replace(
                updated, nugget=updated.nugget.overridden_by(as_param(nugget))
            )
        for name, spec in (
            ("squared_lengthscale", squared_lengthscale),
            ("lengthscale", lengthscale),
        ):
            if spec is None:
                continue
            current = getattr(updated, name)
            if current is None:
                raise ValueError(
                    f"this prior has no {name}; it is in the other kernel space"
                )
            updated = dataclasses.replace(
                updated, **{name: current.overridden_by(as_param(spec))}
            )
        return updated

    @classmethod
    def from_data(
        cls,
        inputs: np.ndarray,
        outputs: np.ndarray,
        kernel: ParamSpec = None,
        nugget: ParamSpec = 1e-4,
        kernel_space: KernelSpace = "squared_lengthscale",
        sample_size: int = 1000,
        key: Optional[Key[Array, ""]] = None,
    ) -> Prior:
        """Defaults from the data, following laGP's darg and garg.

        Squared lengthscale: start at the 10% quantile of the pairwise squared distances, bounded
        by half the smallest and by the largest, with a Gamma(1.5, rate) prior holding 95% of its
        mass below the largest. Nugget: fixed at 1e-4 unless `nugget=Param(learn=True)`, then
        bounded by the largest squared deviation of the outputs. For `kernel_space="lengthscale"`
        the same start and bounds are mapped to l = sqrt(s / 2), the Gamma prior is placed on l,
        and a `kernel` spec is read in lengthscale units.

        A design of more than `sample_size` rows is subsampled to measure distances, which needs
        a `key`. Without one that is an error, never a hidden random draw.
        """
        nugget_param = default_nugget(nugget, outputs)
        if kernel_space == "squared_lengthscale":
            squared = default_squared_lengthscale(kernel, inputs, sample_size, key)
            return cls(nugget=nugget_param, squared_lengthscale=squared)
        if kernel_space != "lengthscale":
            raise ValueError(f"unknown kernel_space {kernel_space!r}")

        spec = as_param(kernel)
        squared = default_squared_lengthscale(
            Param(
                initial=_squared(spec.initial),
                lower=_squared(spec.lower),
                upper=_squared(spec.upper),
                learn=spec.learn,
            ),
            inputs,
            sample_size,
            key,
            include_prior=False,
        )
        upper = float(_rooted(squared.upper))
        prior = spec.prior
        if prior is None and squared.learn:
            prior = GammaPrior.with_mass_below(upper)
        lengthscale = Param(
            initial=_rooted(squared.initial),
            lower=float(_rooted(squared.lower)),
            upper=upper,
            prior=prior,
            learn=squared.learn,
        )
        lengthscale.validate()
        return cls(nugget=nugget_param, lengthscale=lengthscale)


def _squared(lengthscale: Union[float, np.ndarray, None]) -> Union[np.ndarray, None]:
    return (
        None if lengthscale is None else 2.0 * np.square(np.asarray(lengthscale, float))
    )


def _rooted(squared: Union[float, np.ndarray]) -> np.ndarray:
    return np.sqrt(0.5 * np.asarray(squared, float))


def pairwise_distance_summary(
    inputs: np.ndarray,
    quantile: float = 0.1,
    sample_size: int = 1000,
    key: Optional[Key[Array, ""]] = None,
) -> tuple[float, float, float]:
    """(quantile, min, max) of the positive pairwise squared distances; laGP's getDs."""
    inputs = np.atleast_2d(np.asarray(inputs, float))
    if inputs.shape[0] > sample_size:
        if key is None:
            raise ValueError(
                f"the default prior subsamples {inputs.shape[0]} rows down to {sample_size}, "
                "which is random: pass key=jax.random.key(...)"
            )
        rows = jax.random.choice(key, inputs.shape[0], (sample_size,), replace=False)
        inputs = inputs[np.asarray(rows)]
    distances = np.sum(np.square(inputs[:, None, :] - inputs[None, :, :]), axis=-1)
    distances = distances[np.triu_indices_from(distances, k=1)]
    distances = distances[distances > 0]
    return (
        float(np.quantile(distances, quantile)),
        float(distances.min()),
        float(distances.max()),
    )


def default_squared_lengthscale(
    spec: ParamSpec,
    inputs: np.ndarray,
    sample_size: int = 1000,
    key: Optional[Key[Array, ""]] = None,
    include_prior: bool = True,
) -> Param:
    """The squared-lengthscale Param laGP's darg would build."""
    spec = as_param(spec)
    learn = True if spec.learn is None else bool(spec.learn)
    needs_prior = include_prior and spec.prior is None
    needs_distances = spec.initial is None or (
        learn and (spec.upper is None or spec.lower is None or needs_prior)
    )
    start, smallest, largest = (
        pairwise_distance_summary(inputs, sample_size=sample_size, key=key)
        if needs_distances
        else (0.0, 0.0, 0.0)
    )

    initial = np.atleast_1d(
        np.asarray(start if spec.initial is None else spec.initial, float)
    )
    if spec.upper is not None:
        upper = float(spec.upper)
    else:
        upper = largest if learn else float(initial.max())
    if spec.lower is not None:
        lower = float(spec.lower)
    else:
        lower = max(
            smallest / 2.0 if learn else float(initial.min()), ROOT_MACHINE_EPSILON
        )
    if spec.prior is not None or not include_prior:
        prior = spec.prior
    else:
        prior = GammaPrior.with_mass_below(largest) if learn else None

    param = Param(initial=initial, lower=lower, upper=upper, prior=prior, learn=learn)
    param.validate()
    return param


def default_nugget(spec: ParamSpec, outputs: np.ndarray) -> Param:
    """The nugget Param laGP's garg would build."""
    spec = as_param(spec)
    learn = False if spec.learn is None else bool(spec.learn)
    outputs = np.asarray(outputs, float)
    squared_deviations = np.square(outputs - outputs.mean())

    initial = float(
        np.quantile(squared_deviations, 0.025) if spec.initial is None else spec.initial
    )
    if spec.upper is not None:
        upper = float(spec.upper)
    else:
        upper = float(squared_deviations.max()) if learn else initial
    lower = ROOT_MACHINE_EPSILON if spec.lower is None else float(spec.lower)
    if spec.prior is not None:
        prior = spec.prior
    else:
        prior = (
            GammaPrior.with_mass_below(float(squared_deviations.mean()))
            if learn
            else None
        )

    param = Param(initial=initial, lower=lower, upper=upper, prior=prior, learn=learn)
    param.validate()
    return param
