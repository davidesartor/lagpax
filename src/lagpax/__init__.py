"""JAX port of the laGP R package (separable / GPsep family).

Everything here is float64 and follows laGP's parameterization: the correlation is
`exp(-sum_k (x1_k - x2_k)^2 / d_k)` with nugget `g` on the diagonal, the process is
zero-mean, and the scale is profiled out (`phi = Z' Ki Z`).
"""

import jax

jax.config.update("jax_enable_x64", True)

from .alc import alc_gpsep
from .covar import covar_sep, covar_sep_symm, diff_covar_sep, distance
from .gp import (
    SDEPS,
    GPsep,
    Prediction,
    dllik_gpsep_d,
    dllik_gpsep_g,
    llik_gpsep,
    new_gpsep,
    newparams_gpsep,
    pred_gpsep,
    update_gpsep,
)
from .local import AGP, LaGP, agpsep, lagpsep
from .mle import jmle_gpsep, mle_gpsep_d, mle_gpsep_nug, mle_switch
from .optim import bracketed_newton, lbfgsb
from .prior import Darg, Garg, darg, garg, get_ds, igamma_inv

__all__ = [
    "SDEPS",
    "AGP",
    "Darg",
    "GPsep",
    "Garg",
    "LaGP",
    "Prediction",
    "agpsep",
    "alc_gpsep",
    "bracketed_newton",
    "covar_sep",
    "covar_sep_symm",
    "darg",
    "diff_covar_sep",
    "distance",
    "dllik_gpsep_d",
    "dllik_gpsep_g",
    "garg",
    "get_ds",
    "igamma_inv",
    "jmle_gpsep",
    "lagpsep",
    "lbfgsb",
    "llik_gpsep",
    "mle_gpsep_d",
    "mle_gpsep_nug",
    "mle_switch",
    "new_gpsep",
    "newparams_gpsep",
    "pred_gpsep",
    "update_gpsep",
]
