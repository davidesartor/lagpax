"""Regression oracle: dump every user-visible output to .npz so a refactor can be checked.

Run `uv run --no-sync python -m tests.snapshot write out.npz` before a change and
`... check out.npz` after. Nothing here talks to R; it pins *our* numbers to themselves.
The key names outlive any one API, which is how the Phase 0 rename was proved to change
no value: the pre-rename snapshot was written by a script emitting these same keys.
"""

import sys
from pathlib import Path

import numpy as np

import lagpax as lagp
from lagpax.mle import fit
from lagpax.prior import Prior, Range, darg, garg

START, END = 6, 30
D_SMALL = 1e-3


def _design(n, m, seed, noise=0.0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, m))
    Z = np.sin(4 * X[:, 0]) + np.cos(3 * X[:, 1 : min(m, 3)].sum(-1)) + 0.1 * X.sum(-1)
    return X, Z + noise * rng.standard_normal(n)


DESIGNS = {
    "small": _design(40, 3, 0),
    "medium": _design(600, 4, 1),
    "noisy": _design(300, 3, 5, noise=0.2),
    "wide": _design(200, 8, 2),
}


def collect() -> dict[str, np.ndarray]:
    out = {}
    for name in ("small", "medium", "noisy"):
        X, Z = DESIGNS[name]
        d = np.linspace(0.3, 1.1, X.shape[1])
        gp = lagp.GP(X, Z, d, D_SMALL)
        XX = X[:6] + 0.05

        out[f"gp.{name}.phi"] = np.asarray(gp.phi, float)
        out[f"gp.{name}.ldetK"] = np.asarray(gp.ldetK, float)
        out[f"gp.{name}.llik"] = np.asarray(gp.llik(), float)
        out[f"gp.{name}.llik_prior"] = np.asarray(
            gp.llik(Prior(d=Range(ab=(1.5, 3.9)), g=Range(ab=(2.0, 5.0)))), float
        )

        lite = gp.predict(XX, lite=True)
        full = gp.predict(XX, lite=False)
        nonug = gp.predict(XX, lite=True, nugget=False)
        for key, val in [
            ("lite.mean", lite.mean),
            ("lite.s2", lite.s2),
            ("lite.df", lite.df),
            ("lite.llik", lite.llik),
            ("full.mean", full.mean),
            ("full.s2", full.s2),
            ("nonug.s2", nonug.s2),
        ]:
            out[f"pred.{name}.{key}"] = np.asarray(val, float)

        out[f"alc.{name}"] = np.asarray(
            lagp.alc(gp, X[:10] + 0.01, X[10:14] + 0.02), float
        )

        prior = Prior(d=darg(None, X), g=garg(1 / 10000, Z))
        for who, arg in (("d", prior.d), ("g", prior.g)):
            out[f"prior.{name}.{who}.start"] = np.atleast_1d(
                np.asarray(arg.start, float)
            )
            out[f"prior.{name}.{who}.min"] = np.asarray(arg.min, float)
            out[f"prior.{name}.{who}.max"] = np.asarray(arg.max, float)
            out[f"prior.{name}.{who}.ab"] = np.asarray(arg.ab, float)
            out[f"prior.{name}.{who}.estimate"] = np.asarray(float(arg.estimate))

        _, fitted = fit(gp, prior)
        out[f"fit.{name}.d"] = np.asarray(fitted.d, float)
        out[f"fit.{name}.conv"] = np.asarray(fitted.conv, float)

    for name in ("medium", "wide"):
        X, Z = DESIGNS[name]
        XX = X[:8] + 0.015
        emu = lagp.Local(start=START, end=END).condition(X, Z)
        for crit in ("nn", "alc"):
            a = emu.predict(XX, batch=4, criterion=crit)
            out[f"agp.{name}.{crit}.mean"] = np.asarray(a.mean, float)
            out[f"agp.{name}.{crit}.var"] = np.asarray(a.var, float)
            out[f"agp.{name}.{crit}.llik"] = np.asarray(a.llik, float)
            out[f"agp.{name}.{crit}.Xi"] = np.asarray(a.Xi, float)
            out[f"agp.{name}.{crit}.d"] = np.asarray(a.fit.d, float)

            q = emu._replace(scheme=emu.scheme._replace(criterion=crit)).local(
                X[:1] + 0.01
            )
            out[f"lagp.{name}.{crit}.mean"] = np.asarray(q.mean, float)
            out[f"lagp.{name}.{crit}.s2"] = np.asarray(q.s2, float)
            out[f"lagp.{name}.{crit}.df"] = np.asarray(q.df, float)
            out[f"lagp.{name}.{crit}.llik"] = np.asarray(q.llik, float)
            out[f"lagp.{name}.{crit}.Xi"] = np.asarray(q.Xi, float)
            out[f"lagp.{name}.{crit}.d"] = np.asarray(q.fit.d, float)

        r = emu.refine(XX, passes=("nn", "alc"), batch=4)
        out[f"refine.{name}.mean"] = np.asarray(r.mean, float)
        out[f"refine.{name}.var"] = np.asarray(r.var, float)
        out[f"refine.{name}.Xi"] = np.asarray(r.Xi, float)
        out[f"refine.{name}.d"] = np.asarray(r.fit.d, float)

    return out


def check(path: Path) -> int:
    """Compare against a stored snapshot; nonzero exit when anything moved."""
    want = np.load(path)
    got = collect()

    missing = sorted(set(want.files) - set(got))
    extra = sorted(set(got) - set(want.files))
    bad = []
    for key in sorted(set(want.files) & set(got)):
        a, b = np.asarray(want[key], float), np.asarray(got[key], float)
        if a.shape != b.shape or not np.array_equal(a, b):
            err = np.max(np.abs(a - b)) if a.shape == b.shape else np.inf
            bad.append((key, err))

    for key in missing:
        print(f"MISSING {key}")
    for key in extra:
        print(f"NEW     {key}")
    for key, err in bad:
        print(f"CHANGED {key}  max|delta| = {err:.3e}")
    print(f"{len(got) - len(bad) - len(extra)}/{len(want.files)} identical")
    return 1 if (missing or bad) else 0


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "write":
        arrays = collect()
        np.savez(path, **arrays)
        print(f"wrote {len(arrays)} arrays to {path}")
    else:
        raise SystemExit(check(path))
