"""Frozen dataclasses registered as JAX pytrees, for configuration objects.

A field marked `static()` is pytree metadata: it is part of the compiled program (changing
it recompiles) and must be hashable. Every other field is a child: a leaf or a nested
pytree, traced under `jit` and mapped over by `vmap`. That split is how a criterion's
`fmin` can change every call without recompiling while its `nugget` flag selects code.
"""

from dataclasses import dataclass, field, fields

import jax


def static(**kw):
    """A dataclass field that is pytree metadata rather than a leaf."""
    return field(metadata={"static": True}, **kw)


def pytree_config(cls):
    """Freeze `cls` as a dataclass and register it as a pytree, splitting fields by `static`."""
    cls = dataclass(frozen=True)(cls)
    meta = [f.name for f in fields(cls) if f.metadata.get("static")]
    data = [f.name for f in fields(cls) if not f.metadata.get("static")]
    return jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)
