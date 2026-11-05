"""Frozen dataclasses that are JAX pytrees.

A field declared with `static()` is pytree metadata: part of the compiled program, so it must be
hashable and changing it recompiles. Every other field is a child, traced under `jit`.
"""

import dataclasses
from typing import Any, TypeVar

import jax

T = TypeVar("T")


def static(**field_options: Any) -> Any:
    return dataclasses.field(metadata={"static": True}, **field_options)


def pytree_dataclass(cls: type[T]) -> type[T]:
    cls = dataclasses.dataclass(frozen=True)(cls)
    fields = dataclasses.fields(cls)
    return jax.tree_util.register_dataclass(
        cls,
        data_fields=[f.name for f in fields if not f.metadata.get("static")],
        meta_fields=[f.name for f in fields if f.metadata.get("static")],
    )
