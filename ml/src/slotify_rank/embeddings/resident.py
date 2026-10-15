"""Models that stay loaded for the life of a process, when -- and only when -- asked.

The corpus pipeline loads each model once per stage and releases it before the
next stage loads its own (see ``pipeline/stages.py``): Whisper and MiniLM never
resident at once is what keeps a corpus run inside laptop RAM. That is the right
trade for a batch job that touches every model once.

The inference worker (``infer serve``) makes the opposite trade. It answers
request after request, and reloading Whisper, the Whisper encoder and MiniLM for
each one costs more than everything else a short clip needs. Three small CPU
models (~0.5 GB together) resident for the worker's lifetime is cheap by
comparison.

So the stages stay exactly as they are, and the three model wrappers ask this
module for their weights. Outside a :func:`resident_models` scope -- the corpus
pipeline, the one-shot ``infer rank``, every test that does not open one --
:func:`acquire` simply builds, and ``release()`` frees, as before. Inside a scope,
the first ``load()`` of a given model builds it and every later ``load()`` of the
same model gets the same object back, and ``release()`` leaves it alone.

"The same model" is the key the wrapper passes: model id, revision, device and
dtype, plus the wrapper's own kind. Any setting that changes the loaded weights
must be in it, so two configurations can never be served by one cached object.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Hashable, Iterator, TypeVar

__all__ = ["acquire", "is_open", "is_resident", "resident_count", "resident_models"]

T = TypeVar("T")

#: ``None`` outside a scope. Inside one: key -> whatever the builder returned.
_scope: dict[Hashable, Any] | None = None


@contextmanager
def resident_models() -> Iterator[None]:
    """Keep every model loaded inside this block until the block exits.

    Not re-entrant: a nested scope would have to choose between sharing the
    outer one's models and freeing them early, and neither is obviously right.
    """
    global _scope
    if _scope is not None:
        raise RuntimeError("resident_models() is already open in this process")
    _scope = {}
    try:
        yield
    finally:
        held, _scope = _scope, None
        _free_all(held)


def acquire(key: Hashable, build: Callable[[], T]) -> T:
    """``build()``, or -- inside a scope -- the object built earlier for ``key``."""
    if _scope is None:
        return build()
    if key not in _scope:
        _scope[key] = build()
    return _scope[key]


def is_resident(obj: Any) -> bool:
    """True when ``obj`` (or a tuple element of a held value) is held by the scope.

    Compared by identity: a wrapper's ``release()`` asks this about the exact
    model object it was handed, and an equal-but-different object is not held.
    """
    if _scope is None or obj is None:
        return False
    for value in _scope.values():
        if value is obj:
            return True
        if isinstance(value, tuple) and any(item is obj for item in value):
            return True
    return False


def is_open() -> bool:
    """True inside a :func:`resident_models` scope."""
    return _scope is not None


def resident_count() -> int:
    """How many models the open scope holds (0 when no scope is open)."""
    return 0 if _scope is None else len(_scope)


def _free_all(held: dict[Hashable, Any]) -> None:
    from slotify_rank.embeddings.device import free_model

    # Drop the scope's references first; free_model then collects what they
    # were keeping alive and returns any CUDA memory, as for a released model.
    held.clear()
    free_model(None)
