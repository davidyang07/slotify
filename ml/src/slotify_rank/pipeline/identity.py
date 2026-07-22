"""Deterministic cache identity for every derived Phase 3 artifact.

The rule this module exists to enforce: **an artifact is reusable only when
every input that could have changed its bytes is unchanged.** Timestamps are
never part of that judgement -- a file that is newer is not a file that is
correct, and mtimes do not survive a OneDrive sync or a fresh clone anyway.

A :class:`CacheIdentity` is a sorted mapping of named inputs reduced to one
SHA-256. Callers put in whatever actually determined the output: the audio
checksum, the model id and revision, the pooling windows, the library versions.
Anything omitted is, by construction, a claim that it cannot change the result.

Two identities that differ mean *recompute*; two that match mean *reuse*. There
is no third answer and no heuristic in between.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "CacheIdentity",
    "canonical_json",
    "config_digest",
    "library_versions",
]


def canonical_json(payload: Any) -> str:
    """JSON with sorted keys and no insignificant whitespace.

    Every digest in Phase 3 is taken over this rendering, so the same logical
    configuration always produces the same hash regardless of how the mapping
    was built or in what order YAML happened to yield its keys.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_encode_unknown,
    )


def _encode_unknown(value: Any) -> Any:
    # Tuples, sets and frozensets all appear in the config dataclasses. Rendering
    # them as sorted lists keeps the digest stable across runs (set iteration
    # order is not guaranteed between interpreter invocations).
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(
        f"{type(value).__name__} is not JSON-serializable and therefore cannot "
        "take part in a cache identity. Convert it to a primitive explicitly so "
        "the digest stays reproducible."
    )


def config_digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical rendering of a configuration mapping."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def library_versions(*module_names: str) -> dict[str, str]:
    """Installed versions of the named packages, for the cache identity.

    A missing package records ``"absent"`` rather than raising: identity must be
    computable before the heavy extras are installed (the CLI reports stage
    status without importing torch). ``"absent"`` differs from any real version,
    so an artifact produced with a library still invalidates if it is later
    removed -- which is the conservative direction.
    """
    from importlib.metadata import PackageNotFoundError, version

    resolved: dict[str, str] = {}
    for name in module_names:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = "absent"
    return resolved


@dataclass(frozen=True)
class CacheIdentity:
    """The complete set of inputs that determine one derived artifact.

    ``kind`` names the artifact family (``"transcript"``, ``"audio_embedding"``,
    ...) and is part of the digest, so two different stages that happen to share
    every input still get different keys.
    """

    kind: str
    inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("CacheIdentity.kind must be a non-empty string")
        if not isinstance(self.inputs, Mapping):
            raise TypeError(
                f"CacheIdentity.inputs must be a mapping, got "
                f"{type(self.inputs).__name__}"
            )
        # Fail loudly at construction rather than at digest time, so a
        # non-serializable input is attributed to the stage that introduced it.
        canonical_json(dict(self.inputs))

    @property
    def digest(self) -> str:
        """Full 64-character SHA-256 of ``kind`` plus every input."""
        return config_digest({"kind": self.kind, "inputs": dict(self.inputs)})

    @property
    def short(self) -> str:
        """First 16 characters -- enough for a filename, still collision-safe.

        16 hex characters is 64 bits. At corpus scale (order 10^5 artifacts) the
        collision probability is around 10^-10, and a collision would be caught
        anyway: :meth:`matches` compares the *full* digest recorded in the
        sidecar, not the filename.
        """
        return self.digest[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "digest": self.digest, "inputs": dict(self.inputs)}

    def matches(self, recorded: Mapping[str, Any] | None) -> bool:
        """True when a sidecar's recorded identity is this identity.

        A missing or malformed sidecar is *not* a match: an artifact whose
        provenance cannot be read is treated as stale, never as valid.
        """
        if not isinstance(recorded, Mapping):
            return False
        return str(recorded.get("digest", "")) == self.digest

    def explain_mismatch(self, recorded: Mapping[str, Any] | None) -> list[str]:
        """Human-readable reasons ``recorded`` differs from this identity.

        Used by the CLI to say *why* something is being recomputed. Without this
        a stale cache is indistinguishable from a bug in the cache key, and the
        usual response -- delete everything and rerun -- costs hours.
        """
        if recorded is None:
            return ["no cache metadata recorded"]
        if not isinstance(recorded, Mapping):
            return [f"cache metadata is {type(recorded).__name__}, expected a mapping"]
        if str(recorded.get("kind", "")) != self.kind:
            return [
                f"artifact kind {recorded.get('kind')!r} != expected {self.kind!r}"
            ]
        previous = recorded.get("inputs")
        if not isinstance(previous, Mapping):
            return ["cache metadata has no recorded inputs"]

        reasons: list[str] = []
        for key in sorted(set(previous) | set(self.inputs)):
            if key not in previous:
                reasons.append(f"{key}: newly part of the identity")
            elif key not in self.inputs:
                reasons.append(f"{key}: no longer part of the identity")
            elif canonical_json(previous[key]) != canonical_json(self.inputs[key]):
                reasons.append(
                    f"{key}: {canonical_json(previous[key])} -> "
                    f"{canonical_json(self.inputs[key])}"
                )
        return reasons or ["identity digest differs but no input changed"]
