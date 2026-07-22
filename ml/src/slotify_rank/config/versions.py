"""Version stamps embedded in every artifact this package produces.

Bumping any of these invalidates the artifacts that carry the old value. The
rules are:

``PACKAGE_VERSION``
    Source version of ``slotify_rank`` itself. Informational.

``CANDIDATE_SCHEMA_VERSION``
    Shape of the serialized candidate record
    (:mod:`slotify_rank.candidates.schema`). Bump on any field addition,
    removal or semantic change; downstream readers compare it explicitly.

``FEATURE_SPEC_VERSION``
    Reserved for Phase 3. Named here so the schema can carry the field from the
    start and Phase 3 does not need a migration.

``PREPROCESSING_VERSION``
    Reserved for Phase 2/3 (decode + transcription settings).

The heuristic configuration version is **not** listed here: it is owned by
``config/heuristic_offline_v1.json`` and read at load time, so the constants and
their version can never drift apart.
"""

from __future__ import annotations

PACKAGE_VERSION = "0.1.0"
CANDIDATE_SCHEMA_VERSION = "candidate-schema-v1.0.0"

# Reserved; populated when the corresponding phase lands.
FEATURE_SPEC_VERSION = "unset-phase3"
PREPROCESSING_VERSION = "unset-phase2"

__all__ = [
    "PACKAGE_VERSION",
    "CANDIDATE_SCHEMA_VERSION",
    "FEATURE_SPEC_VERSION",
    "PREPROCESSING_VERSION",
]
