"""Version stamps embedded in every artifact this package produces.

Bumping any of these invalidates the artifacts that carry the old value. The
rules are:

``PACKAGE_VERSION``
    Source version of ``slotify_rank`` itself. Informational.

``CANDIDATE_SCHEMA_VERSION``
    Shape of the serialized *ranking* candidate record
    (:mod:`slotify_rank.candidates.schema`). Bump on any field addition,
    removal or semantic change; downstream readers compare it explicitly.

``EPISODE_SCHEMA_VERSION``
    Shape of a dataset episode record (:mod:`slotify_rank.data.schema`).

``DATASET_CANDIDATE_SCHEMA_VERSION``
    Shape of a dataset candidate record (:mod:`slotify_rank.data.schema`). This
    is the labelled/trainable record and is deliberately versioned separately
    from ``CANDIDATE_SCHEMA_VERSION``: the ranking record is a product-parity
    artifact, this one is a dataset artifact, and they move independently.

``SOURCE_MANIFEST_VERSION``
    Shape of ``ml/configs/sources.yaml``.

``PREPROCESSING_VERSION``
    Decode + normalization settings (16 kHz / mono / PCM s16 WAV). Bumping it
    invalidates every cached normalized file: :mod:`slotify_rank.data.normalize`
    re-renders when the recorded version differs from this one.

``CANDIDATE_GENERATION_VERSION``
    Generator set and their parameters' *semantics* (not their tuned values,
    which live in ``ml/configs/dataset_v1.yaml`` and are recorded per run).

``SPLIT_ALGORITHM_VERSION``
    Grouping and assignment algorithm in :mod:`slotify_rank.data.splits`. A
    split manifest is immutable for a given version; changing the algorithm
    requires a bump, which is what makes an existing split safe to trust.

``LABEL_RUBRIC_VERSION``
    The 1-5 naturalness rubric in ``docs/labelling-guide.md``. Stored on every
    label so labels collected under different rubrics are never pooled.

``FEATURE_SPEC_VERSION``
    Reserved for Phase 3 (learned features / embeddings).

The heuristic configuration version is **not** listed here: it is owned by
``config/heuristic_offline_v1.json`` and read at load time, so the constants and
their version can never drift apart.
"""

from __future__ import annotations

PACKAGE_VERSION = "0.2.0"
CANDIDATE_SCHEMA_VERSION = "candidate-schema-v1.0.0"

# Phase 2 (dataset foundation).
EPISODE_SCHEMA_VERSION = "episode-schema-v1.0.0"
DATASET_CANDIDATE_SCHEMA_VERSION = "dataset-candidate-schema-v1.0.0"
SOURCE_MANIFEST_VERSION = "source-manifest-v1.0.0"
PREPROCESSING_VERSION = "preprocess-v1.0.0"
CANDIDATE_GENERATION_VERSION = "candgen-v1.0.0"
SPLIT_ALGORITHM_VERSION = "split-grouped-greedy-v1.0.0"
LABEL_RUBRIC_VERSION = "rubric-v1.0.0"

# Reserved; populated when the corresponding phase lands.
FEATURE_SPEC_VERSION = "unset-phase3"

#: Schema versions this build is able to read. Anything else is a hard failure
#: rather than a best-effort parse, because a silently mis-read manifest would
#: corrupt every number downstream of it.
SUPPORTED_EPISODE_SCHEMA_VERSIONS = frozenset({EPISODE_SCHEMA_VERSION})
SUPPORTED_DATASET_CANDIDATE_SCHEMA_VERSIONS = frozenset(
    {DATASET_CANDIDATE_SCHEMA_VERSION}
)
SUPPORTED_SOURCE_MANIFEST_VERSIONS = frozenset({SOURCE_MANIFEST_VERSION})

__all__ = [
    "PACKAGE_VERSION",
    "CANDIDATE_SCHEMA_VERSION",
    "EPISODE_SCHEMA_VERSION",
    "DATASET_CANDIDATE_SCHEMA_VERSION",
    "SOURCE_MANIFEST_VERSION",
    "PREPROCESSING_VERSION",
    "CANDIDATE_GENERATION_VERSION",
    "SPLIT_ALGORITHM_VERSION",
    "LABEL_RUBRIC_VERSION",
    "FEATURE_SPEC_VERSION",
    "SUPPORTED_EPISODE_SCHEMA_VERSIONS",
    "SUPPORTED_DATASET_CANDIDATE_SCHEMA_VERSIONS",
    "SUPPORTED_SOURCE_MANIFEST_VERSIONS",
]
