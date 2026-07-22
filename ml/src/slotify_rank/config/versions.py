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

``TRANSCRIPTION_VERSION``
    Shape of a stored transcript artifact plus the segment-derivation rules in
    :mod:`slotify_rank.transcription.segments`. The *model* is not part of this
    constant -- it is part of each transcript's cache identity -- so switching
    from ``whisper-tiny.en`` to a larger model invalidates the affected
    transcripts without invalidating the format.

``FEATURE_SPEC_VERSION``
    The handcrafted feature *vocabulary*: which named scalar features exist and
    in what canonical order. Bump on any addition, removal or redefinition; a
    stored feature matrix whose spec version differs is unreadable rather than
    silently mis-columned.

``FEATURE_PIPELINE_VERSION``
    End-to-end behaviour of the Phase 3 pipeline: window construction, pooling,
    context selection, assembly. Participates in every cache identity, so
    bumping it invalidates all derived artifacts.

``FEATURE_MANIFEST_SCHEMA_VERSION``
    Shape of a candidate feature record
    (:mod:`slotify_rank.features.schema`).

``EMBEDDING_STORE_VERSION``
    On-disk layout of a ``.npy`` array plus its JSON sidecar
    (:mod:`slotify_rank.embeddings.store`).

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

# Phase 3 (multimodal features and embeddings).
TRANSCRIPTION_VERSION = "transcript-v1.0.0"
# v1.1.0: the heuristic component names were corrected to the scorer's actual
# additive terms (base/pause/mode/sentence/position/edge), and raw_total and
# clamped were added. The previous names were guesses that matched nothing, so
# every one of those columns was silently 100% missing.
FEATURE_SPEC_VERSION = "featurespec-v1.1.0"
FEATURE_PIPELINE_VERSION = "featurepipeline-v1.0.0"
FEATURE_MANIFEST_SCHEMA_VERSION = "feature-record-schema-v1.0.0"
EMBEDDING_STORE_VERSION = "embedding-store-v1.0.0"

#: Schema versions this build is able to read. Anything else is a hard failure
#: rather than a best-effort parse, because a silently mis-read manifest would
#: corrupt every number downstream of it.
SUPPORTED_EPISODE_SCHEMA_VERSIONS = frozenset({EPISODE_SCHEMA_VERSION})
SUPPORTED_DATASET_CANDIDATE_SCHEMA_VERSIONS = frozenset(
    {DATASET_CANDIDATE_SCHEMA_VERSION}
)
SUPPORTED_SOURCE_MANIFEST_VERSIONS = frozenset({SOURCE_MANIFEST_VERSION})
SUPPORTED_TRANSCRIPTION_VERSIONS = frozenset({TRANSCRIPTION_VERSION})
SUPPORTED_FEATURE_MANIFEST_SCHEMA_VERSIONS = frozenset(
    {FEATURE_MANIFEST_SCHEMA_VERSION}
)
SUPPORTED_EMBEDDING_STORE_VERSIONS = frozenset({EMBEDDING_STORE_VERSION})

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
    "TRANSCRIPTION_VERSION",
    "FEATURE_SPEC_VERSION",
    "FEATURE_PIPELINE_VERSION",
    "FEATURE_MANIFEST_SCHEMA_VERSION",
    "EMBEDDING_STORE_VERSION",
    "SUPPORTED_EPISODE_SCHEMA_VERSIONS",
    "SUPPORTED_DATASET_CANDIDATE_SCHEMA_VERSIONS",
    "SUPPORTED_SOURCE_MANIFEST_VERSIONS",
    "SUPPORTED_TRANSCRIPTION_VERSIONS",
    "SUPPORTED_FEATURE_MANIFEST_SCHEMA_VERSIONS",
    "SUPPORTED_EMBEDDING_STORE_VERSIONS",
]
