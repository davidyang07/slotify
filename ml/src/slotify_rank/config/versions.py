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
    That includes which merged candidates survive the density cap, since a
    change there changes the pool every label and feature is attached to.

``SPLIT_ALGORITHM_VERSION``
    Grouping and assignment algorithm in :mod:`slotify_rank.data.splits`. A
    split manifest is immutable for a given version; changing the algorithm
    requires a bump, which is what makes an existing split safe to trust.

``LABEL_RUBRIC_VERSION``
    The 1-5 naturalness rubric in ``docs/labelling-guide.md``. Stored on every
    label so labels collected under different rubrics are never pooled.

``LABEL_SCHEMA_VERSION``
    Shape of a stored and exported label row. Independent of the rubric: the
    same judgement can be recorded in a richer row without the judgement itself
    meaning anything different.

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
# v1.1.0 added the optional `provenance` mapping: where a fetched episode came
# from upstream and how its licence was established. Additive and optional, so
# v1.0.0 records still read (they simply carry no provenance).
EPISODE_SCHEMA_VERSION = "episode-schema-v1.1.0"
# v1.1.0 added the optional `generation_config_digest`: which candidate
# generation settings produced the record, so a feature cache can tell a pool
# regenerated under different settings from the one it was built for. Additive
# and optional, so v1.0.0 records still read (with an empty digest).
DATASET_CANDIDATE_SCHEMA_VERSION = "dataset-candidate-schema-v1.1.0"
SOURCE_MANIFEST_VERSION = "source-manifest-v1.0.0"
PREPROCESSING_VERSION = "preprocess-v1.0.0"
# v1.1.0: the density cap is applied per one-minute window and ranks
# candidates within a window by signal evidence (pause length, valley depth,
# source agreement). v1.0.0 kept the global top-N by heuristic score, which tied
# almost everything at 0.4/0.5 and so emptied the first and last fifth of every
# episode of everything but silences -- a pool biased toward the baseline it is
# meant to be compared against.
# v1.1.0 also scores a merged survivor with its *own* pause and snippet, as the
# product's ``mergeCandidates`` does, instead of the group's longest pause and
# a borrowed snippet; ``heuristic_score`` on a v1.0.0 row can differ from the
# product's for the same timestamp.
CANDIDATE_GENERATION_VERSION = "candgen-v1.1.0"
SPLIT_ALGORITHM_VERSION = "split-grouped-greedy-v1.0.0"
#: The stratified variant. It groups on series exactly as the greedy algorithm
#: does -- the leakage guarantee is identical -- and differs only in *ordering*:
#: it balances each content type across the partitions separately, so a corpus
#: whose podcasts are a minority of its hours cannot end up with all of them in
#: one partition. Recorded per manifest rather than replacing the constant
#: above, so manifests produced by either algorithm stay readable.
#:
#: v1.1.0 computes each content type's duration targets, count targets and
#: running totals *within that content type*, which is what the algorithm always
#: claimed to do. v1.0.0 measured every stratum against corpus-wide targets, so
#: whichever stratum was assigned first could fill a partition's whole budget
#: and leave the next one nothing: 20 narrated series and 14 podcasts put zero
#: podcasts in validation. v1.1.0 is opt-in: a config gets it only by pinning
#: it (split v5 does).
STRATIFIED_SPLIT_ALGORITHM_VERSION = "split-stratified-greedy-v1.1.0"
#: The algorithm that produced split v4, the canonical split experiment v2 is
#: frozen against. Still implemented, unchanged, and selected by every
#: stratified config that does not pin an algorithm, so v4 regenerates byte for
#: byte.
LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION = "split-stratified-greedy-v1.0.0"
#: Every algorithm version this build can read a manifest from.
SUPPORTED_SPLIT_ALGORITHM_VERSIONS = (
    SPLIT_ALGORITHM_VERSION,
    STRATIFIED_SPLIT_ALGORITHM_VERSION,
    LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION,
)
LABEL_RUBRIC_VERSION = "rubric-v1.0.0"
#: Shape of one stored judgement (:mod:`slotify_rank.labelling.database`) and
#: of one exported label row. v1.1.0 keys a judgement on the *presentation*
#: rather than the candidate, so a blind repeat of the same candidate is a
#: second row instead of an overwrite, and records the split, series, stage,
#: queue version and time-on-item alongside the score.
LABEL_SCHEMA_VERSION = "label-schema-v1.1.0"

# Phase 3 (multimodal features and embeddings).
# v1.1.0: a segment whose end timestamp is missing (Whisper's trailing segment,
# or a chunk decoded without offsets) now ends at the next segment's start or
# the chunk end instead of collapsing to zero length and being dropped. The
# stored shape is unchanged, so v1.0.0 transcripts still read; they are simply
# re-derived because the version is part of every transcript's cache identity.
TRANSCRIPTION_VERSION = "transcript-v1.1.0"
# v1.1.0: the heuristic component names were corrected to the scorer's actual
# additive terms (base/pause/mode/sentence/position/edge), and raw_total and
# clamped were added. The previous names were guesses that matched nothing, so
# every one of those columns was silently 100% missing.
FEATURE_SPEC_VERSION = "featurespec-v1.1.0"
# v1.1.0: a candidate inside a transcript segment is treated as mid-sentence
# (``transcript_sentence_end`` 0, no pause, the straddled text split at the
# candidate into the before/after context). v1.0.0 read the boundary signal off
# the segment *before* the straddled one, so a mid-sentence candidate looked
# like a clean sentence end and the sentence it interrupted vanished from both
# sides. The feature vocabulary is unchanged, so FEATURE_SPEC_VERSION is not.
FEATURE_PIPELINE_VERSION = "featurepipeline-v1.1.0"
FEATURE_MANIFEST_SCHEMA_VERSION = "feature-record-schema-v1.0.0"
EMBEDDING_STORE_VERSION = "embedding-store-v1.0.0"

# Phase 4 (PyTorch ranking dataset, models and training).
#: Shape of an assembled training example plus the eligibility rules that decide
#: which candidates become one. Bumping it invalidates a prepared dataset
#: artifact; it is recorded in every checkpoint so a model can never be resumed
#: against examples built under different rules.
TRAINING_DATASET_VERSION = "training-dataset-v1.0.0"
#: Fitted scalar-normalization artifact: the statistics *and* the fitting rules
#: (train-split-only, epsilon floor, constant-feature handling).
NORMALIZER_VERSION = "normalizer-v1.0.0"
#: Model input contract: the modality construction (which stored vectors are
#: concatenated, in what order) plus the shared ranker interface. A checkpoint
#: whose input schema version differs cannot be loaded, because its weights are
#: aligned to a different column layout.
MODEL_INPUT_SCHEMA_VERSION = "model-input-schema-v1.0.0"
#: Shape of a saved checkpoint payload (:mod:`slotify_rank.training.checkpoint`).
CHECKPOINT_SCHEMA_VERSION = "checkpoint-schema-v1.0.0"
#: Deterministic within-episode pair construction
#: (:mod:`slotify_rank.ranking.pairs`).
PAIR_GENERATION_VERSION = "pairgen-v1.0.0"

# Phase 5 (real corpus bootstrap, human-labelling queues, experiment gate).
#: Shape of a stratified labelling-queue artifact
#: (:mod:`slotify_rank.labelling.queue`). Bumping it invalidates a queue file;
#: the queue records the candidate and split manifest hashes it was built from,
#: so a queue built against a regenerated corpus is detectable rather than
#: silently mismatched.
LABELLING_QUEUE_SCHEMA_VERSION = "labelling-queue-v1.0.0"
#: Shape of the committed reduction of a queue
#: (:func:`slotify_rank.labelling.queue.queue_summary`). The queue artifact
#: itself names candidate ids and lives under the uncommitted ``data/`` tree;
#: this is the counts-and-hashes view that ships as evidence.
LABELLING_QUEUE_SUMMARY_SCHEMA_VERSION = "labelling-queue-summary-v1.0.0"
#: Shape of a frozen label snapshot (:mod:`slotify_rank.experiment.freeze`). A
#: snapshot is immutable once written; a correction is a new version.
LABEL_SNAPSHOT_SCHEMA_VERSION = "label-snapshot-v1.0.0"
#: Shape of the readiness report and the experiment manifest
#: (:mod:`slotify_rank.experiment.readiness`).
EXPERIMENT_MANIFEST_VERSION = "experiment-v1.0.0"

#: Schema versions this build is able to read. Anything else is a hard failure
#: rather than a best-effort parse, because a silently mis-read manifest would
#: corrupt every number downstream of it.
SUPPORTED_EPISODE_SCHEMA_VERSIONS = frozenset(
    {EPISODE_SCHEMA_VERSION, "episode-schema-v1.0.0"}
)
SUPPORTED_DATASET_CANDIDATE_SCHEMA_VERSIONS = frozenset(
    {DATASET_CANDIDATE_SCHEMA_VERSION, "dataset-candidate-schema-v1.0.0"}
)
SUPPORTED_SOURCE_MANIFEST_VERSIONS = frozenset({SOURCE_MANIFEST_VERSION})
SUPPORTED_TRANSCRIPTION_VERSIONS = frozenset(
    {TRANSCRIPTION_VERSION, "transcript-v1.0.0"}
)
SUPPORTED_FEATURE_MANIFEST_SCHEMA_VERSIONS = frozenset(
    {FEATURE_MANIFEST_SCHEMA_VERSION}
)
SUPPORTED_EMBEDDING_STORE_VERSIONS = frozenset({EMBEDDING_STORE_VERSION})
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = frozenset({CHECKPOINT_SCHEMA_VERSION})

__all__ = [
    "PACKAGE_VERSION",
    "CANDIDATE_SCHEMA_VERSION",
    "EPISODE_SCHEMA_VERSION",
    "DATASET_CANDIDATE_SCHEMA_VERSION",
    "SOURCE_MANIFEST_VERSION",
    "PREPROCESSING_VERSION",
    "CANDIDATE_GENERATION_VERSION",
    "SPLIT_ALGORITHM_VERSION",
    "STRATIFIED_SPLIT_ALGORITHM_VERSION",
    "LEGACY_STRATIFIED_SPLIT_ALGORITHM_VERSION",
    "SUPPORTED_SPLIT_ALGORITHM_VERSIONS",
    "LABEL_RUBRIC_VERSION",
    "LABEL_SCHEMA_VERSION",
    "TRANSCRIPTION_VERSION",
    "FEATURE_SPEC_VERSION",
    "FEATURE_PIPELINE_VERSION",
    "FEATURE_MANIFEST_SCHEMA_VERSION",
    "EMBEDDING_STORE_VERSION",
    "TRAINING_DATASET_VERSION",
    "NORMALIZER_VERSION",
    "MODEL_INPUT_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "PAIR_GENERATION_VERSION",
    "LABELLING_QUEUE_SCHEMA_VERSION",
    "LABELLING_QUEUE_SUMMARY_SCHEMA_VERSION",
    "LABEL_SNAPSHOT_SCHEMA_VERSION",
    "EXPERIMENT_MANIFEST_VERSION",
    "SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS",
    "SUPPORTED_EPISODE_SCHEMA_VERSIONS",
    "SUPPORTED_DATASET_CANDIDATE_SCHEMA_VERSIONS",
    "SUPPORTED_SOURCE_MANIFEST_VERSIONS",
    "SUPPORTED_TRANSCRIPTION_VERSIONS",
    "SUPPORTED_FEATURE_MANIFEST_SCHEMA_VERSIONS",
    "SUPPORTED_EMBEDDING_STORE_VERSIONS",
]
