# Capability and claim evidence

**Generated file - do not edit.** Produced by `slotify-rank report claim-evidence` (`npm run claim-evidence`), which reads the artifacts named at the bottom. Every number here comes from one of them; the thresholds come from the committed experiment definition.

- Git SHA: `5658cda766e73955c20bd033e8e5b576137d185a`
- Generated: 2026-09-03T10:45:02+00:00
- Package version: 0.2.0
- Verdict: **19 PASS / 2 FAIL / 6 NOT MEASURED**
  - implemented capabilities: 19 PASS / 0 FAIL / 0 NOT MEASURED
  - empirical results: 0 PASS / 2 FAIL / 6 NOT MEASURED

`NOT MEASURED` means no artifact produced that value. It is not a failure and it is not a zero: a claim nobody has measured and a claim measured and found short are different facts.

Two kinds of claim are tracked and never mixed. An **implementation** claim is settled by committed code, the test that exercises it and the artifact it produces: it says a capability exists and runs, never that it measured anything. An **empirical** claim is one only a measurement can settle. A PASS in the first table is not evidence for anything in the second.

## Implemented capabilities

Settled by code, tests and generated artifacts.

| Capability | Status | Evidence |
| --- | --- | --- |
| A committed, versioned experiment definition fixes the protocol in advance. | **PASS** | experiment-v2 declares the split, the seeds, the headline variant, the canonical baseline and the claim thresholds this report is checked against. |
| A multimodal PyTorch ranker exists and is trained. | **PASS** | gated with 489477 parameters, trained on label_source=weak_heuristic. |
| The ranker consumes both audio and transcript features. | **PASS** | Speech representations from openai/whisper-tiny.en; transcript embeddings from sentence-transformers/all-MiniLM-L6-v2; plus 110 handcrafted acoustic and structural scalars. |
| Deterministic candidate generation runs over the corpus and records which generator proposed each breakpoint. | **PASS** | 13176 candidate(s) generated across 4 generator(s) (fixed_interval, pause, rms_minimum, silence), covered by 2 test module(s). |
| Every candidate is preprocessed into one feature record holding handcrafted scalars, a speech representation and a transcript embedding. | **PASS** | 12930 candidate(s) hold a complete record of 110 handcrafted scalars plus both embeddings; covered by 7 test module(s). |
| A stratified, blind labelling round is built and servable: the queue reaches its target size and carries unmarked repeat presentations. | **PASS** | Queue full-v2 holds 2400 unique candidate(s) against a 2400 target, served as 2460 presentation(s) including 60 blind repeat(s); covered by 8 test module(s). This is the size of the queue, not a count of labels collected against it. |
| Labels are frozen into an immutable, hashed snapshot that a rerun cannot silently rewrite. | **PASS** | ml/src/slotify_rank/experiment/freeze.py writes a hashed snapshot; the experiment declares snapshot version 'full-v2'; covered by 7 test module(s). |
| A grouped, non-degraded train/validation/test split exists and its test groups appear in no other partition. | **PASS** | Split v4 groups by series with seed 42: 30 train / 7 validation / 3 test series, disjoint. |
| The experiment declares a full ablation matrix, every variant it names has a committed config and a registered implementation, and the matrix runner is tested. | **PASS** | 5 variant(s) x 3 seed(s) = 15 cells; every variant has a config under ml/configs/models; the runner is ml/src/slotify_rank/experiment/matrix.py, covered by 2 test module(s). Declaring the matrix is not running it. |
| Checkpoints and seeds are selected on validation only, by a rule fixed before the test split is read. | **PASS** | Checkpoints are selected on validation_ndcg_at_3 and the reported seed is the median_validation_ndcg_at_3; both are declared in experiment-v2 and implemented in ml/src/slotify_rank/training/checkpoint.py, covered by 3 test module(s). |
| A scikit-learn classical baseline is implemented over the handcrafted features, tuned by group-aware cross-validation inside the training split. | **PASS** | ml/src/slotify_rank/baselines/classical.py fits a HistGradientBoostingRegressor selected by GroupKFold, covered by 1 test module(s). Whether it has been fitted on the canonical corpus is a separate, empirical question. |
| The headline metric is cross-checked against sklearn.metrics.ndcg_score by an independent implementation, and publication is blocked if they disagree. | **PASS** | ml/src/slotify_rank/evaluation/crosscheck.py calls sklearn.metrics.ndcg_score against this project's own NDCG, covered by 2 test module(s). The experiment requires the cross-check before a number is published: require_independent_metric_crosscheck=True. |
| Held-out uncertainty is reported as a percentile bootstrap resampled over episodes, with the resample count, confidence and seed fixed in advance. | **PASS** | ml/src/slotify_rank/evaluation/compare.py resamples episodes; the experiment fixes 2000 resamples at 0.95 confidence with seed 20260829, covered by 2 test module(s). |
| The experiment resolves to a manifest that pins the exact bytes it will be reported against: a config digest plus a hash per input artifact. | **PASS** | experiment-v2 resolves to config digest c1e09d5e34108330... and pins 18 input artifact(s): baseline_config, candidate_manifest, corpus_plan, episode_manifest, experiment_config, feature_manifest, labelling_queue, labelling_queue_config, model_config:audio_only, model_config:concat, model_config:gated, model_config:handcrafted, model_config:text_only, source_registry:sources_real_v1.yaml, source_registry:sources_v2.yaml, split_config, split_manifest, training_config. |
| The ranker is served through a TypeScript product: a React frontend and a Node API, each with a lint/typecheck/test command CI runs. | **PASS** | frontend react ^19.2.0 with scripts ['build', 'dev', 'lint', 'preview', 'test', 'verify']; backend scripts ['dev', 'start', 'test', 'typecheck', 'verify']. |
| The AI/ML stack includes PyTorch. | **PASS** | Declared, imported by 14 module(s) (e.g. datasets/collate.py, datasets/ranking_dataset.py, embeddings/device.py), and a training run recorded a torch version and a parameter count. |
| The AI/ML stack includes Hugging Face `transformers`. | **PASS** | Declared, imported by 2 module(s) (e.g. embeddings/whisper_audio.py, transcription/local_whisper.py), and the embedding statistics name Hugging Face model ids for both the speech and the transcript encoder. |
| The AI/ML stack includes librosa. | **PASS** | Declared, imported by 2 module(s) (e.g. features/acoustic.py, transcription/local_whisper.py), and the feature statistics record a librosa version and the handcrafted vocabulary contains librosa-derived spectral descriptors. |
| The award is described precisely as 'UofTHacks 13 — MLH Best Use of ElevenLabs', with no implied overall win. | **PASS** | README.md states 'UofTHacks 13 — MLH Best Use of ElevenLabs' and makes no stronger claim. |

## Empirical results

Settled only by a measurement. Nothing above can supply one.

| Claim | Status | Evidence |
| --- | --- | --- |
| At least 2400 human-labelled candidates exist. | **FAIL** | 0 human-labelled candidates, short of the 2400 the experiment requires. 13176 candidates have been GENERATED, which is not the same thing and is never counted as one. |
| The evaluated checkpoint was trained on human labels, not weak ones. | **FAIL** | Every training run records a non-human label source (weak_heuristic), so no checkpoint reflects human judgement. |
| The published improvement is quoted against the canonical frozen heuristic baseline. | **NOT MEASURED** | No publishable comparison exists, so no baseline was used. |
| The baseline's held-out NDCG@3 is measured. | **NOT MEASURED** | No publishable held-out comparison exists. |
| The learned model's held-out NDCG@3 is measured. | **NOT MEASURED** | No publishable held-out comparison exists. |
| The relative NDCG@3 improvement is measured. | **NOT MEASURED** | No publishable held-out comparison exists. |
| The relative NDCG@3 improvement is at least 18.0 % over the heuristic baseline. | **NOT MEASURED** | No measured improvement, or no declared threshold, so the claim cannot be evaluated. It is NOT supported by default. |
| scikit-learn ran inside a published comparison: an independent NDCG cross-check and a fitted classical baseline. | **NOT MEASURED** | Declared and imported by 2 module(s), but no artifact yet shows it running: a published comparison records BOTH an independent scikit-learn NDCG cross-check and a fitted scikit-learn classical baseline. |

## Measured quantities

| Quantity | Value |
| --- | --- |
| Human-labelled candidates | 0 |
| ...required by the experiment | 2400 |
| Generated candidates (NOT labels) | 13176 |
| Complete multimodal feature records | 12930 |
| Processed audio hours | 18.6135 |
| Episodes | 77 |
| Series | 40 |
| Model variant | gated |
| Model parameters | 489477 |
| Trained on label source | weak_heuristic |
| Audio encoder | openai/whisper-tiny.en |
| Text encoder | sentence-transformers/all-MiniLM-L6-v2 |

## Held-out result

| Field | Value |
| --- | --- |
| Evaluation id | NOT MEASURED |
| Baseline NDCG@3 | NOT MEASURED |
| Model NDCG@3 | NOT MEASURED |
| Absolute improvement | NOT MEASURED |
| Relative improvement | NOT MEASURED |
| ...required by the claim | 18.0000 |

## Detail

### A committed, versioned experiment definition fixes the protocol in advance. - PASS

*implementation claim*

experiment-v2 declares the split, the seeds, the headline variant, the canonical baseline and the claim thresholds this report is checked against.

```json
{
  "experiment_version": "experiment-v2",
  "split_version": "v4",
  "seeds": [
    42,
    43,
    44
  ],
  "headline_variant": "gated"
}
```

### A multimodal PyTorch ranker exists and is trained. - PASS

*implementation claim*

gated with 489477 parameters, trained on label_source=weak_heuristic.

```json
{
  "model_variant": "gated",
  "model_parameter_count": 489477,
  "training_label_source": "weak_heuristic"
}
```

### The ranker consumes both audio and transcript features. - PASS

*implementation claim*

Speech representations from openai/whisper-tiny.en; transcript embeddings from sentence-transformers/all-MiniLM-L6-v2; plus 110 handcrafted acoustic and structural scalars.

```json
{
  "audio_embedding_model": "openai/whisper-tiny.en",
  "text_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "handcrafted_feature_count": 110,
  "complete_multimodal_candidate_count": 12930
}
```

### Deterministic candidate generation runs over the corpus and records which generator proposed each breakpoint. - PASS

*implementation claim*

13176 candidate(s) generated across 4 generator(s) (fixed_interval, pause, rms_minimum, silence), covered by 2 test module(s).

```json
{
  "generated_candidate_count": 13176,
  "candidates_by_primary_source": {
    "fixed_interval": 795,
    "pause": 4346,
    "rms_minimum": 2880,
    "silence": 5155
  },
  "tests": [
    "test_candidates_dataset.py",
    "test_heuristic.py"
  ]
}
```

### Every candidate is preprocessed into one feature record holding handcrafted scalars, a speech representation and a transcript embedding. - PASS

*implementation claim*

12930 candidate(s) hold a complete record of 110 handcrafted scalars plus both embeddings; covered by 7 test module(s).

```json
{
  "handcrafted_feature_count": 110,
  "complete_multimodal_candidate_count": 12930,
  "audio_embedding_model": "openai/whisper-tiny.en",
  "text_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "tests": [
    "test_embeddings_audio.py",
    "test_embeddings_text.py",
    "test_features_acoustic.py",
    "test_features_assemble.py",
    "test_features_cli.py",
    "test_model_smoke.py",
    "test_training_dataset.py"
  ]
}
```

### A stratified, blind labelling round is built and servable: the queue reaches its target size and carries unmarked repeat presentations. - PASS

*implementation claim*

Queue full-v2 holds 2400 unique candidate(s) against a 2400 target, served as 2460 presentation(s) including 60 blind repeat(s); covered by 8 test module(s). This is the size of the queue, not a count of labels collected against it.

```json
{
  "queue_version": "full-v2",
  "target_unique": 2400,
  "unique_candidate_count": 2400,
  "presentation_count": 2460,
  "blind_repeat_presentations": 60,
  "by_split": {
    "test": 360,
    "train": 1680,
    "validation": 360
  },
  "candidate_manifest_hash": "bcb63eac450b7b635812cbcbbc9397c3eb6e2afd4092d7b2dc530987f88a14f5",
  "split_manifest_hash": "8db84a85e80f94f60dea25ad7180088cd3b54e9d9eab58478610180fc116b8e7",
  "human_labels_collected_against_it": 0,
  "tests": [
    "test_experiment_readiness.py",
    "test_labelling.py",
    "test_labelling_quality.py",
    "test_labelling_queue.py",
    "test_labelling_session.py",
    "test_orchestrate.py",
    "test_validate_and_stats.py",
    "test_weak_labels.py"
  ]
}
```

### Labels are frozen into an immutable, hashed snapshot that a rerun cannot silently rewrite. - PASS

*implementation claim*

ml/src/slotify_rank/experiment/freeze.py writes a hashed snapshot; the experiment declares snapshot version 'full-v2'; covered by 7 test module(s).

```json
{
  "module": "ml/src/slotify_rank/experiment/freeze.py",
  "declared_snapshot_version": "full-v2",
  "referenced_symbols": [
    "FileExistsError",
    "atomic_write_bytes",
    "sha256_text"
  ],
  "tests": [
    "test_experiment_canonical.py",
    "test_experiment_matrix.py",
    "test_experiment_readiness.py",
    "test_orchestrate.py",
    "test_splits.py",
    "test_validate_and_stats.py",
    "test_weak_labels.py"
  ]
}
```

### At least 2400 human-labelled candidates exist. - FAIL

*empirical claim*

0 human-labelled candidates, short of the 2400 the experiment requires. 13176 candidates have been GENERATED, which is not the same thing and is never counted as one.

```json
{
  "human_labelled_candidate_count": 0,
  "required": 2400,
  "generated_candidate_count": 13176
}
```

### The evaluated checkpoint was trained on human labels, not weak ones. - FAIL

*empirical claim*

Every training run records a non-human label source (weak_heuristic), so no checkpoint reflects human judgement.

```json
{
  "human_trained_run_count": 0,
  "latest_training_label_source": "weak_heuristic"
}
```

### A grouped, non-degraded train/validation/test split exists and its test groups appear in no other partition. - PASS

*implementation claim*

Split v4 groups by series with seed 42: 30 train / 7 validation / 3 test series, disjoint.

```json
{
  "split_version": "v4",
  "group_by": "series",
  "degraded": false,
  "series_by_split": {
    "test": 3,
    "train": 30,
    "validation": 7
  },
  "split_manifest_hash": "8db84a85e80f94f60dea25ad7180088cd3b54e9d9eab58478610180fc116b8e7"
}
```

### The experiment declares a full ablation matrix, every variant it names has a committed config and a registered implementation, and the matrix runner is tested. - PASS

*implementation claim*

5 variant(s) x 3 seed(s) = 15 cells; every variant has a config under ml/configs/models; the runner is ml/src/slotify_rank/experiment/matrix.py, covered by 2 test module(s). Declaring the matrix is not running it.

```json
{
  "ablations": [
    "handcrafted",
    "text_only",
    "audio_only",
    "concat",
    "gated"
  ],
  "seeds": [
    42,
    43,
    44
  ],
  "cell_count": 15,
  "headline_variant": "gated",
  "committed_model_configs": [
    "audio_only_v1",
    "audio_plus_acoustic_v1",
    "concat_v1",
    "gated_v1",
    "handcrafted_v1",
    "text_only_v1"
  ],
  "matrix_module": "ml/src/slotify_rank/experiment/matrix.py",
  "tests": [
    "test_experiment_matrix.py",
    "test_weak_labels.py"
  ]
}
```

### Checkpoints and seeds are selected on validation only, by a rule fixed before the test split is read. - PASS

*implementation claim*

Checkpoints are selected on validation_ndcg_at_3 and the reported seed is the median_validation_ndcg_at_3; both are declared in experiment-v2 and implemented in ml/src/slotify_rank/training/checkpoint.py, covered by 3 test module(s).

```json
{
  "checkpoint_selection_metric": "validation_ndcg_at_3",
  "seed_selection": "median_validation_ndcg_at_3",
  "module": "ml/src/slotify_rank/training/checkpoint.py",
  "latest_run_best_epoch": 1,
  "latest_run_selection_metric": "NOT MEASURED",
  "tests": [
    "test_inference.py",
    "test_training_prepare.py",
    "test_training_trainer.py"
  ]
}
```

### A scikit-learn classical baseline is implemented over the handcrafted features, tuned by group-aware cross-validation inside the training split. - PASS

*implementation claim*

ml/src/slotify_rank/baselines/classical.py fits a HistGradientBoostingRegressor selected by GroupKFold, covered by 1 test module(s). Whether it has been fitted on the canonical corpus is a separate, empirical question.

```json
{
  "module": "ml/src/slotify_rank/baselines/classical.py",
  "referenced_symbols": [
    "GroupKFold",
    "HistGradientBoostingRegressor"
  ],
  "declared_classical_baseline": "classical_handcrafted_v1",
  "tests": [
    "test_sklearn_integration.py"
  ]
}
```

### The headline metric is cross-checked against sklearn.metrics.ndcg_score by an independent implementation, and publication is blocked if they disagree. - PASS

*implementation claim*

ml/src/slotify_rank/evaluation/crosscheck.py calls sklearn.metrics.ndcg_score against this project's own NDCG, covered by 2 test module(s). The experiment requires the cross-check before a number is published: require_independent_metric_crosscheck=True.

```json
{
  "module": "ml/src/slotify_rank/evaluation/crosscheck.py",
  "referenced_symbols": [
    "ndcg_score"
  ],
  "required_by_experiment": true,
  "tests": [
    "test_evaluation_uncertainty.py",
    "test_sklearn_integration.py"
  ]
}
```

### Held-out uncertainty is reported as a percentile bootstrap resampled over episodes, with the resample count, confidence and seed fixed in advance. - PASS

*implementation claim*

ml/src/slotify_rank/evaluation/compare.py resamples episodes; the experiment fixes 2000 resamples at 0.95 confidence with seed 20260829, covered by 2 test module(s).

```json
{
  "module": "ml/src/slotify_rank/evaluation/compare.py",
  "bootstrap_resamples": 2000,
  "bootstrap_confidence": 0.95,
  "bootstrap_seed": 20260829,
  "referenced_symbols": [
    "Random"
  ],
  "tests": [
    "test_evaluation_compare.py",
    "test_evaluation_uncertainty.py"
  ]
}
```

### The experiment resolves to a manifest that pins the exact bytes it will be reported against: a config digest plus a hash per input artifact. - PASS

*implementation claim*

experiment-v2 resolves to config digest c1e09d5e34108330... and pins 18 input artifact(s): baseline_config, candidate_manifest, corpus_plan, episode_manifest, experiment_config, feature_manifest, labelling_queue, labelling_queue_config, model_config:audio_only, model_config:concat, model_config:gated, model_config:handcrafted, model_config:text_only, source_registry:sources_real_v1.yaml, source_registry:sources_v2.yaml, split_config, split_manifest, training_config.

```json
{
  "experiment_version": "experiment-v2",
  "config_digest": "c1e09d5e341083306a3edfdbca5e013f3d447602dc89119c9363373eeb33e99b",
  "pinned_artifacts": [
    "baseline_config",
    "candidate_manifest",
    "corpus_plan",
    "episode_manifest",
    "experiment_config",
    "feature_manifest",
    "labelling_queue",
    "labelling_queue_config",
    "model_config:audio_only",
    "model_config:concat",
    "model_config:gated",
    "model_config:handcrafted",
    "model_config:text_only",
    "source_registry:sources_real_v1.yaml",
    "source_registry:sources_v2.yaml",
    "split_config",
    "split_manifest",
    "training_config"
  ],
  "package_version": "0.2.0"
}
```

### The ranker is served through a TypeScript product: a React frontend and a Node API, each with a lint/typecheck/test command CI runs. - PASS

*implementation claim*

frontend react ^19.2.0 with scripts ['build', 'dev', 'lint', 'preview', 'test', 'verify']; backend scripts ['dev', 'start', 'test', 'typecheck', 'verify'].

```json
{
  "frontend_react_version": "^19.2.0",
  "frontend_scripts": [
    "build",
    "dev",
    "lint",
    "preview",
    "test",
    "verify"
  ],
  "backend_scripts": [
    "dev",
    "start",
    "test",
    "typecheck",
    "verify"
  ]
}
```

### The published improvement is quoted against the canonical frozen heuristic baseline. - NOT MEASURED

*empirical claim*

No publishable comparison exists, so no baseline was used.

```json
{
  "declared_baseline": "heuristic_offline_v1",
  "comparison_baseline": "NOT MEASURED",
  "baseline_config_hash": "eb23e3409c0c7516b257386f548b61bd71b70baca789728cd03773f6619f6178"
}
```

### The baseline's held-out NDCG@3 is measured. - NOT MEASURED

*empirical claim*

No publishable held-out comparison exists.

```json
{
  "baseline_ndcg_at_3": "NOT MEASURED",
  "evaluation_id": "NOT MEASURED"
}
```

### The learned model's held-out NDCG@3 is measured. - NOT MEASURED

*empirical claim*

No publishable held-out comparison exists.

```json
{
  "model_ndcg_at_3": "NOT MEASURED",
  "evaluation_id": "NOT MEASURED"
}
```

### The relative NDCG@3 improvement is measured. - NOT MEASURED

*empirical claim*

No publishable held-out comparison exists.

```json
{
  "relative_improvement_percent": "NOT MEASURED",
  "evaluation_id": "NOT MEASURED"
}
```

### The relative NDCG@3 improvement is at least 18.0 % over the heuristic baseline. - NOT MEASURED

*empirical claim*

No measured improvement, or no declared threshold, so the claim cannot be evaluated. It is NOT supported by default.

```json
{
  "measured_relative_improvement_percent": "NOT MEASURED",
  "required_relative_improvement_percent": 18.0,
  "bootstrap_interval": "NOT MEASURED"
}
```

### The AI/ML stack includes PyTorch. - PASS

*implementation claim*

Declared, imported by 14 module(s) (e.g. datasets/collate.py, datasets/ranking_dataset.py, embeddings/device.py), and a training run recorded a torch version and a parameter count.

```json
{
  "declared_in_pyproject": true,
  "importing_modules": [
    "datasets/collate.py",
    "datasets/ranking_dataset.py",
    "embeddings/device.py",
    "embeddings/whisper_audio.py",
    "inference/predictor.py",
    "models/base.py",
    "models/variants.py",
    "ranking/losses.py"
  ],
  "importing_module_count": 14,
  "training_torch_version": "2.13.0",
  "model_parameter_count": 489477
}
```

### The AI/ML stack includes Hugging Face `transformers`. - PASS

*implementation claim*

Declared, imported by 2 module(s) (e.g. embeddings/whisper_audio.py, transcription/local_whisper.py), and the embedding statistics name Hugging Face model ids for both the speech and the transcript encoder.

```json
{
  "declared_in_pyproject": true,
  "importing_modules": [
    "embeddings/whisper_audio.py",
    "transcription/local_whisper.py"
  ],
  "importing_module_count": 2,
  "audio_embedding_model": "openai/whisper-tiny.en",
  "text_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "feature_pipeline_transformers_version": "4.57.6"
}
```

### The AI/ML stack includes librosa. - PASS

*implementation claim*

Declared, imported by 2 module(s) (e.g. features/acoustic.py, transcription/local_whisper.py), and the feature statistics record a librosa version and the handcrafted vocabulary contains librosa-derived spectral descriptors.

```json
{
  "declared_in_pyproject": true,
  "importing_modules": [
    "features/acoustic.py",
    "transcription/local_whisper.py"
  ],
  "importing_module_count": 2,
  "feature_pipeline_librosa_version": "0.11.0",
  "librosa_derived_feature_count": 24,
  "example_features": [
    "onset_strength_max_around_context",
    "onset_strength_max_around_medium",
    "onset_strength_max_around_short",
    "onset_strength_mean_around_context"
  ]
}
```

### scikit-learn ran inside a published comparison: an independent NDCG cross-check and a fitted classical baseline. - NOT MEASURED

*empirical claim*

Declared and imported by 2 module(s), but no artifact yet shows it running: a published comparison records BOTH an independent scikit-learn NDCG cross-check and a fitted scikit-learn classical baseline.

```json
{
  "declared_in_pyproject": true,
  "importing_modules": [
    "baselines/classical.py",
    "evaluation/crosscheck.py"
  ],
  "importing_module_count": 2,
  "metric_crosscheck_available": "NOT MEASURED",
  "metric_crosscheck_agrees": "NOT MEASURED",
  "metric_crosscheck_sklearn_version": "NOT MEASURED",
  "classical_baseline_available": "NOT MEASURED",
  "classical_baseline_ndcg_at_3": "NOT MEASURED"
}
```

### The award is described precisely as 'UofTHacks 13 — MLH Best Use of ElevenLabs', with no implied overall win. - PASS

*implementation claim*

README.md states 'UofTHacks 13 — MLH Best Use of ElevenLabs' and makes no stronger claim.

```json
{
  "exact_wording_present": true,
  "overstated_patterns_matched": []
}
```

## Artifacts read

- `candidate_statistics`: `artifacts/dataset/candidate_statistics.json`
- `dataset_statistics`: `artifacts/dataset/dataset_statistics.json`
- `embedding_statistics`: `artifacts/features/embedding_statistics.json`
- `experiment_config`: `ml/configs/experiment_v2.yaml`
- `experiment_manifest`: `artifacts/experiments/experiment-v2.json`
- `feature_statistics`: `artifacts/features/feature_statistics.json`
- `label_statistics`: `artifacts/dataset/label_statistics.json`
- `labelling_queue_summary`: `artifacts/labelling/queue_summary.json`
- `latest_training_run`: `artifacts/training/gated-d8ed976101aa4c3b/training_summary.json`
- `readiness_report`: `artifacts/experiments/readiness_report.json`
