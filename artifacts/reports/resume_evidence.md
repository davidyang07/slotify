# Resume evidence

**Generated file - do not edit.** Produced by `slotify-rank report resume-evidence` (`npm run resume-evidence`), which reads the artifacts named at the bottom. Every number here comes from one of them; the thresholds come from the committed experiment definition.

- Git SHA: `fa0c7b349456c335861f813b2d95c0b698819744`
- Generated: 2026-08-29T15:11:15+00:00
- Package version: 0.2.0
- Verdict: **6 PASS / 2 FAIL / 8 NOT MEASURED**

`NOT MEASURED` means no artifact produced that value. It is not a failure and it is not a zero: a claim nobody has measured and a claim measured and found short are different facts.

## Checklist

| Claim | Status | Evidence |
| --- | --- | --- |
| A committed, versioned experiment definition fixes the protocol in advance. | **PASS** | experiment-resume-v1 declares the split, the seeds, the headline variant, the canonical baseline and the claim thresholds this report is checked against. |
| A multimodal PyTorch ranker exists and is trained. | **PASS** | gated with 489477 parameters, trained on label_source=weak_heuristic. |
| The ranker consumes both audio and transcript features. | **PASS** | Speech representations from openai/whisper-tiny.en; transcript embeddings from sentence-transformers/all-MiniLM-L6-v2; plus 110 handcrafted acoustic and structural scalars. |
| At least 2400 human-labelled candidates exist. | **FAIL** | 0 human-labelled candidates, short of the 2400 the experiment requires. 595 candidates have been GENERATED, which is not the same thing and is never counted as one. |
| The evaluated checkpoint was trained on human labels, not weak ones. | **FAIL** | Every training run records a non-human label source (weak_heuristic), so no checkpoint reflects human judgement. |
| A grouped, non-degraded train/validation/test split exists and its test groups appear in no other partition. | **NOT MEASURED** | No experiment manifest has been resolved. |
| The published improvement is quoted against the canonical frozen heuristic baseline. | **NOT MEASURED** | No publishable comparison exists, so no baseline was used. |
| The baseline's held-out NDCG@3 is measured. | **NOT MEASURED** | No publishable held-out comparison exists. |
| The learned model's held-out NDCG@3 is measured. | **NOT MEASURED** | No publishable held-out comparison exists. |
| The relative NDCG@3 improvement is measured. | **NOT MEASURED** | No publishable held-out comparison exists. |
| The relative NDCG@3 improvement is at least 18.0 % over the heuristic baseline. | **NOT MEASURED** | No measured improvement, or no declared threshold, so the claim cannot be evaluated. It is NOT supported by default. |
| The AI/ML stack includes PyTorch. | **PASS** | Declared, imported by 14 module(s) (e.g. datasets/collate.py, datasets/ranking_dataset.py, embeddings/device.py), and a training run recorded a torch version and a parameter count. |
| The AI/ML stack includes Hugging Face `transformers`. | **PASS** | Declared, imported by 2 module(s) (e.g. embeddings/whisper_audio.py, transcription/local_whisper.py), and the embedding statistics name Hugging Face model ids for both the speech and the transcript encoder. |
| The AI/ML stack includes librosa. | **NOT MEASURED** | Declared and imported by 2 module(s), but no artifact yet shows it running: the feature statistics record a librosa version and the handcrafted vocabulary contains librosa-derived spectral descriptors. |
| The AI/ML stack includes scikit-learn, used meaningfully. | **NOT MEASURED** | Declared and imported by 2 module(s), but no artifact yet shows it running: a published comparison records BOTH an independent scikit-learn NDCG cross-check and a fitted scikit-learn classical baseline. |
| The award is described precisely as 'UofTHacks 13 — MLH Best Use of ElevenLabs', with no implied overall win. | **PASS** | README.md states 'UofTHacks 13 — MLH Best Use of ElevenLabs' and makes no stronger claim. |

## Measured quantities

| Quantity | Value |
| --- | --- |
| Human-labelled candidates | 0 |
| ...required by the experiment | 2400 |
| Generated candidates (NOT labels) | 595 |
| Complete multimodal feature records | 583 |
| Processed audio hours | 0.8387 |
| Episodes | 13 |
| Series | 10 |
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

experiment-resume-v1 declares the split, the seeds, the headline variant, the canonical baseline and the claim thresholds this report is checked against.

```json
{
  "experiment_version": "experiment-resume-v1",
  "split_version": "v3",
  "seeds": [
    42,
    43,
    44
  ],
  "headline_variant": "gated"
}
```

### A multimodal PyTorch ranker exists and is trained. - PASS

gated with 489477 parameters, trained on label_source=weak_heuristic.

```json
{
  "model_variant": "gated",
  "model_parameter_count": 489477,
  "training_label_source": "weak_heuristic"
}
```

### The ranker consumes both audio and transcript features. - PASS

Speech representations from openai/whisper-tiny.en; transcript embeddings from sentence-transformers/all-MiniLM-L6-v2; plus 110 handcrafted acoustic and structural scalars.

```json
{
  "audio_embedding_model": "openai/whisper-tiny.en",
  "text_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "handcrafted_feature_count": 110,
  "complete_multimodal_candidate_count": 583
}
```

### At least 2400 human-labelled candidates exist. - FAIL

0 human-labelled candidates, short of the 2400 the experiment requires. 595 candidates have been GENERATED, which is not the same thing and is never counted as one.

```json
{
  "human_labelled_candidate_count": 0,
  "required": 2400,
  "generated_candidate_count": 595
}
```

### The evaluated checkpoint was trained on human labels, not weak ones. - FAIL

Every training run records a non-human label source (weak_heuristic), so no checkpoint reflects human judgement.

```json
{
  "human_trained_run_count": 0,
  "latest_training_label_source": "weak_heuristic"
}
```

### A grouped, non-degraded train/validation/test split exists and its test groups appear in no other partition. - NOT MEASURED

No experiment manifest has been resolved.

```json
{
  "split_version": "NOT MEASURED",
  "group_by": "NOT MEASURED",
  "degraded": "NOT MEASURED",
  "series_by_split": {},
  "split_manifest_hash": "NOT MEASURED"
}
```

### The published improvement is quoted against the canonical frozen heuristic baseline. - NOT MEASURED

No publishable comparison exists, so no baseline was used.

```json
{
  "declared_baseline": "heuristic_offline_v1",
  "comparison_baseline": "NOT MEASURED",
  "baseline_config_hash": "NOT MEASURED"
}
```

### The baseline's held-out NDCG@3 is measured. - NOT MEASURED

No publishable held-out comparison exists.

```json
{
  "baseline_ndcg_at_3": "NOT MEASURED",
  "evaluation_id": "NOT MEASURED"
}
```

### The learned model's held-out NDCG@3 is measured. - NOT MEASURED

No publishable held-out comparison exists.

```json
{
  "model_ndcg_at_3": "NOT MEASURED",
  "evaluation_id": "NOT MEASURED"
}
```

### The relative NDCG@3 improvement is measured. - NOT MEASURED

No publishable held-out comparison exists.

```json
{
  "relative_improvement_percent": "NOT MEASURED",
  "evaluation_id": "NOT MEASURED"
}
```

### The relative NDCG@3 improvement is at least 18.0 % over the heuristic baseline. - NOT MEASURED

No measured improvement, or no declared threshold, so the claim cannot be evaluated. It is NOT supported by default.

```json
{
  "measured_relative_improvement_percent": "NOT MEASURED",
  "required_relative_improvement_percent": 18.0,
  "bootstrap_interval": "NOT MEASURED"
}
```

### The AI/ML stack includes PyTorch. - PASS

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
  "feature_pipeline_transformers_version": "NOT MEASURED"
}
```

### The AI/ML stack includes librosa. - NOT MEASURED

Declared and imported by 2 module(s), but no artifact yet shows it running: the feature statistics record a librosa version and the handcrafted vocabulary contains librosa-derived spectral descriptors.

```json
{
  "declared_in_pyproject": true,
  "importing_modules": [
    "features/acoustic.py",
    "transcription/local_whisper.py"
  ],
  "importing_module_count": 2,
  "feature_pipeline_librosa_version": "NOT MEASURED",
  "librosa_derived_feature_count": 24,
  "example_features": [
    "onset_strength_max_around_context",
    "onset_strength_max_around_medium",
    "onset_strength_max_around_short",
    "onset_strength_mean_around_context"
  ]
}
```

### The AI/ML stack includes scikit-learn, used meaningfully. - NOT MEASURED

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
- `experiment_config`: `ml/configs/experiment_resume_v1.yaml`
- `feature_statistics`: `artifacts/features/feature_statistics.json`
- `label_statistics`: `artifacts/dataset/label_statistics.json`
- `latest_training_run`: `artifacts/training/gated-d8ed976101aa4c3b/training_summary.json`
- `readiness_report`: `artifacts/experiments/readiness_report.json`

## Artifacts missing

- `artifacts/experiments/experiment-*.json`
