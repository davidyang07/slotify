# Resume evidence report

**Generated file - do not edit.** Produced by `slotify-rank report resume-evidence`, which reads the artifacts named at the bottom. Every number here comes from one of them.

- Git SHA: `1a25a798189c7719ef38173872d6451b428fe9db`
- Generated: 2026-08-18T08:05:45+00:00
- Package version: 0.2.0

`NOT YET SUPPORTED` means no artifact produced that value. It is not a zero: a measured zero (for example, zero human labels) is printed as `0`.

## Claim verdicts

### Claim A - SUPPORTED

> Built a multimodal PyTorch ranker using audio and transcript features to detect natural podcast ad breaks.

- A gated PyTorch ranker with 489477 parameters consumes openai/whisper-tiny.en speech representations and sentence-transformers/all-MiniLM-L6-v2 transcript embeddings, and is served through the product's inference path.
- The served checkpoint was trained with label_source=weak_heuristic, so it demonstrates the architecture and the serving path, not ranking quality.

### Claim B - NOT YET SUPPORTED

> Trained on 2,400+ labelled candidates.

- human_labelled_candidate_count = 0 (measured from the label database, not estimated).
- generated_candidate_count = 595 -- candidates produced by the generators, which are NOT labels.
- No human label exists, so no count of labelled candidates can be quoted.

### Claim C - NOT YET SUPPORTED

> Improving NDCG@3 by 18% over the heuristic baseline.

- No publishable held-out comparison exists, so no NDCG@3 improvement can be quoted.
- 1 comparison(s) ran but were blocked; see measurements.blocked_comparisons for the measured values and the reasons they may not be published.

## Measured quantities

| Quantity | Value |
| --- | --- |
| Human labelled candidates | 0 |
| Human annotators | 0 |
| Human labelled audio hours | 0.0 |
| Generated candidates (NOT labels) | 595 |
| Complete multimodal feature records | 583 |
| Processed audio hours | 0.8387 |
| Processed episodes | 13 |
| Held-out evaluation candidates | 0 |
| Phase 5B experiment gate ready | False |

## Model

| Field | Value |
| --- | --- |
| Variant | gated |
| Parameters | 489477 |
| Run id | gated-d8ed976101aa4c3b |
| Trained on label source | weak_heuristic |
| Training data provenance | weak_supervision |
| Audio encoder | openai/whisper-tiny.en |
| Text encoder | sentence-transformers/all-MiniLM-L6-v2 |

## Held-out comparison

| Field | Value |
| --- | --- |
| Evaluation id | NOT YET SUPPORTED |
| Baseline NDCG@3 | NOT YET SUPPORTED |
| Model NDCG@3 | NOT YET SUPPORTED |
| Relative improvement | NOT YET SUPPORTED |

### Comparisons that ran but may not be published

These measured real numbers on real artifacts. They are shown so the pipeline is auditable, and they are **not** the claim.

- `eval-5fdb9c01c499f43f`: baseline 1.0, model 1.0, relative 0.0 %
  - blocked: ground truth is label_source='weak_heuristic', not human. A comparison against heuristic_offline_v1 scored with heuristic-derived labels is circular: the baseline is the teacher.
  - blocked: the model was trained with label_source='weak_heuristic', so its ordering reflects its teacher rather than human judgement.

## Artifacts read

- `candidate_statistics`: `artifacts/dataset/candidate_statistics.json`
- `dataset_statistics`: `artifacts/dataset/dataset_statistics.json`
- `embedding_statistics`: `artifacts/features/embedding_statistics.json`
- `feature_statistics`: `artifacts/features/feature_statistics.json`
- `label_statistics`: `artifacts/dataset/label_statistics.json`
- `readiness_report`: `artifacts/experiments/readiness_report.json`
- `split_statistics`: `artifacts/dataset/split_statistics.json`
- `training_summary`: `artifacts/training/gated-d8ed976101aa4c3b/training_summary.json`
