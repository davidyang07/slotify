# Training run `concat-3a9f3c90166e0027`

> **SYNTHETIC SMOKE RUN. This model was trained on generated fixtures, not on real audio or human labels. The metrics below demonstrate that the training system works end to end. They are NOT evidence of model quality and must never be quoted as a result.**

- **Evidence class**: synthetic smoke run - not a model-quality result
- **Data provenance**: synthetic_fixture (label source: synthetic)
- **Generated**: 2026-07-22T23:43:54+00:00

## Model

| Field | Value |
| --- | --- |
| Variant | `concat` |
| Experiment | `concat` |
| Trainable parameters | 447,490 |

## Dataset

| Field | Value |
| --- | --- |
| Training episodes | 6 |
| Training candidates | 48 |
| Training pairs | 133 |
| Validation episodes | 2 |
| Validation candidates | 16 |

## Result

| Metric | Value |
| --- | --- |
| Best epoch | 1 |
| Epochs run | 9 |
| Validation NDCG@3 | 1.0000 |
| Validation pairwise accuracy | 0.9565 |
| Validation classification F1 | 0.8966 |
| Runtime (s) | 7.4360 |
| Device | `cpu` (float32) |
| Stop reason | validation metric did not improve by more than 0.0001 for 8 epoch(s); best was 1.0 at epoch 1 |

## Resolved overrides

| Setting | From | To |
| --- | --- | --- |
| `epochs` | 30 | 10 |

## Artifacts

- `best_checkpoint`
- `dataset_summary`
- `directory`
- `epoch_metrics`
- `last_checkpoint`
- `normalizer`
- `resolved_config`
