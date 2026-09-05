# Training run `gated-d8ed976101aa4c3b`

> **WEAKLY SUPERVISED BOOTSTRAP RUN. The targets were derived from heuristic_offline_v1's own score, not from human judgement, so this model is a distillation of the baseline. Its validation NDCG measures how well it reproduces its teacher and is NOT a ranking-quality result. It must never be compared against heuristic_offline_v1, because that baseline IS its teacher. Its purpose is to prove the inference path end to end.**

- **Evidence class**: weakly supervised bootstrap - distills heuristic_offline_v1; not a model-quality result and never comparable against that baseline
- **Data provenance**: weak_supervision (label source: weak_heuristic)
- **Generated**: 2026-08-18T07:13:01+00:00

## Model

| Field | Value |
| --- | --- |
| Variant | `gated` |
| Experiment | `gated` |
| Trainable parameters | 489,477 |

## Dataset

| Field | Value |
| --- | --- |
| Training episodes | 11 |
| Training candidates | 260 |
| Training pairs | 271 |
| Validation episodes | 1 |
| Validation candidates | 67 |

## Result

| Metric | Value |
| --- | --- |
| Best epoch | 1 |
| Epochs run | 9 |
| Validation NDCG@3 | 1.0000 |
| Validation pairwise accuracy | 0.9015 |
| Validation classification F1 | 0.3125 |
| Runtime (s) | 7.5740 |
| Device | `cpu` (float32) |
| Stop reason | validation metric did not improve by more than 0.0001 for 8 epoch(s); best was 1.0 at epoch 1 |

## Resolved overrides

None; the configuration file was used unchanged.

## Artifacts

- `best_checkpoint`
- `dataset_summary`
- `directory`
- `epoch_metrics`
- `last_checkpoint`
- `normalizer`
- `resolved_config`
