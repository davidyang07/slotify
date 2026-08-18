# Evaluation eval-5fdb9c01c499f43f

- **Generated**: 2026-08-18T07:31:42+00:00
- **Git SHA**: 988aeb427de7e8d5ffe01215d8e94cad5d564f15
- **Split**: test (v2)
- **Episodes**: 1  **Candidates**: 268  **Relevant**: 107
- **Ground truth**: label_source=weak_heuristic
- **Model**: gated (gated-d8ed976101aa4c3b), trained on label_source=weak_heuristic
- **Baseline**: heuristic_offline_v1

## Headline - NOT PUBLISHABLE

This comparison ran, but the number below **must not be quoted**:

- ground truth is label_source='weak_heuristic', not human. A comparison against heuristic_offline_v1 scored with heuristic-derived labels is circular: the baseline is the teacher.
- the model was trained with label_source='weak_heuristic', so its ordering reflects its teacher rather than human judgement.

- Baseline NDCG@3: 1.0000
- Model NDCG@3: 1.0000
- Relative improvement: 0.00 %

Computed as `100 * (model - baseline) / baseline` in `slotify_rank.evaluation.compare.relative_improvement_percent`.

## All cutoffs

| Metric | Baseline | Model |
| --- | --- | --- |
| ndcg_at_k (at_1) | 1.0000 | 1.0000 |
| precision_at_k (at_1) | 1.0000 | 1.0000 |
| recall_at_k (at_1) | 1.0000 | 1.0000 |
| mrr (at_1) | 1.0000 | 1.0000 |
| ndcg_at_k (at_3) | 1.0000 | 1.0000 |
| precision_at_k (at_3) | 1.0000 | 1.0000 |
| recall_at_k (at_3) | 1.0000 | 1.0000 |
| mrr (at_3) | 1.0000 | 1.0000 |
| ndcg_at_k (at_5) | 1.0000 | 0.8521 |
| precision_at_k (at_5) | 1.0000 | 1.0000 |
| recall_at_k (at_5) | 1.0000 | 1.0000 |
| mrr (at_5) | 1.0000 | 1.0000 |

## Caveats

- only 1 evaluation episode(s); metrics are macro-averaged over episodes, so this estimate has very wide uncertainty.
- a single training run was evaluated; no seed-to-seed variance is reported.
