# Resume claim-to-evidence matrix

Every claim below is **unverified until its generating command succeeds and its artifact exists in
this repository**. No number in this file may be typed by hand — each is read from the artifact named
in its row, which is produced by `slotify-rank report`.

Status vocabulary: `not started` → `in progress` → `evidence generated` → `verified`
(`verified` = artifact exists, was produced by the current `git_sha`, and the plan's acceptance
criterion for that phase passed).

Phase references point at `docs/multimodal-ranking-mvp-plan.md` §36–37.

---

## Phase 1 — foundation (complete)

| Intended claim | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| A deterministic, reproducible signal-based baseline exists and is the comparison denominator | TypeScript↔Python parity on golden fixtures covering 11 scenarios; every score component serialized; config version stamped into every record | `cd backend; npx tsx scripts/dump-heuristic-golden.ts --out ..\ml\tests\fixtures\heuristic_golden.json` then `cd ml; .\.venv\Scripts\python.exe -m pytest` | `ml/tests/fixtures/heuristic_golden.json`, `config/heuristic_offline_v1.json`, 169 passing tests | **verified** (2026-07-22) |
| The refactor did not change product behaviour | SHA-256 of the golden output identical before and after hoisting constants; `analyze_cli` output identical on real audio; `npm run typecheck` clean | see above | verification log in this session | **verified** (2026-07-22) |

Baseline naming, used consistently in every later report:

| Name | What it is | Needs credentials? |
|---|---|---|
| `heuristic_offline_v1` | **canonical baseline / headline denominator** | no |
| `product_api_baseline_v1` | same path plus paid `whisper-1` + `gpt-4o-mini` | yes — reported separately, never required |

---

## Bullet 1 — the model

> Trained a multimodal PyTorch ranker combining waveform features, speech representations, and
> transcript embeddings to identify natural podcast ad breaks, outperforming a signal-based baseline
> by **X %** NDCG@3.

| Intended claim | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| Multimodal PyTorch ranker exists and is trained | Architecture definition, trained checkpoint, parameter count, per-modality gate values | `slotify-rank train --config configs/train_gated_v1.yaml` | `artifacts/models/{run_id}/best.pt`, `artifacts/models/model_card.md` | not started |
| Three modalities genuinely contribute | 6-model comparison + 6 required ablations × 3 seeds, mean ± std | `slotify-rank ablate --config configs/ablations.yaml --seeds 3` | `artifacts/evaluation/ablation_results.csv` | not started |
| Signal-based baseline is the real production heuristic | Byte-exact parity between the Python port and `backend/src/lib/candidates.ts` | `pytest ml/tests/unit/test_heuristic_parity.py` | `ml/tests/fixtures/heuristic_golden.json` | not started |
| **X % NDCG@3 improvement** | Baseline + model NDCG@3 on held-out **human-labelled** test episodes, with bootstrap 95 % CI over episodes | `slotify-rank evaluate --config configs/eval_v1.yaml` then `slotify-rank report` | `artifacts/evaluation/baseline_results.json`, `artifacts/evaluation/model_results.json`, `artifacts/reports/final_results.md` | not started |
| Supporting metrics | P@3, R@3, binary F1, MRR, pairwise accuracy, per-episode table, mean + p95 latency | same as above | `artifacts/evaluation/per_episode_results.csv` | not started |

`X` is computed only by:

```
ndcg_improvement_pct = 100 * (model_ndcg_at_3 - baseline_ndcg_at_3) / baseline_ndcg_at_3
```

in `ml/src/slotify_rank/eval/report.py`. **If the CI on `X` includes zero, the bullet must say so or
be dropped.**

---

## Bullet 2 — the dataset and evaluation pipeline

> Built a dataset and evaluation pipeline spanning **`processed_audio_hours`** hours of audio and
> **`generated_candidate_count`** candidate breakpoints, evaluated against a held-out
> human-labelled subset, achieving **Y %** agreement with human-preferred placements and reducing
> manual editing time by **Z %**.

The wording must make clear that the large corpus is *processed*, not *hand-labelled*. Phrasings such
as "10,000+ hand-labelled breakpoints" are prohibited: the human-labelled subset is ≈1 500 candidates
over ≈8–12 hours, and that is the number tied to any labelling claim.

### The five dataset quantities, tracked and reported separately

Never collapsed into a single "labelled dataset" figure. Each has its own field in the statistics
artifacts and its own row below.

| Field | Meaning | Target |
|---|---|---|
| `processed_audio_hours` | audio decoded, transcribed and featurised | ≥ 50 |
| `generated_candidate_count` | candidates produced by the generators | ≥ 10 000 |
| `human_labelled_candidate_count` | reviewed by a person against the 1–5 rubric | ≈ 1 500 |
| `human_labelled_audio_hours` | audio duration of the episodes containing those labels | ≈ 8–12 |
| `held_out_evaluation_candidate_count` | human-labelled **and** in the test split — the only source of final test metrics | reported as measured |

| Intended claim | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| **`processed_audio_hours`** (target 50+) | Measured total duration; episode count; breakdown by source, licence and content type; per-split duration | `slotify-rank dataset-stats` | `artifacts/dataset/dataset_statistics.json` | not started |
| **`generated_candidate_count`** (target 10 000+) | Measured candidate count; counts by generator source; **separate** totals for human / weak / unlabelled | `slotify-rank candidate-stats` | `artifacts/dataset/candidate_statistics.json` | not started |
| **`human_labelled_candidate_count`** (target ≈1 500) and **`human_labelled_audio_hours`** (≈8–12) | Label ledger by `label_source` and annotator hash; staged at 200–300 → 750 → 1 500 | `slotify-rank dataset-stats` | `artifacts/dataset/dataset_statistics.json` | not started |
| **`held_out_evaluation_candidate_count`** | Human-labelled candidates in the test split; test split restricted to podcast-like content (no AMI) | `slotify-rank validate` + `slotify-rank dataset-stats` | `artifacts/dataset/split_statistics.json` | not started |
| Splits are leak-free | Episode-level (series-aware) splits; assertions that no candidate, episode, or series spans splits; no non-human label used as test ground truth | `slotify-rank split --group-by series` then `slotify-rank validate` | `artifacts/dataset/split_statistics.json` | not started |
| **Y % human agreement** | Blind, randomized-order evaluation on held-out test episodes; **≥1 non-author evaluator required, 2 preferred**; A1 (primary), A2, A3 with `n`, tie rate, `n_evaluators` and inter-rater κ | `slotify-rank human-eval --split test --mode blind-pairwise` then `slotify-rank human-eval-report` | `artifacts/evaluation/human_preference_results.json` | not started |
| **Z % editing-time reduction** | Counterbalanced within-subject timed study, manual vs assisted, **≥3 participants (5 targeted)**, raw per-session rows + median/mean with bootstrap CI. Reported as measured — the ~80 % figure is a prior expectation, not a target to engineer toward. | `slotify-rank benchmark-run` then `slotify-rank benchmark-report` | `artifacts/benchmarks/editing_time_raw.csv`, `artifacts/benchmarks/editing_time_results.csv` | not started |
| FastAPI integration works end to end | Upload → rank → preview → export in the real product; legacy fallback path still matches the Phase-1 golden | `slotify-rank serve` + `pytest ml/tests/integration -m e2e` | integration test report, `docs/architecture.md` | not started |

### The definition of `Y` (must be printed next to the number, everywhere)

**A1 — top-3 acceptability hit rate (primary).** Over held-out test episodes, the fraction of episodes
in which at least one of the model's top-3 predicted breakpoints matches — within a **±3 s** tolerance
— a candidate that a human annotator scored **≥ 4** on the 1–5 naturalness rubric.

Also reported, never conflated with A1:

- **A2 — top-1 match rate:** the model's rank-1 prediction lies within tolerance of the episode's
  highest human-scored candidate.
- **A3 — blind pairwise win rate vs the heuristic:** model win rate / heuristic win rate / tie rate,
  systems hidden, order randomized, with `n` and Cohen's κ across evaluators.

### The definition of `Z`

`100 * (manual_time - assisted_time) / manual_time`, per matched (participant, episode) pair, reported
as median and mean with a bootstrap 95 % CI and the participant/episode counts stated.

---

## Non-negotiables

1. **No number is hand-copied.** `artifacts/reports/final_results.md` is generated from the JSON
   artifacts, and a test asserts the two agree.
2. **Weak labels are never called human labels.** `label_source ∈ {human, weak_heuristic,
   metadata_derived, unlabelled}` is carried on every candidate; `validate.py` hard-fails if a test-set
   ground-truth label is not `human`.
3. **Human preference is never computed from model or heuristic output.** `human-eval` refuses to run
   outside the test split and requires human labels.
4. **Targets are not results.** 50 hours, 10 000 candidates, and ~80 % are *design targets*. The
   resume quotes the measured values from the artifacts above, whatever they turn out to be.
5. **No row is marked `verified` while its command has never been run on the current commit.**
6. **The headline denominator is credential-free.** `heuristic_offline_v1` needs no API key, no
   network and no GPU, so the NDCG@3 comparison is reproducible by anyone who clones the repo.
   `product_api_baseline_v1` may be reported alongside but never replaces it.
7. **Evaluation never reads the frontend.** The fabricated fallback slots in
   `frontend/src/App.tsx:453-482` (≈22/48/72 % with hard-coded confidences) are product technical
   debt and are excluded from every metric by construction — `ml/` computes rankings independently.
   Synthetic padding produced by the *baseline itself* is flagged `is_synthetic` and carries
   `rank: null`, so it cannot enter a ranking either.
