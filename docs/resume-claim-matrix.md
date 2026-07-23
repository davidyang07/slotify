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

## Phase 2 — dataset foundation (workflow complete, corpus not acquired)

The commands, schemas, splits, labelling loop, validation and statistics all exist and are tested.
**No dataset target has been reached.** The pipeline has been exercised only on the repository's
~4.6-minute smoke fixtures, and every measured number below is that smoke run.

| Intended claim | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| A reproducible dataset pipeline exists | Import → probe → normalize → generate → split → validate → stats runs end to end with non-zero exit codes on failure; 379 passing tests | see `ml/README.md` § Dataset workflow | `artifacts/dataset/*.json`, `artifacts/dataset/dataset_summary.md` | **verified** (2026-07-22) |
| Dataset integrity is checked, not assumed | 26 named checks including duplicate IDs, checksum mismatch, out-of-bounds candidates, split leakage by episode **and** by series, synthetic-eligibility violations, and a test partition with no target-domain audio — each with a test that makes it fail | `dataset validate --deep` | `artifacts/dataset/validation_report.json` | **verified** (2026-07-22) |
| Human labels are collected and exportable | Local FastAPI UI, SQLite persistence, resumable per-annotator sessions, update-in-place, versioned JSONL export with rubric and threshold metadata | `label serve` then `label export` | `data/labels/labels_v1.jsonl` (+ `.meta.json`) | **implemented, not yet populated** |

Status vocabulary for the dataset rows below:
`implemented but not yet populated` → `partially populated using smoke data` → `fully supported by
real measured data`.

## Phase 3 — multimodal feature pipeline (workflow complete, corpus not acquired)

The pipeline exists, is tested, and has been run end to end with the **real local
models**. As with Phase 2, it has been exercised only on the repository's
~2.2-minute smoke fixtures. Every number below is that smoke run and is labelled
**partially populated using smoke data** — none of it supports a claim about
model quality, and Phase 3 trains nothing.

| Intended claim | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| A reproducible, resumable, CPU-first feature pipeline exists | Transcribe → acoustic → audio embed → text embed → assemble → validate → stats, non-zero exit on integrity failure; 607 passing tests, 89% coverage | `pipeline features --deep` | `artifacts/features/*.json`, `feature_summary.md` | **verified** (2026-07-22) |
| Expensive outputs are cached and interrupted work resumes | Second identical run: 0 recomputed, 4/4 model stages cache-hit, 102 s → 1.2 s; `partial` ledger entries force reprocessing | `pipeline features` twice | `data/features/state/*.json` | **verified** (2026-07-22) |
| Stale artifacts are detected, not silently reused | Bumping `FEATURE_SPEC_VERSION` recomputed the acoustic and text stages and correctly left transcription and audio embeddings cached | see `docs/feature-pipeline.md` §7 | ledger digests | **verified** (2026-07-22) |
| Dimensions are what the models actually produce | Opt-in `model_smoke` tests assert 384 (not 768) tiny.en, 20 ms/frame derived and cross-checked, 384 native MiniLM, 1536 constructed | `pytest -m model_smoke` | 8 passing model-smoke tests | **verified** (2026-07-22) |

### Phase 3 measured quantities

Read from `artifacts/features/feature_statistics.json`. **Smoke data only.**

| Field | Value | Note |
|---|---|---|
| `implemented_feature_family` | acoustic, structural, transcript-scalar, Whisper speech representation, MiniLM transcript embedding | 5 families |
| `processed_candidate_count` | 20 | eligible, non-synthetic |
| `successfully_embedded_candidate_count` | 18 | `feature_status = complete` |
| `missing_feature_candidate_count` | 2 | `audio_only`; no usable transcript context |
| `transcribed_audio_hours` | 0.0351 | speech covered by segments, **not** episode duration |
| `audio_embedding_model` | `openai/whisper-tiny.en` | frozen encoder, 384-d |
| `text_embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | frozen, 384-d native |
| `feature_pipeline_version` | `featurepipeline-v1.0.0` | feature spec `featurespec-v1.1.0` |
| handcrafted feature count | 110 | raw, unnormalized, each with a missing mask |
| constructed text vector | 1536 | 384 × 4 blocks — arithmetic, not a model width |

**Evidence classification: `partially populated using smoke data`.**

What this does *not* yet support: any statement about ranking quality, any
comparison against `heuristic_offline_v1`, and any claim resting on corpus size.
7 episodes totalling ~2.2 minutes is a plumbing test, not a dataset.

---

## Phase 4 — PyTorch ranking training system (machinery complete, real corpus not acquired)

The training system exists, is tested, and has been run end to end on CPU. It
loads Phase 3 features **without recomputing any transcript or embedding**,
validates training eligibility, fits train-only scalar normalization, generates
within-episode ranking pairs, trains five model variants through one shared
interface, selects checkpoints on validation NDCG@3, and resumes interrupted
runs. **It has been exercised only on deterministic synthetic fixtures** (see
below); every training metric here is a synthetic smoke result and supports no
claim about model quality.

| Intended claim | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| A genuinely trainable multimodal PyTorch ranker exists | Five variants behind one interface; each forward pass runs on the real Phase 3 records; parameter counts reported; all under 1M params | `slotify-rank models describe --model gated` | `artifacts/training/<run_id>/training_summary.json` | **verified** (2026-07-22, synthetic smoke) |
| Training is reproducible and CPU-first | Deterministic seeds; content-addressed run id; device/dtype recorded; resume equivalent to uninterrupted within 1e-4; runs on CPU with no GPU | `slotify-rank training run --model-config ml/configs/models/gated_v1.yaml` | `artifacts/training/<run_id>/resolved_config.json`, `environment.json` | **verified** (2026-07-22, synthetic smoke) |
| Phase 3 features load without recomputation | Loader reads cached `.npy` arrays; a run imports neither Whisper nor MiniLM; eligibility accounts for every candidate | `slotify-rank training prepare` | `artifacts/training/<run_id>/dataset_summary.json` | **verified** (2026-07-22) |
| Normalization does not leak | Statistics fitted on the train split only; refit refuses validation/test rows; artifact records the fit provenance | (fitted during `training run`) | `artifacts/training/<run_id>/normalizer.json` | **verified** (2026-07-22) |
| No cross-episode or cross-split pairs | Pairs formed only within one episode of one split; generator refuses a mixed split | `slotify-rank training pairs` | `artifacts/training/<run_id>/pairs.jsonl` | **verified** (2026-07-22) |
| Checkpoints are safe and resumable | `state_dict` only, `weights_only=True` load, incompatible/corrupt rejected, OneDrive-safe atomic write | `slotify-rank training inspect --checkpoint …` | ignored `.pt` + `training_summary.json` | **verified** (2026-07-22) |

### Phase 4 evidence fields (synthetic smoke run)

Read from `artifacts/training/gated-58e27d4507da3401/training_summary.json`.
**Synthetic fixture data — not a model-quality result.**

| Field | Value | Note |
|---|---|---|
| `data_provenance` | `synthetic_fixture` | generated numbers, not real audio or human labels |
| `training_dataset_version` | `training-dataset-v1.0.0` | |
| `training_episode_count` | 6 | synthetic |
| `training_candidate_count` | 48 | synthetic |
| `training_pair_count` | 133 | within-episode, deterministic |
| `validation_episode_count` | 2 | synthetic |
| `model_variant` | `gated` | primary architecture |
| `model_parameter_count` | 464,389 | on the synthetic 12-feature schema; ~490k on the real 110-feature layout (`models describe`) |
| `best_validation_ndcg_at_3` | 1.0 | **synthetic** — the latent target is learnable by construction |
| `checkpoint_path` | `artifacts/training/gated-58e27d4507da3401/best_checkpoint.pt` | git-ignored |
| `training_config_hash` | in `resolved_config.json` | content-addressed |
| `training_run_id` | `gated-58e27d4507da3401` | derived from config + data + code |

**Evidence classification: `partially populated using smoke data` (synthetic).**
The system is proven to train, checkpoint, evaluate and resume. It is **not**
proven to rank real ad breaks well — that requires the 200–300+ human labels
Phase 5 depends on. No comparison against `heuristic_offline_v1` is made or
implied in Phase 4.

---

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
| Multimodal PyTorch ranker exists and is trained | Architecture definition, trained checkpoint, parameter count, per-modality gate values | `slotify-rank training run --model-config ml/configs/models/gated_v1.yaml` | `artifacts/training/{run_id}/best_checkpoint.pt`, `training_summary.md` | **machinery verified on synthetic smoke data (2026-07-22); awaiting real labels** |
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
| `weakly_labelled_candidate_count` | derived labels; never counted as human | reported as measured |
| `unlabelled_candidate_count` | generated but never rated | reported as measured |
| `processed_episode_count` | episodes at status `normalized` | reported as measured |

| Intended claim | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| **`processed_audio_hours`** (target 50+) | Measured total duration; episode count; breakdown by source, licence and content type; per-split duration | `dataset import-local` / `fetch` → `dataset probe` → `dataset normalize` → `dataset stats` | `artifacts/dataset/dataset_statistics.json` → `processed_audio_hours` | **implemented, smoke-populated** (0.0365 h / 7 episodes) |
| **`generated_candidate_count`** (target 10 000+) | Measured candidate count; counts by generator source; before/after merge; % from fixed intervals; **separate** totals for human / weak / unlabelled | `candidates generate --config ml/configs/dataset_v1.yaml` then `dataset stats` | `artifacts/dataset/candidate_statistics.json` → `generated_candidate_count` | **implemented, smoke-populated** (20) |
| **`human_labelled_candidate_count`** (target ≈1 500) and **`human_labelled_audio_hours`** (≈8–12) | Label ledger by annotator pseudonym and rubric version; staged at 200–300 → 750 → 1 500 | `label serve` → `label export` → `dataset stats` | `artifacts/dataset/label_statistics.json`, `data/labels/labels_v1.jsonl.meta.json` | **implemented, not yet populated** (0; round-trip verified on smoke data) |
| **`weakly_labelled_candidate_count`** / **`unlabelled_candidate_count`** | Tracked as their own fields; never summed into the human count | `dataset stats` | `artifacts/dataset/label_statistics.json` | **implemented, smoke-populated** (0 weak / 20 unlabelled) |
| **`held_out_evaluation_candidate_count`** | Human-labelled candidates in the test split; test split restricted to podcast-like content (no AMI, no music) | `dataset split` + `dataset validate` + `dataset stats` | `artifacts/dataset/label_statistics.json`, `artifacts/dataset/split_statistics.json` | **implemented, not yet populated** (0 — the smoke corpus has too few series to split) |
| Splits are leak-free | Series-aware grouped splits; validation fails on episode-level or series-level leakage; out-of-domain audio excluded from test | `dataset split --config ml/configs/splits_v1.yaml` then `dataset validate` | `data/manifests/splits_v1.json`, `artifacts/dataset/split_statistics.json`, `artifacts/dataset/validation_report.json` | **implemented, leakage checks verified by failing tests** |
| Synthetic product fallbacks never enter the data | `is_synthetic` candidates pinned to both eligibility flags false by the schema constructor; validation fails otherwise; excluded from every statistic | `candidates generate --include-product-padding` then `dataset validate` + `dataset stats` | `artifacts/dataset/candidate_statistics.json` → `synthetic_product_padding_count` | **verified** (2026-07-22) |
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
