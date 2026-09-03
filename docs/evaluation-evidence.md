# Evaluation evidence matrix

Every capability below is **unverified until its generating command succeeds and its artifact exists
in this repository**. No number in this file may be typed by hand — each is read from the artifact
named in its row.

> **Start here.** The authoritative, machine-generated answer is
> [`artifacts/reports/model_evidence.md`](../artifacts/reports/model_evidence.md) (what each
> capability's artifacts establish), produced by `npm run evidence`.
> This document explains the *reasoning*; that report carries the *numbers*, and it is
> regenerated from the artifacts on every run. Where it disagrees with this file, it is right
> and this file is stale — and CI fails if the report disagrees with its own inputs.

Status vocabulary: `not started` → `in progress` → `evidence generated` → `verified`
(`verified` = artifact exists, was produced by the current `git_sha`, and the plan's acceptance
criterion for that phase passed).

Phase references point at `docs/multimodal-ranking-mvp-plan.md` §36–37.

---

## Phase 1 — foundation (complete)

| Capability | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| A deterministic, reproducible signal-based baseline exists and is the comparison denominator | TypeScript↔Python parity on golden fixtures covering 11 scenarios; every score component serialized; config version stamped into every record | `cd backend; npx tsx scripts/dump-heuristic-golden.ts --out ..\ml\tests\fixtures\heuristic_golden.json` then `cd ml; .\.venv\Scripts\python.exe -m pytest` | `ml/tests/fixtures/heuristic_golden.json`, `config/heuristic_offline_v1.json`, the ML suite | **verified**, re-verified on every push by the `parity` CI job |
| The refactor did not change product behaviour | SHA-256 of the golden output identical before and after hoisting constants; `analyze_cli` output identical on real audio; `npm run typecheck` clean | see above | verification log in this session | **verified** (2026-07-22) |

## Phase 2 — dataset foundation (workflow complete, canonical corpus acquired)

The commands, schemas, splits, labelling loop, validation and statistics all exist and are tested.
**No dataset target has been reached.** The pipeline has been exercised only on the repository's
~4.6-minute smoke fixtures, and every measured number below is that smoke run.

| Capability | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| A reproducible dataset pipeline exists | Import → probe → normalize → generate → split → validate → stats runs end to end with non-zero exit codes on failure, under the ML suite CI runs on every push | see `ml/README.md` § Dataset workflow | `artifacts/dataset/*.json`, `artifacts/dataset/dataset_summary.md` | **verified on the canonical corpus** |
| Dataset integrity is checked, not assumed | 25 named checks including duplicate IDs, checksum mismatch, out-of-bounds candidates, split leakage by episode **and** by series, synthetic-eligibility violations, and a test partition with no target-domain audio — each with a test that makes it fail. The list is in `checks_run`, so it cannot drift from this row unnoticed | `dataset validate --deep` | `artifacts/dataset/validation_report.json` | **verified on the canonical corpus** (0 errors) |
| Human labels are collected and exportable | Local FastAPI UI, SQLite persistence, resumable per-annotator sessions, update-in-place, versioned JSONL export with rubric and threshold metadata | `label serve` then `label export` | `data/labels/labels_full-v2.jsonl` (+ `.meta.json`) | **implemented, not yet populated** — collecting the labels is manual work, not missing software |

Status vocabulary for the dataset rows below:
`implemented but not yet populated` → `partially populated using smoke data` → `fully supported by
real measured data`.

## Phase 3 — multimodal feature pipeline (workflow complete, corpus not acquired)

The pipeline exists, is tested, and has been run end to end with the **real local
models**. As with Phase 2, it has been exercised only on the repository's
~2.2-minute smoke fixtures. Every number below is that smoke run and is labelled
**partially populated using smoke data** — none of it supports a claim about
model quality, and Phase 3 trains nothing.

| Capability | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| A reproducible, resumable, CPU-first feature pipeline exists | Transcribe → acoustic → audio embed → text embed → assemble → validate → stats, non-zero exit on integrity failure, under the ML suite CI runs on every push | `pipeline features --deep` | `artifacts/features/*.json`, `feature_summary.md` | **verified** (2026-07-22) |
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

| Capability | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| A genuinely trainable multimodal PyTorch ranker exists | Five variants behind one interface; each forward pass runs on the real Phase 3 records; parameter counts reported; all under 1M params | `slotify-rank models describe --model gated` | `artifacts/training/<run_id>/training_summary.json` | **verified** (2026-07-22, synthetic smoke) |
| Training is reproducible and CPU-first | Deterministic seeds; content-addressed run id; device/dtype recorded; resume equivalent to uninterrupted within 1e-4; runs on CPU with no GPU | `slotify-rank training run --model-config ml/configs/models/gated_v1.yaml` | `artifacts/training/<run_id>/resolved_config.json`, `environment.json` | **verified** (2026-07-22, synthetic smoke) |
| Phase 3 features load without recomputation | Loader reads cached `.npy` arrays; a run imports neither Whisper nor MiniLM; eligibility accounts for every candidate | `slotify-rank training prepare` | `artifacts/training/<run_id>/dataset_summary.json` | **verified** (2026-07-22) |
| The held-out partition is made of enough different shows to mean something | ≥ 3 independent series per partition, enforced as a hard split failure; the test partition restricted to podcast-format series; no series may occupy more than half a partition's hours | `dataset split --config ml/configs/splits_v4.yaml` | `data/manifests/splits_v4.json` | **verified**, with tests reconstructing the v3 corpus that failed each rule |
| The corpus on disk is the corpus the plan declares | Episodes no committed source registry declares are removed before the split reads them; `--check` reports drift without writing | `dataset reconcile --check` | `artifacts/dataset/reconcile_report.json` | **verified** |
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
proven to rank real ad breaks well — that requires the human labels the
benchmark experiment's gate of 2,400 depends on. No comparison against `heuristic_offline_v1` is made or
implied in Phase 4.

---

## Phase 5A — real corpus bootstrap and labelling machinery (complete; awaiting human labels)

A real, public-domain, target-domain corpus is registered and processed, a
deterministic stratified labelling queue is built, and the readiness
gate is implemented and reports honestly. **Zero human labels exist**, so the
gate reports what is missing rather than proceeding. See
[`docs/human-labelling-workflow.md`](human-labelling-workflow.md).

Classification vocabulary for the rows below (per the Phase 5 brief):
`not yet supported` → `implemented but not populated` → `supported by
preliminary real data` → `supported by held-out real evaluation`.

| Capability | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| A real target-domain corpus is registered under clear licences | ≥ 6 public-domain series, all `direct_download` with `license_name`+`license_url`, all target-domain, verified against the IA metadata API | `dataset fetch --sources ml/configs/sources_v2.yaml` | `ml/configs/sources_real_v1.yaml` + `ml/configs/sources_v2.yaml`, `data/manifests/episodes.jsonl` | **supported by real data** (77 episodes, 40 series, 18.61 h) |
| Enough real audio is processed to yield ≥ 300 eligible candidates with complete features | probe → normalize → generate → `pipeline features`; complete multimodal records reported | `dataset stats --split-version v4` | `artifacts/dataset/*.json`, `data/manifests/features.jsonl` | **supported by real data** (18.61 h, 13,176 generated candidates, 12,930 with a complete multimodal record) |
| A deterministic, stratified labelling queue exists | balanced (split × score-tertile) strata, round-robin episode/series spread, per-episode cap, pilot/primary/overlap/consistency stages, byte-identical on re-run | `label queue --config ml/configs/labelling_queue_full_v2.yaml --split-version v4` | `data/labels/queue_full-v2.json` (git-ignored) | **implemented, verified on real candidates** |
| Synthetic product fallbacks never enter the queue | `is_synthetic` / non-eligible excluded from the pool by construction; tested | `label queue …` | queue coverage + `test_labelling_queue.py` | **verified** |
| The labelling UI hides bias signals and sessions save/resume | reveal-hints off by default; served candidate carries no heuristic score; save → resume preserves labels; export is clean; transcript context resolved from the cache; `--stage pilot` runs a controlled 24-candidate session and rejects out-of-stage labels | `label serve --queue … --stage pilot` → `label export` | pilot service smoke, `test_labelling.py`, `test_labelling_queue.py` | **verified** (round-trip on the real queue) |
| The readiness gate blocks when labels are insufficient | every gate condition checked; blocking reasons listed; `--require-ready` exits non-zero | `experiment readiness --queue … --split-version v4` | `artifacts/experiments/readiness_report.json` | **implemented**; whether it passes depends on the label count, which the report reads |
| An immutable frozen label snapshot can be produced before training | hashes of label/candidate/feature/split manifests, distribution, exclusions, version-immutability | `experiment freeze --snapshot-version v1` | `data/labels/label_snapshot_v1.json` (git-ignored) | **implemented but not populated** |
| `human_labelled_candidate_count` (the experiment's gate: 2,400) | resumable per-annotator SQLite store; versioned export; quality controls | `label run-experiment` → `label export` → `label check` | `data/labels/labels_full-v2.jsonl`, `label_statistics.json` | **not yet supported** (0 human labels; this is a manual data-collection dependency, not missing software — the queue, the UI, the export and the quality controls are all built and tested) |

**The four quantities stay separate and are never conflated:** processed audio
hours (18.61 h, real), generated candidates (13,176, real, **unlabelled**),
human-labelled candidates (0), held-out evaluation candidates (0). No statement
implies that processed candidates were manually labelled. Re-read them from
`artifacts/reports/claim_evidence.md`, which is generated; the values above are
a snapshot of it.

---

Baseline naming, used consistently in every later report:

| Name | What it is | Needs credentials? |
|---|---|---|
| `heuristic_offline_v1` | **canonical baseline / headline denominator** | no |
| `product_api_baseline_v1` | same path plus paid `whisper-1` + `gpt-4o-mini` | yes — reported separately, never required |

---

## Ranking quality — the model

The result this project is built to produce, and the evidence each part of it requires:

> A multimodal PyTorch ranker combining waveform features, speech representations and transcript
> embeddings identifies natural podcast ad breaks, outperforming the signal-based baseline by
> **X %** NDCG@3 — where `X` is whatever the held-out comparison measures.

| Capability | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| Multimodal PyTorch ranker exists and is trained | Architecture definition, trained checkpoint, parameter count, per-modality gate values | `slotify-rank training run --model-config configs/models/gated_v1.yaml` | `artifacts/training/{run_id}/best_checkpoint.pt`, `training_summary.md` | **implemented and served** (`gated`, 489,477 parameters, on the product's inference path); the shipped checkpoint is a weak-label bootstrap, so it evidences the architecture, not ranking quality |
| Three modalities genuinely contribute | 5 variants (`handcrafted`, `text_only`, `audio_only`, `concat`, `gated`) × 3 seeds = 15 cells, checkpoint-selected on validation NDCG@3, reported at the median seed | `slotify-rank experiment train --labels ../data/labels/labels_full-v2.jsonl --split-version v4` | `artifacts/training/<variant>-<hash>/`, matrix summary | **implemented and tested; not run** — the matrix needs human labels, and `experiment train` refuses a non-human label source |
| Signal-based baseline is the real production heuristic | Byte-exact parity between the Python port and the TypeScript scorer; the fixture is regenerated from the TypeScript in CI and any drift fails the build | `cd backend && npx tsx scripts/dump-heuristic-golden.ts --out ../ml/tests/fixtures/heuristic_golden.json` then `cd ml && python -m pytest tests/test_heuristic.py` | `ml/tests/fixtures/heuristic_golden.json`, `config/heuristic_offline_v1.json` | **verified**, and re-verified on every push by the `parity` CI job |
| **X % NDCG@3 improvement** | Baseline + model NDCG@3 on held-out **human-labelled** test episodes, with a percentile bootstrap 95 % CI over episodes, and agreement with `sklearn.metrics.ndcg_score` before anything is published | `slotify-rank evaluation compare --labels ../data/labels/labels_full-v2.jsonl --model <checkpoint> --split test --split-version v4 --require-publishable` | `artifacts/evaluation/<id>/comparison.json`, `metrics.json`, `per_episode_metrics.json`, `summary.md` | **evaluator implemented and tested; not measured** — every publication precondition is enforced, and none is met without human labels |
| Supporting metrics | P@k, R@k, F1@k, MRR and pairwise accuracy at k = 1, 3, 5, plus the per-episode table behind every average | same as above | `artifacts/evaluation/<id>/metrics.json`, `per_episode_metrics.json` | **implemented and tested; not measured** |

`X` is computed only by:

```
ndcg_improvement_pct = 100 * (model_ndcg_at_3 - baseline_ndcg_at_3) / baseline_ndcg_at_3
```

written once, in
`ml/src/slotify_rank/evaluation/compare.py::relative_improvement_percent`. A zero
baseline yields `None`, never an infinite improvement. **If the bootstrap interval on `X` includes
zero, the report says so and no improvement may be reported as a result.**

---

## Dataset and evaluation pipeline

> A dataset and evaluation pipeline spans **`processed_audio_hours`** hours of audio and
> **`generated_candidate_count`** candidate breakpoints, is evaluated against a held-out
> human-labelled subset, reaches **Y %** agreement with human-preferred placements, and reduces
> manual editing time by **Z %**.

Any description of the corpus must make clear that it is *processed*, not *hand-labelled*. Phrasings
such as "10,000+ hand-labelled breakpoints" are prohibited. The human-labelled subset is whatever
`artifacts/dataset/label_statistics.json` measures — currently **0** — against the experiment's gate
of **2,400**, and that measured number is the one tied to any labelling statement. The 2,400-item
*queue* is built and is a separate quantity: `artifacts/labelling/queue_summary.json` records it, and
it is never added to the label count.

### The five dataset quantities, tracked and reported separately

Never collapsed into a single "labelled dataset" figure. Each has its own field in the statistics
artifacts and its own row below.

| Field | Meaning | Target |
|---|---|---|
| `processed_audio_hours` | audio decoded, transcribed and featurised | ≥ 50 |
| `generated_candidate_count` | candidates produced by the generators | ≥ 10 000 |
| `human_labelled_candidate_count` | reviewed by a person against the 1–5 rubric | 2 400 (the experiment's gate, read from `experiment_v2.yaml`) |
| `human_labelled_audio_hours` | audio duration of the episodes containing those labels | reported as measured |
| `held_out_evaluation_candidate_count` | human-labelled **and** in the test split — the only source of final test metrics | reported as measured |
| `weakly_labelled_candidate_count` | derived labels; never counted as human | reported as measured |
| `unlabelled_candidate_count` | generated but never rated | reported as measured |
| `processed_episode_count` | episodes at status `normalized` | reported as measured |

| Capability | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| **`processed_audio_hours`** (target 50+) | Measured total duration; episode count; breakdown by source, licence and content type; per-split duration | `dataset import-local` / `fetch` → `dataset probe` → `dataset normalize` → `dataset stats` | `artifacts/dataset/dataset_statistics.json` → `processed_audio_hours` | **populated from the canonical corpus** (18.61 h / 77 episodes / 40 series) |
| **`generated_candidate_count`** (target 10 000+) | Measured candidate count; counts by generator source; before/after merge; % from fixed intervals; **separate** totals for human / weak / unlabelled | `candidates generate --config ml/configs/dataset_v1.yaml` then `dataset stats` | `artifacts/dataset/candidate_statistics.json` → `generated_candidate_count` | **populated from the canonical corpus** (13,176, target met) |
| **`human_labelled_candidate_count`** (the experiment's gate: 2,400) and **`human_labelled_audio_hours`** | Label ledger by annotator pseudonym and rubric version; the queue is built and allocated 1,680 / 360 / 360 across the splits | `label run-experiment` → `label export` → `dataset stats` | `artifacts/dataset/label_statistics.json`, `artifacts/labelling/queue_summary.json` | **queue built, 0 labels collected** — the queue size and the label count are two different numbers and are never added |
| **`weakly_labelled_candidate_count`** / **`unlabelled_candidate_count`** | Tracked as their own fields; never summed into the human count | `dataset stats` | `artifacts/dataset/label_statistics.json` | **populated from the canonical corpus** (0 weak / 13,176 unlabelled) |
| **`held_out_evaluation_candidate_count`** | Human-labelled candidates in the test split; test split restricted to podcast-format series | `dataset split` + `dataset validate` + `dataset stats` | `artifacts/dataset/label_statistics.json`, `artifacts/dataset/split_statistics.json` | **implemented, not yet populated** (0 — the split holds 3 test series and 2,275 candidates, none labelled) |
| Splits are leak-free | Series-aware grouped splits; validation fails on episode-level or series-level leakage; out-of-domain audio excluded from test; a partition holding fewer than three independent series fails outright | `dataset split --config ml/configs/splits_v4.yaml` then `dataset validate` | `data/manifests/splits_v4.json`, `artifacts/dataset/split_statistics.json`, `artifacts/dataset/validation_report.json` | **verified on the canonical split** (v4: 30 / 7 / 3 series, disjoint, not degraded) |
| Synthetic product fallbacks never enter the data | `is_synthetic` candidates pinned to both eligibility flags false by the schema constructor; validation fails otherwise; excluded from every statistic | `candidates generate --include-product-padding` then `dataset validate` + `dataset stats` | `artifacts/dataset/candidate_statistics.json` → `synthetic_product_padding_count` | **verified** (2026-07-22) |
| **Y % human agreement** — a blind, randomized-order preference study against held-out episodes, needing at least one non-author evaluator | **out of scope for this repository.** No such command exists and none is planned; the design is recorded in the MVP plan so that a claim of human agreement is visibly unbacked here rather than quietly absent. | — | — | **not started, not implemented** |
| **Z % editing-time reduction** — a counterbalanced within-subject timed study of manual versus assisted editing | **out of scope for this repository.** No such command exists and none is planned. Any figure of this kind would be a prior expectation, never a measurement. | — | — | **not started, not implemented** |
| Product integration works end to end | Upload → rank → preview → merge → export in the real product; the heuristic fallback path still matches the golden fixture. The product API is **Express**, not FastAPI; FastAPI hosts only the local labelling UI | `npm run demo`, then `npm run verify` | `docs/model-inference.md`, `docs/demo-runbook.md`, backend and frontend test suites | **verified**, and re-verified on every push by the `backend` and `frontend` CI jobs |

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
4. **Targets are not results.** 50 hours, 10 000 candidates, and ~80 % are *design targets*. Every
   report quotes the measured values from the artifacts above, whatever they turn out to be.
5. **No row is marked `verified` while its command has never been run on the current commit.**
6. **The headline denominator is credential-free.** `heuristic_offline_v1` needs no API key, no
   network and no GPU, so the NDCG@3 comparison is reproducible by anyone who clones the repo.
   `product_api_baseline_v1` may be reported alongside but never replaces it.
7. **Evaluation never reads the frontend.** The fabricated fallback slots in
   `frontend/src/App.tsx:453-482` (≈22/48/72 % with hard-coded confidences) are product technical
   debt and are excluded from every metric by construction — `ml/` computes rankings independently.
   Synthetic padding produced by the *baseline itself* is flagged `is_synthetic` and carries
   `rank: null`, so it cannot enter a ranking either.

---

# Phase 6 — product inference, evaluation, and the current evidence

Phase 6 closed the gap the earlier phases left: a substantial ML package existed,
and the running product did not use it. It also built the machinery that turns
"we could compute an improvement" into "we computed it, and here is why it may or
may not be quoted."

## What changed

| Capability | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| The learned ranker is genuinely wired into the product | `/api/insert-sections` scores real uploads with a real checkpoint; every response names the model, its run id and the labels it was trained on | `npm run demo`, then upload audio | live response `provenance.source = learned_ranker` | **verified** (2026-08-18) |
| Product inference cannot drift from training | The upload is featurised by the *same* Phase 3 stage functions the corpus uses, not a serving-specific reimplementation | `slotify-rank infer rank --audio …` | `ml/src/slotify_rank/inference/episode.py` | **verified** |
| An incompatible checkpoint is refused, not coerced | Variant, feature ordering, dimensions, pipeline versions and normalizer identity all checked before any weight loads; a state-dict shape mismatch becomes `IncompatibleCheckpoint` | `pytest ml/tests/test_inference.py` | 16 passing tests | **verified** |
| The product never invents a recommendation | Selection returns at most `len(candidates)`; the route reports analyser failure as failure; the UI renders exactly what the server sent | `npm run verify` | 45 backend + 15 frontend tests | **verified** |
| A ranking score is never presented as a probability | `placement_score` + `raw_score` + `scoreScale` + `isCalibratedProbability: false` on every response | `npm run verify:backend` | `backend/src/lib/placement.ts` | **verified** |
| Analysis works with no credentials | Server starts and ranks with no `ELEVENLABS_API_KEY`; `GET /api/capabilities` reports `placement: true` | `npm run demo` with no `.env` | `/api/capabilities` | **verified** |
| A held-out comparison exists and refuses unearned numbers | Four independent gates: non-human ground truth, non-human-trained model, non-test split, episode overlap. Plus: a zero baseline yields `None`, never an infinite improvement | `slotify-rank evaluation compare …` | `artifacts/evaluation/<id>/comparison.json` | **verified**, and correctly **blocked** on current data |
| The evidence report distinguishes zero from unmeasured | `NOT YET AVAILABLE` vs `0` are different strings all the way to the markdown | `slotify-rank report model-evidence` | `artifacts/reports/model_evidence.{json,md}` | **verified** |

## The bootstrap checkpoint, stated plainly

`artifacts/training/gated-d8ed976101aa4c3b` is committed and is what the demo
serves. Its targets are `heuristic_offline_v1`'s own score, binned into the 1–5
rubric within each episode. It is a **distillation of the baseline**.

| Property | Value |
|---|---|
| `label_source` | `weak_heuristic` |
| `data_provenance` | `weak_supervision` |
| `evidence_class` | weakly supervised bootstrap — not a model-quality result |
| Validation NDCG@3 | 1.0 — **meaningless as quality**: it measures fidelity to its teacher |

It is admissible evidence for **multimodal learned ranking** (a multimodal
PyTorch ranker exists, is trained on real audio features, and is served). It is
inadmissible for **human-labelled dataset scale** and **held-out ranking
improvement**, and four separate mechanisms enforce that:

1. `datasets/labels.py` refuses weak labels unless the source is named on the
   command line;
2. `training/reporting.py` classifies the run as a bootstrap and attaches a
   warning to its summary;
3. `evaluation/compare.py` blocks the headline when either the ground truth or
   the model's training labels are non-human;
4. `services/ranker.ts` attaches a warning to every product response the model
   produces.

## Current evidence status

Read from `artifacts/reports/model_evidence.md`. Re-run `npm run evidence` for
live values; these were the values at the time of writing.

| Capability | Evidence status | Why |
|---|---|---|
| **Multimodal learned ranking** — a multimodal PyTorch ranker consumes audio and transcript features to rank candidate ad breaks, and serves the product. | **SUPPORTED** | A 489,477-parameter gated fusion model consumes `openai/whisper-tiny.en` speech representations, `sentence-transformers/all-MiniLM-L6-v2` transcript embeddings and 110 handcrafted scalars, and is served through `/api/insert-sections`. The served checkpoint is weakly supervised, which the report states. |
| **Human-labelled dataset scale** — enough human labels, spread widely enough, for the readiness gate to pass. | read `human_labelled_candidate_count` from the report | Generated candidates are **not** labels, and the report keeps the two fields apart precisely so they cannot be conflated. Whatever the generated count is, it says nothing about this row. |
| **Held-out ranking improvement** — a measured relative NDCG@3 improvement over `heuristic_offline_v1`. | **NOT YET SUPPORTED** | No publishable held-out comparison exists. One comparison ran and is recorded, blocked, with its real measured numbers visible. |

**Neither the dataset-scale nor the ranking-improvement capability may be
reported as a result in its current form.** The infrastructure to establish both
is complete and tested; what is missing is human labelling time, which is human
work and cannot be synthesised.

## What would establish the two outstanding capabilities

The infrastructure for both is complete and tested. What is missing is human
labelling time, which is human work and cannot be synthesised.

```powershell
cd ml
# 1. Everything mechanical: acquire, generate, featurise, split, queue.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset prepare-experiment

# 2. The only manual step. 2,400 unique candidates is the experiment's gate.
.\.venv\Scripts\python.exe -m slotify_rank.cli label run-experiment

# 3. Gate, export, freeze, pin.
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment readiness --split-version v4 `
    --experiment-config configs\experiment_v2.yaml --require-ready
.\.venv\Scripts\python.exe -m slotify_rank.cli label export --dataset-version full-v2
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment freeze --snapshot-version full-v2 --split-version v4
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment manifest --require-ready

# 4. Five ablations x three seeds; the MEDIAN seed by validation NDCG@3 is reported.
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment train --labels ..\data\labels\labels_full-v2.jsonl --split-version v4

# 5. Once, at the end, on the frozen test split.
.\.venv\Scripts\python.exe -m slotify_rank.cli evaluation compare --split test --split-version v4 --require-publishable

# 6. The report then fills itself in.
npm run evidence
```

Every number that comes out of step 5 is whatever it is. The claim threshold
lives in `ml/configs/experiment_v2.yaml` and the measured improvement comes
from the comparison artifact. If the measured value is below the threshold the
comparison refuses to mark itself publishable and prints the measured number;
there is no code path in this repository that can do anything else, and a test
asserts the threshold is not a literal in the checker.

---

## the headline claims and what each one requires

The verdict for each claim is read from the artifact named in the third column.

| Claim | What establishes it | Where the verdict is read from |
|---|---|---|
| A multimodal PyTorch ranker exists | a training run summary naming a variant and a parameter count | `artifacts/training/*/training_summary.json` |
| It uses audio *and* transcript | both encoders named in the embedding statistics; handcrafted count from the feature statistics | `artifacts/features/*.json` |
| ≥ 2,400 human-labelled candidates | the label count against the gate in the experiment definition | `artifacts/dataset/label_statistics.json` |
| The checkpoint is human-trained | a run recording `label_source: human` | `artifacts/training/*/training_summary.json` |
| A frozen held-out test set | a non-degraded, series-grouped split whose test groups appear in no other partition | `artifacts/experiments/experiment-v2.json` |
| The canonical baseline was used | the comparison's baseline matches the one the experiment declares | `artifacts/evaluation/*/comparison.json` |
| Measured baseline / model NDCG@3, and the improvement | a **publishable** comparison; a blocked one is never treated as a result | `artifacts/evaluation/*/comparison.json` |
| Improvement ≥ the claimed threshold | measured value vs the committed threshold | both of the above |
| PyTorch / transformers / librosa / scikit-learn | declared in `pyproject.toml`, imported by a committed module (AST, not grep), **and** recorded as having run by a named artifact field | `ml/pyproject.toml`, feature statistics, comparison |
| Award wording | the precise string present in the README, and no stronger claim present | `README.md` |

### Why scikit-learn's bar is the highest

It would be the easiest of the four to fake: add a line to `pyproject.toml` and
the claim "the stack includes scikit-learn" is true in a trivial sense. So it
requires a published comparison recording **both** an independent NDCG
cross-check that agreed *and* a fitted classical baseline. Its two jobs are:

- `sklearn.metrics.ndcg_score` independently recomputes the headline metric. The
  result is a ratio of two NDCG values from one implementation, so a bug there
  moves numerator and denominator together and every test checking that
  implementation against itself still passes. A disagreement blocks publication.
- `HistGradientBoostingRegressor` with `GroupKFold` provides the classical
  comparison point — a strong tabular model on the handcrafted scalars alone,
  tuned inside the training split with the episode as the group. It answers
  "would a good tabular model have done just as well?", which is the question a
  reader should ask about any multimodal result.

If either were removed, the claim would drop to NOT MEASURED and the honest
action would be to stop listing scikit-learn as part of the stack. That is the
behaviour the check is designed to produce.
