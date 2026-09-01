# `slotify_rank` — multimodal breakpoint ranking

Python package for the Slotify learning-to-rank system. It is independent of the
Node/Express backend: `backend/` never imports it, and it never imports
`ad_inserter`. The only thing they share is the canonical baseline configuration
at `config/heuristic_offline_v1.json`.

**Status:** Phase 1 (the deterministic `heuristic_offline_v1` baseline and the
evaluation metrics) and Phase 2 (the dataset foundation: sources, ingestion,
normalization, candidate generation, splits, labelling, validation, statistics)
are implemented. No PyTorch, no learned embeddings, no trained model yet — see
`docs/multimodal-ranking-mvp-plan.md`.

Everything runs **offline on CPU**. The single exception is
`dataset fetch`, which downloads declared source URLs; no other command touches
the network, and nothing anywhere downloads a model or calls a paid API.

Audio decoding reuses the `ffmpeg` / `ffprobe` binaries the product already
requires — set `FFMPEG_BIN` / `FFPROBE_BIN` if they are not on `PATH`.

## Setup (Windows / PowerShell)

Requires Python 3.12.13 and [uv](https://docs.astral.sh/uv/). The backend keeps
its own Python environment; this one is separate and does not disturb it.

```powershell
cd ml

# OneDrive-synced checkouts reject hardlinks (os error 396), so uv must copy.
$env:UV_LINK_MODE = "copy"

uv venv --python 3.12.13 .venv
# [dev]      pytest + coverage + httpx
# [label]    FastAPI + uvicorn for the labelling UI
# [features] torch, transformers, sentence-transformers, librosa (Phase 3)
# [sklearn]  scikit-learn: the independent NDCG cross-check and the classical
#            comparison point (evaluation/crosscheck.py, baselines/classical.py)
#
# [features] is ~1 GB of wheels. Omit it and the Phase 1 baseline and the whole
# Phase 2 dataset pipeline still install in seconds. [sklearn] is small and is
# needed to publish a headline: a comparison whose metric was not independently
# verified is blocked.
uv pip install --python .\.venv\Scripts\python.exe -e ".[dev,label,features,sklearn]"

# Verify
.\.venv\Scripts\python.exe -m slotify_rank.cli version
```

To activate the environment for an interactive session:

```powershell
.\.venv\Scripts\Activate.ps1
```

<details>
<summary>bash / Git Bash equivalent</summary>

```bash
cd ml
export UV_LINK_MODE=copy
uv venv --python 3.12.13 .venv
uv pip install --python ./.venv/Scripts/python.exe -e ".[dev]"
./.venv/Scripts/python.exe -m slotify_rank.cli version
```
</details>

If `uv` is unavailable, the stdlib equivalent works too:

```powershell
& "$env:LOCALAPPDATA\Python\pythoncore-3.12-64\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Tests

```powershell
cd ml
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest --cov=slotify_rank --cov-report=term-missing
```

No test in the default run reaches the network, calls a paid API, downloads a
model, requires a GPU, or trains anything. The HTTP layer is stubbed in the fetch
tests, and the two Phase 3 models are stubbed at the stage boundary so the whole
pipeline -- caching, resumability, assembly -- runs offline.

The one deliberate exception is the `model_smoke` marker, which downloads and
runs the real `whisper-tiny.en` and `all-MiniLM-L6-v2` weights on one short
fixture. It is **deselected by default**:

```powershell
.\.venv\Scripts\python.exe -m pytest -m model_smoke   # opt in (~1 min, CPU)
```

Its job is to turn the mocked dimension claims into evidence: that tiny.en's
encoder really is 384-dimensional and not 768, that its temporal resolution
really is 20 ms per frame, and that the constructed transcript vector really is
1536.

Tests that decode audio synthesise their own WAVs; the one test that reads a
real repository fixture is marked `audio` and skips cleanly without FFmpeg:

```powershell
.\.venv\Scripts\python.exe -m pytest -m audio        # only the real-audio test
.\.venv\Scripts\python.exe -m pytest -m "not audio"  # skip it
```

## Commands

```powershell
# Version stamps (package, schema, heuristic config, canonical baseline)
.\.venv\Scripts\python.exe -m slotify_rank.cli version

# Resolved baseline configuration
.\.venv\Scripts\python.exe -m slotify_rank.cli config show
.\.venv\Scripts\python.exe -m slotify_rank.cli config show --profile legacy_cli_v1

# Rank candidate breakpoints for one or more episodes
.\.venv\Scripts\python.exe -m slotify_rank.cli heuristic rank `
    --input episodes.json --output artifacts/evaluation/baseline_rankings.json

# Score saved rankings against graded relevance labels
.\.venv\Scripts\python.exe -m slotify_rank.cli evaluate `
    --predictions artifacts/evaluation/baseline_rankings.json `
    --labels data/labels/relevance.json `
    --output artifacts/evaluation/baseline_results.json
```

Exit codes: `0` success, `1` runtime failure (message on stderr), `2` usage error.
JSON goes to `--output`; the human-readable summary goes to stdout.

The console script `slotify-rank` is installed as an alias for
`python -m slotify_rank.cli`.

### Input format

`heuristic rank` takes candidate *metadata*, so the baseline is testable with no
audio. Audio-driven candidate generation arrives in Phase 3.

```json
{
  "episodes": [
    {
      "episode_id": "ep-001",
      "duration_seconds": 600.0,
      "mode": "podcast",
      "count": 3,
      "silence_candidates": [
        { "ms": 60000, "silence_ms": 1800, "snippet": "So that was the whole story." }
      ],
      "transcript_candidates": [
        { "ms": 180000, "silence_ms": 900, "snippet": "That's exactly right." }
      ]
    }
  ]
}
```

Labels for `evaluate` use graded relevance (the 1–5 rubric in
`docs/multimodal-ranking-mvp-plan.md` §12):

```json
{ "episodes": [ { "episode_id": "ep-001", "relevance": { "ep-001:000060000": 5.0 } } ] }
```

## Baseline parity

`heuristic_offline_v1` is a port of the live TypeScript product scorer. The
fixture `tests/fixtures/heuristic_golden.json` is generated **only** from the
TypeScript implementation:

```powershell
cd backend
npx tsx scripts/dump-heuristic-golden.ts --out ..\ml\tests\fixtures\heuristic_golden.json
cd ..\ml
.\.venv\Scripts\python.exe -m pytest tests/test_heuristic.py
```

If parity fails, the Python port or the TypeScript scorer has drifted. **Diagnose
the difference — never regenerate the fixture to make a failing test pass.**

## the benchmark experiment, in two commands

Everything mechanical, then the one step that is not:

```powershell
cd ml
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset prepare-experiment
.\.venv\Scripts\python.exe -m slotify_rank.cli label run-experiment
```

The first acquires, generates, featurises, splits, validates, counts and queues,
then prints a *measured* readiness summary. The second pre-cuts every clip,
reports how many labels remain, and opens the labelling UI.

Once the labels exist:

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment readiness --split-version v4 `
    --experiment-config configs\experiment_v2.yaml --require-ready
.\.venv\Scripts\python.exe -m slotify_rank.cli label export --dataset-version full-v2
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment freeze --snapshot-version full-v2 --split-version v4
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment manifest --require-ready

# Five ablations x three seeds; reports the MEDIAN seed by validation NDCG@3.
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment train `
    --labels ..\data\labels\labels_full-v2.jsonl --split-version v4

# Once, at the end, on the frozen test split.
.\.venv\Scripts\python.exe -m slotify_rank.cli evaluation compare `
    --labels ..\data\labels\labels_full-v2.jsonl `
    --model ..\artifacts\training\<run>\best_checkpoint.pt `
    --split test --split-version v4 `
    --experiment-config configs\experiment_v2.yaml --require-publishable
```

See [`../docs/evaluation-evidence.md`](../docs/evaluation-evidence.md) for what
each of those is for and what is frozen when.

## Dataset workflow

The individual stages, in order -- this is what `prepare-experiment`
runs. Every command is resumable: re-running a completed stage is a cheap no-op,
so an interrupted run costs only the work actually lost.

```powershell
cd ml

# 0. Resolve the committed corpus plan into a source registry (networked,
#    metadata only). --check re-resolves and fails instead of writing.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset discover `
    --plan configs/corpus_v2.yaml --output configs/sources_v2.yaml

# 1. Declare where local audio comes from and under what licence.
#    Edit configs/sources.yaml first -- see "Adding your own audio" below.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset import-local --sources configs/sources.yaml

# 2. Download any direct_download sources. The ONLY networked command.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset fetch --sources configs/sources.yaml

# 3. Measure real duration / sample rate / channels / format with ffprobe.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset probe

# 4. Render to the ML format: 16 kHz mono PCM WAV, cached, atomic.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset normalize

# 5. Generate the candidate pool.
.\.venv\Scripts\python.exe -m slotify_rank.cli candidates generate --config configs/dataset_v1.yaml

# 6. Deterministic, series-grouped, leakage-safe splits. A manifest is
#    immutable per version: v1 was the smoke corpus, v2 the first six real
#    series, v3 the current one.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset split --config configs/splits_v4.yaml

# 7. Integrity gate. Non-zero exit on any error; --deep re-verifies every SHA-256.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset validate --deep

# 8. Statistics -> artifacts/dataset/*.json + dataset_summary.md
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset stats

# 9. Unreferenced generated files. `reconcile` drops manifest episodes but never
#    deletes audio, so a corpus version bump leaves the renders and transcripts
#    of removed episodes behind. This reports them and deletes nothing.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset orphans
```

Add `--data-root <path>` to any of them to work against a different corpus
directory (or set `SLOTIFY_DATA_ROOT`).

### Adding your own audio

Append to `configs/sources.yaml`. **`series_id` is the field that matters most**
— it is what the splitter groups on, so every episode of one show must share it,
or the split will leak.

```yaml
  - id: my-show-ep-001                      # lowercase slug, unique
    source_type: local_file                 # copied into data/raw/
    path: C:/Users/you/Audio/ep-001.mp3     # absolute, or repo-relative
    title: "My Show 001 - guest name"
    series_id: my-show                      # SAME for every episode of the show
    source_name: "Private recording"
    content_type: podcast                   # podcast|interview|conversational|narrated|meeting|music|other
```

For a remote file use `source_type: direct_download` with a `url`, and declare
`license_name` **and** `license_url` — both are mandatory and enforced at parse
time. Add `expected_sha256` when you know it; a mismatch aborts the download and
deletes the partial file. If you cannot name the licence, download it yourself
and register it as a `local_file`: private, never redistributed, no licence
claimed on your behalf.

Then run steps 1, 3, 4, 5 above. Episode IDs are derived from the audio's
SHA-256, so re-importing the same file is idempotent and two copies of one
recording collapse into one episode.

### Labelling

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli label serve --port 8000
# open http://127.0.0.1:8000/ , enter an annotator id (a pseudonym is fine)

# Restrict to a queue, or to just its pilot stage for a controlled first pass:
.\.venv\Scripts\python.exe -m slotify_rank.cli label serve `
    --queue ..\data\labels\queue_full-v2.json --stage pilot --port 8000

.\.venv\Scripts\python.exe -m slotify_rank.cli label export
```

Rubric and guidance: `docs/labelling-guide.md`; the step-by-step pilot session is
`docs/pilot-labelling.md`. Ratings save immediately, sessions resume where you
stopped, and the heuristic's score is hidden from the annotator by default to
avoid biasing the labels. Transcript context either side of the break is
resolved from the cached episode transcript (the same selection the feature
pipeline uses), so what the annotator reads matches what the model consumes.

### What is and is not committed

| Committed | Ignored |
|---|---|
| `configs/*.yaml`, code, tests | `data/` — audio, manifests, SQLite, exports |
| `artifacts/dataset/*.json`, `dataset_summary.md` | normalized renders, clip cache, `.venv`, coverage |

No audio is ever committed. The corpus is reconstructed from
`configs/sources.yaml` by `dataset import-local` / `dataset fetch`.

### The route from smoke data to the real corpus

The committed statistics currently describe seven repository fixtures totalling
~4.6 minutes. That is a **pipeline test, not a corpus.** The staged route:

| Stage | Audio | Candidates | Labels |
|---|---|---|---|
| now (smoke) | ~0.04 h | ~20 | 0 |
| pipeline debug | 2–4 h | ~500+ | 200–300 |
| first model | 8–12 h | ~2 000+ | ~750 |
| MVP target | 50+ h | 10 000+ | ~1 500 |

At least 6 independent series are needed before a train/validation/test split is
meaningful; below that the splitter emits a single `development` partition and
marks the result degraded rather than pretending otherwise.

## Layout

```
src/slotify_rank/
  jsnum.py                  ECMAScript rounding semantics (Math.round, toFixed)
  cli.py                    argparse entrypoint (Phase 1 commands)
  dataset_cli.py            dataset / candidates / label commands
  config/settings.py        loader for config/heuristic_offline_v1.json
  config/versions.py        version stamps embedded in artifacts
  candidates/schema.py      canonical ranking candidate/episode schema
  candidates/heuristic.py   the heuristic_offline_v1 port
  candidates/config.py      candidate-generation settings (dataset_v1.yaml)
  candidates/audio_candidates.py    silence / pause / RMS-minimum / fixed-interval
  candidates/transcript_candidates.py   optional timestamped-transcript input
  candidates/merge.py       provenance-preserving merge + product-parity guard
  candidates/generate.py    per-episode generation orchestration
  data/schema.py            episode + dataset-candidate records
  data/sources.py           sources.yaml parsing, licence and secret enforcement
  data/fetch.py             the only networked module
  data/import_local.py      local files and repository fixtures
  data/probe.py             ffprobe metadata
  data/normalize.py         16 kHz mono PCM WAV rendering + cache
  data/audio_io.py          millisecond energy envelopes
  data/manifests.py         deterministic JSONL manifests
  data/splits.py            series-grouped deterministic splitting
  data/validate.py          the integrity gate
  data/stats.py             the eight tracked quantities
  labelling/database.py     SQLite label store
  labelling/service.py      local FastAPI labelling UI
  labelling/export.py       versioned JSONL export
  evaluation/metrics.py     NDCG@k, P@k, R@k, F1, MRR, pairwise accuracy
configs/heuristic_v1.yaml   run settings (never baseline constants)
configs/sources.yaml        source registry
configs/dataset_v1.yaml     candidate-generation recall settings
configs/splits_v1.yaml      split ratios, seed, grouping
  transcription/schema.py   versioned transcript records (integer ms)
  transcription/segments.py chunk planning, overlap reconciliation, sentence ends
  transcription/local_whisper.py  local whisper-tiny.en via Transformers
  transcription/cache.py    transcript read/write + identity enforcement
  features/windows.py       candidate-centred windows, clipped at episode edges
  features/acoustic.py      RMS / spectral / onset descriptors
  features/structural.py    position, provenance, heuristic component scores
  features/transcript.py    deterministic context selection + text scalars
  features/schema.py        feature spec and candidate feature record
  features/assemble.py      the identity-based joins and the feature manifest
  features/validate.py      the Phase 3 integrity gate
  features/stats.py         feature / transcription / embedding / cache reports
  embeddings/whisper_audio.py  frozen encoder, pooled per candidate (384-d)
  embeddings/minilm_text.py    frozen MiniLM (384-d native, 1536 constructed)
  embeddings/pooling.py     frame-to-time mapping and window pooling
  embeddings/store.py       .npy + JSON sidecar, never pickle
  embeddings/device.py      CUDA autodetect, CPU default
  pipeline/identity.py      deterministic cache identities
  pipeline/state.py         missing/partial/complete/failed/stale ledgers
  pipeline/stages.py        the five resumable stages
  pipeline/feature_pipeline.py  end-to-end orchestration
configs/transcription_v1.yaml   local Whisper settings
configs/features_v1.yaml        windows, acoustics, transcript context
configs/embeddings_v1.yaml      model ids, devices, pooling
configs/feature_pipeline_v1.yaml  orchestration policy (not cache-identity)
tests/                      unit + parity + dataset + CLI + service + feature tests
```

## Feature pipeline (Phase 3)

Full detail in [`docs/feature-pipeline.md`](../docs/feature-pipeline.md). The
short version -- local models, CPU, resumable, cached:

```powershell
cd ml
$py = ".\.venv\Scripts\python.exe"

# Everything, then validate and report.
& $py -m slotify_rank.cli pipeline features `
    --transcription-config configs/transcription_v1.yaml `
    --features-config configs/features_v1.yaml `
    --embeddings-config configs/embeddings_v1.yaml `
    --deep

# Progress, without loading a model.
& $py -m slotify_rank.cli pipeline status
```

One episode, or one split:

```powershell
& $py -m slotify_rank.cli pipeline features --episode-id <EPISODE_ID> --deep
& $py -m slotify_rank.cli pipeline features --split train --deep
```

Individual stages (`transcribe run`, `features acoustic`, `embeddings audio`,
`embeddings text`, `features assemble`, `features validate`, `features stats`)
run the same work and share the same caches.

Dimensions, stated because they are easy to misreport:

| Quantity | Value |
|---|---|
| whisper-tiny.en encoder hidden size | **384** (not 768 -- that is whisper-base) |
| all-MiniLM-L6-v2 native output | **384** |
| constructed transcript vector | **1536** = 384 x 4 blocks |
| handcrafted scalar features | **110**, raw and unnormalized, each with a mask |

Re-running is cheap: a second identical run is entirely cache hits and skips
every model. Artifacts are invalidated by their *inputs* -- audio checksum, model
id and revision, config digests, library versions -- never by timestamps.

## Training (Phase 4)

Full detail in [`docs/model-training.md`](../docs/model-training.md). The
CPU-first PyTorch learning-to-rank system that consumes the Phase 3 feature
artifacts -- **without recomputing any transcript or embedding** -- and trains
ranking models over candidates grouped by episode. Five variants
(`handcrafted`, `text_only`, `audio_only`, `concat`, `gated`) behind one
interface; a pairwise margin ranking objective plus an auxiliary acceptability
head; validation NDCG@3 grouped per episode; resumable checkpoints.

```powershell
cd ml
$py = ".\.venv\Scripts\python.exe"

# The model registry.
& $py -m slotify_rank.cli models list
& $py -m slotify_rank.cli models describe --model gated

# Prepare a dataset (eligibility accounting) and generate within-episode pairs.
& $py -m slotify_rank.cli training prepare --labels data\labels\labels_v1.jsonl
& $py -m slotify_rank.cli training pairs   --labels data\labels\labels_v1.jsonl

# Train the gated multimodal ranker on CPU, then validate / inspect / resume.
& $py -m slotify_rank.cli training run `
    --labels data\labels\labels_v1.jsonl `
    --model-config configs\models\gated_v1.yaml
& $py -m slotify_rank.cli training validate --labels data\labels\labels_v1.jsonl --checkpoint artifacts\training\<run_id>\best_checkpoint.pt
& $py -m slotify_rank.cli training inspect  --checkpoint artifacts\training\<run_id>\best_checkpoint.pt
```

**Smoke training on synthetic fixtures** (until real labels exist -- generated
numbers, never reported as data):

```powershell
& $py -m slotify_rank.cli training synthesize --data-root ..\data-synthetic --episodes 10 --candidates-per-episode 8
& $py -m slotify_rank.cli training run `
    --data-root ..\data-synthetic `
    --labels ..\data-synthetic\labels\labels_synthetic.jsonl `
    --model-config configs\models\gated_v1.yaml --smoke
```

Every run writes `artifacts/training/<run_id>/` with `resolved_config.json`,
`environment.json`, `dataset_summary.json`, `normalizer.json`,
`epoch_metrics.jsonl`, `training_summary.{json,md}` and the git-ignored
`best_checkpoint.pt` / `last_checkpoint.pt`. Model hyperparameters live in
`configs/models/*.yaml` and `configs/training_v1.yaml`, never in source.

Phase 4 metrics on synthetic data are **not** model-quality evidence -- see
`docs/model-training.md` § "How Phase 4 differs from Phase 5".

---

## Product inference (Phase 6)

Loading a trained checkpoint and scoring one audio file. Never trains anything —
`slotify_rank.inference` has no code path that can fit a model.

```bash
# What is in a checkpoint, without scoring anything.
python -m slotify_rank.cli infer describe \
  --checkpoint ../artifacts/training/gated-d8ed976101aa4c3b/best_checkpoint.pt

# Rank one audio file. JSON on stdout, progress on stderr.
python -m slotify_rank.cli infer rank \
  --audio ../backend/audio_tests/rogan-test1.mp3 \
  --checkpoint ../artifacts/training/gated-d8ed976101aa4c3b/best_checkpoint.pt \
  --top 3

# Faster, at the cost of the text modality. The response says the text block was
# masked; it never pretends a transcript existed.
python -m slotify_rank.cli infer rank --audio EPISODE.mp3 --checkpoint … --no-transcribe
```

The Express API calls exactly this command (`backend/src/services/learned-ranker.ts`).
The upload becomes a throwaway single-episode corpus and runs through the real
Phase 3 stages, so the feature layout at serving time is the layout the model was
trained on by construction. See [`../docs/model-inference.md`](../docs/model-inference.md).

## Weak bootstrap labels

There are no human labels yet, which correctly blocks every quality claim. It
also blocked something much smaller: proving the inference path works end to end
on real audio with real weights. `label weak` unblocks only the second thing.

```bash
# Grade every candidate by binning heuristic_offline_v1's own score into the
# 1-5 rubric within its episode. NOT human labels; stamped weak_heuristic.
python -m slotify_rank.cli label weak

# Train on them. The source must be named; there is no default that reaches it.
python -m slotify_rank.cli training run \
  --model-config configs/models/gated_v1.yaml \
  --labels ../data/labels/weak_labels_v1.jsonl \
  --allow-label-source weak_heuristic \
  --split-version v2
```

A model trained this way is a **distillation of the baseline**. Its validation
NDCG measures fidelity to its teacher, not ranking quality, and
`evaluation compare` refuses to publish an improvement computed from it. The run
summary classifies it as a bootstrap and carries a warning saying all of this.

## Held-out evaluation

```bash
python -m slotify_rank.cli evaluation compare \
  --labels ../data/labels/labels_v1.jsonl \
  --model ../artifacts/training/<run_id>/best_checkpoint.pt \
  --baseline heuristic_offline_v1 \
  --split test --split-version v2 \
  --require-publishable
```

Writes `artifacts/evaluation/<evaluation_id>/`:

| File | Contents |
|---|---|
| `comparison.json` | Inputs, both systems' metrics, the headline, and every blocking reason |
| `metrics.json` | Aggregates at k = 1, 3, 5 |
| `per_episode_metrics.json` | The per-episode detail behind every average |
| `ranked_candidates.jsonl` | Both systems' score and the label, per candidate |
| `summary.md` | Rendered from the same dict as the JSON |

The headline is
`100 * (model_ndcg_at_3 - baseline_ndcg_at_3) / baseline_ndcg_at_3`, written once
in `evaluation/compare.py::relative_improvement_percent`. It is marked **not
publishable** — and `--require-publishable` exits 1 — when any of these holds:

- the ground truth is not `human` (comparing against the baseline using the
  baseline's own labels is circular);
- the model was trained on non-human labels;
- the split is not `test`;
- the baseline is not `heuristic_offline_v1`;
- any evaluation episode also appears in the training run's episode list;
- no candidate in the split meets the relevance threshold, so NDCG is undefined.

A zero baseline returns `None`, never an infinite improvement.

## The evidence report

`model-evidence` says what each capability's artifacts currently establish, with
every value read from a generated artifact rather than typed.

```bash
python -m slotify_rank.cli report model-evidence

# or, from the repository root, regenerating every upstream artifact first:
npm run evidence

# CI mode: regenerate and fail if the committed report has drifted.
python -m slotify_rank.cli report model-evidence --check
```

Reads every generated artifact and writes
`artifacts/reports/model_evidence.{json,md}` with an evidence status per
capability. Nothing in it is typed. A metric nothing produced renders
`NOT YET AVAILABLE`; a measured
zero renders `0`. The two are different strings on purpose — rendering both as
`0` would let a reader think a metric was measured and came out badly when it was
never measured at all.
