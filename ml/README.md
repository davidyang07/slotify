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
# [dev] = pytest + coverage + httpx; [label] = FastAPI + uvicorn for the
# labelling UI. Omit [label] if you only need the dataset pipeline.
uv pip install --python .\.venv\Scripts\python.exe -e ".[dev,label]"

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

No test reaches the network, calls a paid API, downloads a model, requires a
GPU, or trains anything. The HTTP layer is stubbed in the fetch tests.

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

## Dataset workflow

The full loop, in order. Every command is resumable: re-running a completed
stage is a cheap no-op, so an interrupted run costs only the work actually lost.

```powershell
cd ml

# 1. Declare where audio comes from and under what licence.
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

# 6. Deterministic, series-grouped, leakage-safe splits.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset split --config configs/splits_v1.yaml

# 7. Integrity gate. Non-zero exit on any error; --deep re-verifies every SHA-256.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset validate --deep

# 8. Statistics -> artifacts/dataset/*.json + dataset_summary.md
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset stats
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

.\.venv\Scripts\python.exe -m slotify_rank.cli label export
```

Rubric and guidance: `docs/labelling-guide.md`. Ratings save immediately,
sessions resume where you stopped, and the heuristic's score is hidden from the
annotator by default to avoid biasing the labels.

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
tests/                      unit + parity + dataset + CLI + service tests
```
