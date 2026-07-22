# `slotify_rank` — multimodal breakpoint ranking

Python package for the Slotify learning-to-rank system. It is independent of the
Node/Express backend: `backend/` never imports it, and it never imports
`ad_inserter`. The only thing they share is the canonical baseline configuration
at `config/heuristic_offline_v1.json`.

**Phase 1 status:** the deterministic offline baseline (`heuristic_offline_v1`)
and the evaluation metrics are implemented. No PyTorch, no FastAPI, no dataset,
no pretrained models yet — see `docs/multimodal-ranking-mvp-plan.md`.

Everything in Phase 1 runs **offline on CPU**: no network access, no paid APIs,
no model downloads.

## Setup (Windows / PowerShell)

Requires Python 3.12.13 and [uv](https://docs.astral.sh/uv/). The backend keeps
its own Python environment; this one is separate and does not disturb it.

```powershell
cd ml

# OneDrive-synced checkouts reject hardlinks (os error 396), so uv must copy.
$env:UV_LINK_MODE = "copy"

uv venv --python 3.12.13 .venv
uv pip install --python .\.venv\Scripts\python.exe -e ".[dev]"

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

No test reaches the network, downloads a model, or reads real audio.

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

## Layout

```
src/slotify_rank/
  jsnum.py                  ECMAScript rounding semantics (Math.round, toFixed)
  cli.py                    argparse entrypoint
  config/settings.py        loader for config/heuristic_offline_v1.json
  config/versions.py        version stamps embedded in artifacts
  candidates/schema.py      canonical candidate/episode schema
  candidates/heuristic.py   the heuristic_offline_v1 port
  evaluation/metrics.py     NDCG@k, P@k, R@k, F1, MRR, pairwise accuracy
configs/heuristic_v1.yaml   run settings (never baseline constants)
tests/                      unit + parity + CLI tests
```
