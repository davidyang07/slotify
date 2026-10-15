# Slotify

**Semantic and signal-aware ad insertion for audio.** Upload a podcast episode
or a song, and Slotify finds the best places for an ad to go, writes and voices
the sponsor read, and stitches it into the original audio with loudness matching
and crossfades.

UofTHacks 13 Winner — MLH Best Use of ElevenLabs.
[Video demo.](https://www.youtube.com/watch?v=S4m1lpipni0)

---

## What it does

An ad break in the wrong place is the difference between a listener tolerating a
sponsor read and skipping it. Slotify treats placement as a ranking problem
inside a single episode — of the couple hundred plausible cuts, which three are
best — and reads both the waveform and the transcript to answer it.

| Stage | What happens |
| --- | --- |
| **Analyze** | Candidate breakpoints generated from silence runs, pauses and RMS valleys across the whole episode, then ranked by a multimodal PyTorch model (see [Model status](#model-status)) or the frozen signal-based heuristic |
| **Preview** | Listen to the audio either side of a proposed cut before committing |
| **Generate** | Sponsor copy written for the product in the chosen tone and language, then voiced — optionally in a cloned voice |
| **Merge / export** | The read is spliced in — the episode pauses and resumes after it — with loudness matching and crossfaded joins, via `ffmpeg` |

Placement analysis runs credential-free. API keys only unlock the copywriting
and voice halves, and `GET /api/capabilities` tells the UI exactly which
features the running deployment can offer.

## Architecture

```mermaid
flowchart LR
  A[React 19 + Vite<br/>TypeScript UI] --> B[Node + Express API]
  B --> C[Python: candidate<br/>generation]
  C --> D[Handcrafted features<br/>110 acoustic + structural]
  D --> W[Whisper tiny.en<br/>speech representation]
  D --> X[MiniLM-L6-v2<br/>transcript embeddings]
  W & X & D --> E[PyTorch ranker<br/>gated multimodal fusion]
  E --> R[Ranked ad breaks] --> A
  B --> O[OpenAI<br/>sponsor copy]
  B --> G[ElevenLabs<br/>TTS + cloning]
  G --> M[ffmpeg merge<br/>loudness + crossfade] --> A
```

| Component | Stack | Role |
| --- | --- | --- |
| `frontend/` | React 19, TypeScript, Vite | One-screen workspace: drop a file, pick breaks, render |
| `backend/src/` | Node, Express, TypeScript (`tsx`, no build step) | Product API, ffmpeg muxing, ElevenLabs and OpenAI calls |
| `backend/ad_inserter/` | Python, librosa, ffmpeg | Audio analysis and the insertion/mixing pipeline |
| `ml/src/slotify_rank/` | Python, PyTorch, transformers, sentence-transformers | Features, model, training, evaluation, product inference |

Each candidate becomes one multimodal record: 110 handcrafted acoustic and
structural scalars, a frozen Whisper speech representation, and a frozen MiniLM
embedding of the transcript either side of the cut. A gated fusion model
combines them, giving an unavailable modality exactly zero weight rather than a
zero vector. Product inference reuses the training pipeline's feature code, and
before scoring the predictor checks the served features' column names and order,
and the normalizer's, against what the checkpoint was trained on.

## Model status

There are **no human labels yet**. The checkpoint the product serves,
`artifacts/training/gated-d8ed976101aa4c3b`, was trained on weak labels derived
from the heuristic baseline's own scores, with that score among its inputs — it
is a distillation of the heuristic, and it exists to prove the learned inference
path end to end. Its ranking is not independent evidence of placement quality,
and the evaluation refuses to report a headline for any model not trained on
human labels. The API says so in every learned
response (`provenance.trainingLabelSource` plus a warning), and the UI shows it.

Future weak-label runs drop the `heuristic_*` inputs their labels came from, and
the evaluation protocol (series-level bootstrap, a pre-registered 18 % threshold,
a frozen label snapshot) is enforced in code for when human labels exist. See
[`docs/model-training.md`](docs/model-training.md) and
[`docs/experiment-protocol.md`](docs/experiment-protocol.md).

## Tech stack

| Area | Used for |
| --- | --- |
| **PyTorch** | The ranking model, the trainer, checkpointing |
| **Hugging Face `transformers`** | `openai/whisper-tiny.en` encoder for speech representations |
| **`sentence-transformers`** | `all-MiniLM-L6-v2` transcript embeddings |
| **librosa** | Spectral and onset descriptors in the handcrafted block |
| **scikit-learn / NumPy / SciPy** | Feature assembly, metrics, bootstrap resampling |
| **TypeScript, React 19, Vite** | The product frontend |
| **Node, Express** | The product API |
| **OpenAI** | Sponsor copy generation |
| **ElevenLabs** | Text-to-speech and voice cloning |
| **ffmpeg / ffprobe** | Decoding, loudness matching, crossfades, muxing |

## Run it locally

Requires Node.js >= 22, Python 3.12, and `ffmpeg` + `ffprobe` on `PATH`.

```bash
git clone https://github.com/davidyang07/slotify
cd slotify

# 1) Node workspaces
npm run install:all

# 2) Python: the ML package (CPU-only) and the audio pipeline's requirements,
#    in one venv
cd ml && python -m venv .venv
.venv/Scripts/pip install -e ".[dev,label,features,sklearn]" -r ../backend/requirements.txt   # Windows
# .venv/bin/pip install -e ".[dev,label,features,sklearn]" -r ../backend/requirements.txt     # macOS / Linux
cd ..

# 3) Point the API at that interpreter
echo "PYTHON_BIN=$(pwd)/ml/.venv/Scripts/python.exe" >> backend/.env   # Windows (Git Bash)
# echo "PYTHON_BIN=$(pwd)/ml/.venv/bin/python" >> backend/.env       # macOS / Linux

# 4) Download the pinned Whisper and MiniLM snapshots once (~240 MB);
#    the learned ranker then runs offline
(cd ml && .venv/Scripts/python -m slotify_rank.cli infer warm)   # Windows
# (cd ml && .venv/bin/python -m slotify_rank.cli infer warm)     # macOS / Linux

# 5) Check your environment
npm run preflight

# 6) Run it
npm run demo
```

The learned ranker runs as one persistent worker process, so models load once:
the first ranking after startup takes a few seconds longer, the rest about 2 s
for a short clip. `SLOTIFY_RANKER_PRELOAD=1` warms it at startup;
`SLOTIFY_RANKER_WORKER=0` falls back to one process per request.

Then open <http://localhost:5173> and drop in an audio file; analysis starts on drop.
In development the UI calls the API through Vite's proxy, so it is same-origin
and needs no CORS setup.

Optional keys in `backend/.env`:

| Variable | Unlocks |
| --- | --- |
| `OPENAI_API_KEY` | Sponsor copy, tone and language, and slot narration |
| `ELEVENLABS_API_KEY` | Voice cloning and TTS |
| `HUGGINGFACE_TOKEN` | `pyannote` diarization for two-speaker mode |

### Before exposing the API

The API is open by default, for the local demo. Anything that can reach it can
spend your OpenAI and ElevenLabs credit, so before it listens anywhere but your
own machine:

| Variable | Effect |
| --- | --- |
| `SLOTIFY_API_TOKEN` | When set, every route except `GET`/`HEAD` `/api/health` and `/api/capabilities` requires `Authorization: Bearer <token>`. No child process ever receives it |
| `VITE_API_TOKEN` (frontend) | The token the UI sends. It is compiled into the bundle, so anyone who can load the page can read it: it keeps out clients that don't have the page, not users who do |
| `SLOTIFY_RATE_LIMIT_PAID_PER_MIN` | Per-IP limit on routes that call OpenAI or ElevenLabs (default 10; `0` disables) |
| `SLOTIFY_RATE_LIMIT_COMPUTE_PER_MIN` | Per-IP limit on analysis and merge (default 30) |
| `SLOTIFY_MAX_UPLOAD_MB` | Upload size cap (default 200). Uploads stream to a per-request temp dir, not memory |
| `SLOTIFY_MAX_AUDIO_MINUTES` | Longest upload, by duration, any route processes (default 180; longer is a 413) |
| `SLOTIFY_MAX_SONG_MINUTES` | Longest upload song-mode analysis accepts (default 10): beat tracking needs ~180 MB of memory per minute |
| `SLOTIFY_MAX_CONCURRENT_JOBS` | Analyses, rankings, ad pipelines and merges running at once (default half the CPUs, at least 1) |
| `SLOTIFY_MAX_QUEUED_JOBS` | Heavy jobs allowed to wait for a slot (default 8); beyond that, 503 with `Retry-After` |
| `SLOTIFY_TTS_MAX_STATEMENTS`, `SLOTIFY_TTS_MAX_CHARS` | Per-request `/api/tts` caps, checked before any ElevenLabs call (defaults 8 statements, 1,500 characters) |
| `SLOTIFY_MAX_CLONE_SAMPLES`, `SLOTIFY_MAX_CLONE_SAMPLE_MB` | Voice-clone upload caps (defaults 3 files, each up to `SLOTIFY_MAX_UPLOAD_MB`) |
| `SLOTIFY_CLONE_SAMPLE_SECONDS` | Seconds of each clone sample sent to ElevenLabs (default 180), past the first 30 s when the file is long enough, as mono MP3 |
| `SLOTIFY_VOICE_TTL_MIN` | Cloned voices idle this long are deleted from the ElevenLabs account (default 30) |
| `SLOTIFY_MAX_LIVE_VOICES` | Cloned voices kept at once; the least recently used is deleted first (default 3) |
| `SLOTIFY_OPENAI_MODELS` | Comma-separated OpenAI models `/ad/insert` clients may request as `llmModel` (default `gpt-4o-mini`) |
| `SLOTIFY_FFMPEG_TIMEOUT_MS`, `SLOTIFY_ANALYSER_TIMEOUT_MS`, `SLOTIFY_PIPELINE_TIMEOUT_MS` | Stop a stuck child process (defaults 2, 5 and 10 minutes) |
| `SLOTIFY_KILL_GRACE_MS` | How long a stopped child gets between SIGTERM and SIGKILL to clean up (default 5000) |

`GET /api/health` is a readiness check: 503, naming the failed check, when
ffmpeg, ffprobe or the analyser's Python imports are unusable. Every response
carries an `X-Request-Id` (yours, if you send one), error bodies repeat it as
`requestId`, and every log line for the request is prefixed with it. On
SIGTERM or SIGINT the API stops taking connections, stops its child processes,
deletes the voices it cloned and exits.

Rate limits key on the client IP Express sees; behind a reverse proxy, that is
the proxy's unless Express's `trust proxy` is configured.

### Ranker modes

```bash
RANKER_MODE=auto       # default: the learned model when a checkpoint is configured,
                       # falling back to the heuristic (with a warning) if it fails
RANKER_MODE=learned    # require the model; fail loudly if it is unusable
RANKER_MODE=heuristic  # the frozen signal-based scorer, ~7 s per episode
```

### Scoring a file from the CLI

```bash
cd ml
python -m slotify_rank.cli infer rank \
  --audio ../backend/audio_tests/ryan-test1.mp3 \
  --checkpoint ../artifacts/training/gated-d8ed976101aa4c3b/best_checkpoint.pt \
  --top 3
```

The output's `warnings` state that this checkpoint is a weak-label distillation
of the heuristic (see [Model status](#model-status)).

### Tests

```bash
npm run verify:all                      # frontend, backend, scripts and the ML suite
python -m pytest backend/tests/python   # the audio pipeline (needs backend/requirements.txt)
```

## Where things are

```text
.
├── backend/
│   ├── src/             # Express API: routes/, services/, lib/, middleware/
│   ├── ad_inserter/     # Python audio analysis + insertion pipeline
│   └── audio_tests/     # Sample audio for manual and CLI testing
├── frontend/src/        # React UI
├── ml/src/slotify_rank/ # Features, embeddings, models, training, inference, evaluation
├── config/              # The frozen signal-based scorer, shared by TS and Python
├── docs/                # Pipeline, dataset, training and demo documentation
└── scripts/             # preflight, demo, verify
```

## Documentation

| Document | What it covers |
| --- | --- |
| [`docs/feature-pipeline.md`](docs/feature-pipeline.md) | The multimodal feature pipeline |
| [`docs/model-training.md`](docs/model-training.md) | The training system |
| [`docs/model-inference.md`](docs/model-inference.md) | How a request becomes a learned ranking |
| [`docs/ad-inserter.md`](docs/ad-inserter.md) | The insertion pipeline, two-speaker modes and voice cloning |
| [`docs/demo-runbook.md`](docs/demo-runbook.md) | The 2-3 minute demo walkthrough |
| [`DESIGN.md`](DESIGN.md) | The UI design system: tokens, components and rules (read by AI coding agents too) |
| [`ml/README.md`](ml/README.md) | Every ML command |

## License

MIT. See [`LICENSE`](LICENSE).
