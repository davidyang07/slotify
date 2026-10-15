# Model inference: how a request becomes a learned ranking

This traces one `POST /api/insert-sections` from the browser to the PyTorch
forward pass and back. It is the path the demo runs, and the one that closes the
gap between "an ML package exists in this repository" and "the product uses it".

---

## The request

```
React UI
  │  multipart: audio, count, mode, sponsors
  ▼
POST /api/insert-sections          backend/src/routes/insert-sections.ts
  │
  ├─ getAudioDuration                          services/ffmpeg.ts   (ffprobe)
  ├─ runPythonAnalyze                          services/python.ts
  │    └─ python -m ad_inserter.analyze_cli    → candidates, snippets
  ├─ mergeCandidates                           lib/candidates.ts
  │
  ▼
rankCandidates                                 services/ranker.ts
  │
  ├─ resolveRankerMode                         lib/ranker-mode.ts
  │
  ├─ mode = heuristic ──► rankWithHeuristic    lib/ranking.ts
  │
  └─ mode = learned ───► runLearnedRanker      services/learned-ranker.ts
         ├─ one long-lived worker (default):
         │    <ml python> -m slotify_rank.cli infer serve --checkpoint …
         │      ◄─► one JSON line per request / response
         └─ SLOTIFY_RANKER_WORKER=0: per request
              <ml python> -m slotify_rank.cli infer rank --audio … --work-dir …
                └─ slotify_rank.inference      (below)
  ▼
applySpacing → clampSlotMs → dedupeByMs → buildPlacementSignals
  ▼
JSON: placementStatus, slots[], provenance, warning
```

Two things about this shape are deliberate.

**One decision point.** Routes never choose a ranker. `rankCandidates` is the
only place the mode is resolved, so there is no code path in which heuristic
output can be labelled as model output. The `source` field on every slot comes
from whichever ranker actually ran.

**The two modes do not rank the same candidate set.** In heuristic mode the
ranked candidates are the ones `ad_inserter.analyze_cli` proposed. In learned
mode the ML package regenerates candidates itself, from the uploaded audio,
inside `slotify_rank.inference.episode.prepare_episode` (`generate_for_episode`
under the committed `ml/configs/dataset_v1.yaml`) — it has to, because a feature
vector needs the candidate's own acoustic and transcript context, and those are
built by the corpus pipeline, not by the analyser. The learned result is
authoritative: `services/ranker.ts` keeps the ML package's timestamps and only
joins the analyser's snippet and silence length back on, by exact timestamp or
else the nearest analyser candidate within 1.5 s. A learned slot with no analyser
candidate that close carries an empty snippet. So a learned-vs-heuristic
difference in the product mixes two things — a different candidate pool and a
different ordering — and is not a like-for-like ranking comparison. The
like-for-like comparison is `evaluation compare`, which scores both systems on
the same labelled candidates.

---

## Inside the Python side

`slotify-rank infer rank` — and each request to `infer serve`, which runs the
same routine (`inference/service.py::rank_audio`) — does five things, in
`slotify_rank/inference/`.

### 1. A throwaway single-episode corpus

`episode.py` copies the upload into a scratch directory, builds a `DataPaths`
rooted there, and runs **the real Phase 3 stages** over it: `normalize_episode`,
`generate_for_episode`, `run_transcribe`, `run_acoustic`, `run_audio_embeddings`,
`run_text_embeddings`, `assemble_episode`, `build_examples`.

The stages read their settings from the committed `ml/configs/*.yaml` — the same
files the corpus pipeline reads — unless `--config-dir` names another directory.
The backend never passes `--config-dir`, so this default is what the product
runs. (It used to be the dataclass defaults in code, which matched the YAML only
because nobody had yet edited one without the other.)

This is the single most important design decision in the inference path. The
obvious alternative — a lean, purpose-built feature function for serving — would
drift from the training layout the first time a feature was added, silently, and
the model would score mis-columned inputs. Mis-columned inputs do not crash; they
produce confident nonsense. Reusing the corpus code makes that bug
*inexpressible*: there is exactly one implementation of "what a candidate's
feature vector is".

The cost is a scratch directory per request and the models' load time. The
benefit is that training/serving skew cannot happen. The worker (below) pays the
load once per process instead of once per request; the scratch directory lives
inside a work directory the caller owns.

The episode is assigned the `development` split — never `train`, `validation` or
`test`, so an uploaded file can never look like an evaluation row.

### 2. Loading the checkpoint

`predictor.py::RankerPredictor.load`:

- `torch.load(..., weights_only=True, map_location="cpu")` — a checkpoint format
  that must be *trusted* before it can be read is a remote-code-execution
  primitive with a `.pt` extension, and model files get emailed around;
- reads the input schema and model config *out of the checkpoint* and rebuilds
  the architecture from them, so nothing depends on a YAML that may have moved
  on;
- `assert_compatible(..., allow_split_mismatch=True)` before a single weight is
  loaded. This compares the checkpoint **with itself and with its normalizer** —
  its declared schema against its stored feature ordering — not with the data
  about to be scored; that comparison happens at scoring time (step 3);
- requires the normalizer (beside the checkpoint, or `--normalizer` /
  `SLOTIFY_RANKER_NORMALIZER`) and checks its identity: its column names must be
  the checkpoint's, and its `training_split_hash` must be the one the checkpoint
  recorded. Scoring with statistics fitted on other data shifts every input
  silently, so a missing or foreign `normalizer.json` is a hard failure;
- translates a state-dict shape mismatch into `IncompatibleCheckpoint`, so a
  checkpoint whose declared schema disagrees with its own weights fails the same
  way as every other incompatible checkpoint;
- reads what the model was trained on — `label_source`, `data_provenance` — from
  the checkpoint. Checkpoints written before those keys existed (the committed
  bootstrap is one) fall back to their own run's `training_summary.json`, and
  only when its `run_id` matches; otherwise the source is `unknown`;
- `model.eval()`, so dropout is off.

### 3. Scoring

`predictor.py::rank_examples` first compares the **served** schema — the column
names and order the feature path just built for this audio
(`prepared.dataset.schema`) — with the checkpoint's input schema
(`align_examples`):

- a learned block of a different width, part order or input-schema version is
  refused with `IncompatibleCheckpoint`;
- a handcrafted column the checkpoint was trained on but the served features
  lack is refused — a zero standing in for a missing measurement looks like a
  real zero to the model;
- a served layout that reorders or extends the checkpoint's columns is
  projected onto them **by name**. This is how a weak-label checkpoint, trained
  without the `heuristic_*` columns (see `docs/model-training.md`), scores
  features that still carry them;
- a feature spec or pipeline version that differs from the checkpoint's is
  disclosed in `warnings`, not refused: names and widths are what the weights
  depend on, and a version bump elsewhere must not take the served model down.

It then normalizes the handcrafted block with the fitted statistics, stacks the
three blocks plus their availability flags, and runs one
`torch.inference_mode()` forward pass over the whole batch — one pass, so a batch
boundary cannot change a result.

Non-finite output after normalization is an error rather than something to
repair: it means the fitted statistics do not describe this audio's features, and
scoring through that produces numbers with no meaning.

### 4. Ranking

`score_examples` quantizes every raw score to 6 decimals **once**, then sorts by
descending quantized score, breaking ties on the earlier timestamp then the
candidate id, and normalizes and reports the same quantized values. Quantizing
first matters: a float32 forward pass does not give two identical candidates
bit-identical scores (copies of one vector in one batch have scored ~7e-9
apart), and sorting or min-max scaling the raw floats would let that noise pick
the order and stretch it into a 100-vs-0 `normalized_score` — which reaches the
UI as `placement_score`. Two runs over the same audio produce byte-identical
output.

`normalize_scores` maps the quantized scores onto 0–100 *within this episode*.
When every candidate scores the same (a spread under one quantum) it returns 50
for all of them rather than manufacturing a spread the model did not produce.

### 5. The response

`schema.py::InferenceResult` carries the ranking plus everything needed to
reproduce it: model variant, run id, the checkpoint's git commit, parameter
count, all four schema versions, the normalizer version, and — most importantly —
`training_label_source` and `training_data_provenance`.

That last pair is why every response says what the model learned from. The
response does not leave that to a field a client may not display: when the
checkpoint's `training_label_source` is `weak_heuristic`, `warnings` carries a
plain sentence (`inference/schema.py::WEAK_LABEL_DISCLOSURE`) stating that the
model is a distillation of the heuristic baseline, trained without any human
labels, so its ranking is not independent evidence. Synthetic and unknown
supervision carry their own sentence. `InferenceResult` adds it itself, so no
code path can build a response without it; a warning string does not change the
document's shape, so `schema_version` stays `inference-v1.0.0`. A model that
cannot say what it was trained on cannot be served honestly.

---

## Modes and fallback

| `RANKER_MODE` | Behaviour |
| --- | --- |
| `heuristic` | The frozen offline baseline. No Python ML process, no model, ~7 s. |
| `learned` | Requires a usable checkpoint. Throws at startup if there isn't one. |
| `auto` (default) | Learned when `SLOTIFY_RANKER_CHECKPOINT` points at a usable checkpoint and its normalizer; otherwise the heuristic, with the reason in the response. |

`auto` resolves to `heuristic` when no checkpoint is configured. Under
`npm run demo`, `scripts/lib/select-checkpoint.mjs` configures one: a
human-trained run if any exists, and otherwise the committed bootstrap
`gated-d8ed976101aa4c3b` — which is **weakly supervised**: trained on labels
derived from `heuristic_offline_v1`'s own scores, with the baseline's score among
its inputs, validated on a single episode. It proves the serving path, not
ranking quality, and every response it produces carries the distillation
disclosure above. That is the honest default. The alternative — quietly labelling heuristic output
as model output so the demo looks better — is the failure the whole module exists
to prevent, and `GET /api/capabilities` reports the resolution and its reason so
a client never has to guess.

---

## The persistent worker

`services/learned-ranker.ts` keeps **one** `python -m slotify_rank.cli infer
serve --checkpoint … --normalizer …` process (`slotify_rank/inference/worker.py`).
It loads torch, the checkpoint, its normalizer, Whisper, the Whisper encoder and
MiniLM once, keeps the three encoders resident (`embeddings/resident.py`; the
corpus pipeline still loads and frees one model per stage), and then answers
requests one at a time:

```
stdin   {"id": 7, "audio_path": "…/audio.mp3", "work_dir": "/tmp/slotify-rank-XXXX",
         "episode_id": "upload", "skip_transcription": false}
stdout  {"id": 7, "ok": true,  "payload": { …the infer-rank document… }}
        {"id": 7, "ok": false, "error": "<last meaningful line of the error>"}
```

`payload` is exactly the `infer rank` document, so `validateLearnedRanking`
applies to it unchanged; only `timings_seconds` differs (`model_load` is `0.0`:
this request did not load the model). Nothing but responses reaches stdout — the
worker moves file descriptor 1 onto stderr before importing anything that could
print, and keeps a private copy for the protocol. Unknown request keys are
refused, not ignored.

**Lifecycle (Node side).**

- Started on the first learned request, or at API startup with
  `SLOTIFY_RANKER_PRELOAD=1`. Bound to one interpreter, checkpoint and
  normalizer; a request for another checkpoint replaces it.
- Requests are queued and sent one at a time. `SLOTIFY_RANKER_TIMEOUT_MS`
  (default 600 000) runs from the moment a request is sent; the value is
  validated at startup — anything but a positive number of milliseconds stops
  the API rather than becoming a 1 ms timer.
- On a timeout, a crash, a protocol violation, or an abort of the *running*
  request (the client went away), the worker gets SIGTERM, then SIGKILL after
  `SLOTIFY_KILL_GRACE_MS`, and the next request starts a fresh one. An aborted
  *queued* request is dropped; the worker never sees it. A request is rejected
  only after its worker has exited, so its work directory is never removed
  under a running process.
- A request the worker answers with `ok: false` (an unreadable upload, an
  incompatible checkpoint) is an error for that request only; the worker keeps
  serving. An exception the package does not raise on purpose is answered and
  then the worker exits, so the next request gets a clean process.
- An `onShutdown` hook (`lib/lifecycle.ts`) stops the worker and refuses
  whatever is queued. The worker also exits on its own when its stdin closes,
  so it cannot outlive a crashed API.

**Errors keep their contract.** Every failure is a `LearnedRankerError` whose
`message` is client-safe (the last meaningful line, paths removed) and whose
`detail` is the worker's stderr for the server log. `services/ranker.ts` is
unchanged: `RANKER_MODE=learned` fails loudly, `auto` falls back to the
heuristic and says why.

**Scratch files.** Node creates a work directory per request (`mkdtemp`) and
removes it when the request settles; the worker builds the throwaway corpus in
a fresh subdirectory of it and points `tempfile` there for the request. Nothing
with the user's upload is written to the system temp directory, so a killed
worker leaves nothing behind that Node does not remove. `infer rank` takes the
same `--work-dir`, and turns SIGTERM into a clean unwind either way.

**Environment.** The child (worker or per-request CLI) gets an allow-list, never
`process.env`: `PATH`, `HOME`, `LANG`, `LC_*`, `TMPDIR`, `PYTHON*`, `HF_*`,
`TRANSFORMERS_*`, `SENTENCE_TRANSFORMERS_*`, `TORCH*`, `OMP_*`, `MKL_*`,
`SLOTIFY_RANKER_*`, plus the few paths the ML package reads (`XDG_CACHE_HOME`,
`HUGGINGFACE_HUB_CACHE`, `FFMPEG_BIN`, `FFPROBE_BIN`, `SLOTIFY_REPO_ROOT`) and,
on Windows, the system variables Python needs. Any name containing `TOKEN`,
`SECRET`, `PASSWORD`, `CREDENTIAL` or `API_KEY` is dropped even if a prefix
matches (`HF_TOKEN`), so `SLOTIFY_API_TOKEN`, `SLOTIFY_INTERNAL_TOKEN` and
provider keys never reach it. The ranker never calls back into the API.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SLOTIFY_RANKER_WORKER` | `1` | `0` runs `infer rank` once per request instead of the worker. |
| `SLOTIFY_RANKER_PRELOAD` | `0` | `1` starts the worker at API startup, not on first use. |
| `SLOTIFY_RANKER_TIMEOUT_MS` | `600000` | Per-request limit, from the moment the request is sent. Validated at startup. |
| `SLOTIFY_KILL_GRACE_MS` | `5000` | SIGTERM-to-SIGKILL grace when the worker is stopped (shared with every child). |
| `SLOTIFY_RANKER_SKIP_TRANSCRIPTION` | `0` | `1` skips Whisper per request (text modality masked, and the response says so). |

---

## Model snapshots: pinned, offline, warmed

The three encoders load through `from_pretrained` at the `model_revision` the
committed configs name. Those were `main` — "whatever the Hub serves today" —
which meant every load resolved `main` over the network (a slow or unreachable
Hub slowed or broke ranking) and a moved upstream `main` would have changed the
served weights while every config said the same thing. They are now commit SHAs:

| Model | Config | Revision |
| --- | --- | --- |
| `openai/whisper-tiny.en` | `transcription_v1.yaml`, `embeddings_v1.yaml` (audio) | `87c7102498dcde7456f24cfd30239ca606ed9063` |
| `sentence-transformers/all-MiniLM-L6-v2` | `embeddings_v1.yaml` (text) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |

Each is the commit `main` resolved to — on the Hub and in the local cache the
shipped checkpoint was served from — when it was pinned (Whisper's `main` has
not moved since 2024-01-22; MiniLM's last moved on 2026-06-01, a README metadata
commit, and the weights it loads date from 2024 and 2021). So pinning changed no
weights.

**Pinning and the shipped checkpoint.** Changing a revision string changes the
config digests and every cache identity that includes them (transcript, audio
and text embeddings, and through the transcript the handcrafted features). None
of those is recorded in, or compared against, the checkpoint or its normalizer:
`assert_compatible` and `align_examples` compare column names, block widths and
parts, the feature spec / pipeline / input-schema versions and the normalizer's
identity — code constants and extracted values, never a model revision. In
serving, the identities are only compared within the request's own fresh
workspace. `tests/test_inference_hub.py` checks this end to end: the served
schema and the ranking are identical under `main` and under the pinned SHA. In a
corpus under `data/` built with `main`, the next `pipeline features` run treats
those caches as stale and recomputes them — the same bytes, at the cost of the
time.

**Offline mode.** When every revision is a SHA whose snapshot (config and
weights) is in the local Hugging Face cache — `HF_HUB_CACHE`, `HF_HOME`, … or
`TRANSFORMERS_CACHE` / `SENTENCE_TRANSFORMERS_HOME` for their loaders — `infer
rank` and `infer serve` set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
before any Hugging Face library is imported (`inference/hub.py`). When a pinned
snapshot is missing, `infer serve` refuses to start with an error that names the
fix; the API reports it as the learned ranker's failure (auto mode then serves
the heuristic and says why). `infer rank` says the same on stderr and downloads
the pinned revision for that run, as it always has.

**`infer warm`.** Downloads exactly the pinned snapshots, through the same
loaders serving uses, so what lands in the cache is what an offline load asks
for. `--check` downloads nothing and exits 1 if anything is missing; that is the
step a preflight runs, and `infer warm` is the step a setup or a demo runs once.

```bash
cd ml
.venv/bin/python -m slotify_rank.cli infer warm          # fetch (network)
.venv/bin/python -m slotify_rank.cli infer warm --check  # verify (offline)
```

---

## Latency

`backend/audio_tests/ryan-test1.mp3` (22.9 s), CPU, warm Hugging Face cache, the
shipped checkpoint, three consecutive `POST /api/insert-sections` (wall time of
the whole request, analyser included):

| Path | 1st | 2nd | 3rd |
| --- | --- | --- | --- |
| Before: `infer rank` per request, revisions `main`, online | 15.4 s | 13.7 s | 12.9 s |
| Worker, started on first use | 8.3 s | 2.1 s | 2.7 s |
| Worker, `SLOTIFY_RANKER_PRELOAD=1` | 6.7 s | 2.0 s | 2.1 s |

Inside a warm worker the ranking itself takes ~1.4 s for that clip: normalize
0.25, transcription 0.9, acoustic 0.06 (1.2 on a worker's first request, while
librosa's numba kernels compile), audio embeddings 0.13, text embeddings 0.04,
forward pass 0.002. One-shot `infer rank` on the same clip went from 20.5 s to
9.3 s from the pinned revisions and offline mode alone. The rankings are byte-identical across all of these
paths.

The model is still a negligible share of the wall clock; what the worker removes
is the per-request torch import and the three encoder loads, which dominated
short clips. On long audio, transcription dominates again; the levers there are
a smaller Whisper, `SLOTIFY_RANKER_SKIP_TRANSCRIPTION=1` / `--no-transcribe`
(which masks the text modality — and says so in the response), or a longer
`SLOTIFY_RANKER_TIMEOUT_MS`.
