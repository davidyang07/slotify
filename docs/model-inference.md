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
         └─ <ml python> -m slotify_rank.cli infer rank --audio … --checkpoint …
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

`slotify-rank infer rank` does five things, in `slotify_rank/inference/`.

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

The cost is a scratch directory per request and a cold model load per process.
The benefit is that training/serving skew cannot happen.

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

## Latency

Measured on CPU for a 19-second clip:

| Stage | Seconds |
| --- | --- |
| ffmpeg normalize | 4.0 |
| candidate generation | 0.1 |
| Whisper transcription | 25.6 |
| acoustic + structural + text scalars | 8.0 |
| Whisper encoder embeddings | 1.0 |
| MiniLM embeddings | 3.4 |
| assemble | 0.2 |
| **model load** | **0.14** |
| **forward pass** | **0.005** |

The model is 0.03 % of the wall clock. Transcription is 60 %. That is why the
integration is a subprocess rather than a persistent model service: a service
would optimise the two fastest stages. If latency needs to come down, the levers
are a smaller Whisper, a cached transcript, or `--no-transcribe` (which masks the
text modality — and says so in the response, rather than pretending it had a
transcript).

---

## What to change if you want a persistent service

`RankerPredictor` is already built to be constructed once and reused, and holds
no request state. A FastAPI or long-running worker wrapping it would be a new
entry point next to `inference_cli.py`, not a rewrite of anything below it. The
Express side would swap `runLearnedRanker`'s `spawn` for an HTTP call; the
`parseLearnedRanking` validation, and everything it refuses, stays as is.
