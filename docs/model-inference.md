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

**Generation is not ranking.** `ad_inserter.analyze_cli` proposes candidates in
both modes. Only the ordering differs. That keeps the comparison against the
baseline honest — both systems rank the same candidate set — and it means a
learned-ranker failure degrades to a worse ordering, never to no candidates.

---

## Inside the Python side

`slotify-rank infer rank` does five things, in `slotify_rank/inference/`.

### 1. A throwaway single-episode corpus

`episode.py` copies the upload into a scratch directory, builds a `DataPaths`
rooted there, and runs **the real Phase 3 stages** over it: `normalize_episode`,
`generate_for_episode`, `run_transcribe`, `run_acoustic`, `run_audio_embeddings`,
`run_text_embeddings`, `assemble_episode`, `build_examples`.

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
  loaded — scoring a new upload legitimately has a different split hash, but the
  feature ordering, dimensions and pipeline versions must match exactly;
- requires the normalizer beside the checkpoint. Scoring with statistics fitted
  on other data shifts every input silently, so a missing `normalizer.json` is a
  hard failure, not a default;
- translates a state-dict shape mismatch into `IncompatibleCheckpoint`, so a
  checkpoint whose declared schema disagrees with its own weights fails the same
  way as every other incompatible checkpoint;
- `model.eval()`, so dropout is off.

### 3. Scoring

`predictor.py::score` normalizes the handcrafted block with the fitted
statistics, stacks the three blocks plus their availability flags, and runs one
`torch.inference_mode()` forward pass over the whole batch — one pass, so a batch
boundary cannot change a result.

Non-finite output after normalization is an error rather than something to
repair: it means the fitted statistics do not describe this audio's features, and
scoring through that produces numbers with no meaning.

### 4. Ranking

`score_examples` sorts by descending ranking score, breaking ties on the earlier
timestamp then the candidate id. Two runs over the same audio therefore produce
byte-identical output.

`normalize_scores` maps the raw scores onto 0–100 *within this episode*. When
every candidate scores the same it returns 50 for all of them rather than
manufacturing a spread the model did not produce.

### 5. The response

`schema.py::InferenceResult` carries the ranking plus everything needed to
reproduce it: model variant, run id, the checkpoint's git commit, parameter
count, all four schema versions, the normalizer version, and — most importantly —
`training_label_source` and `training_data_provenance`.

That last pair is why the API can attach "this model was trained on weak labels,
its ordering is not evidence of ranking quality" to every response the bootstrap
checkpoint produces. A model that cannot say what it was trained on cannot be
served honestly.

---

## Modes and fallback

| `RANKER_MODE` | Behaviour |
| --- | --- |
| `heuristic` | The frozen offline baseline. No Python ML process, no model, ~7 s. |
| `learned` | Requires a usable checkpoint. Throws at startup if there isn't one. |
| `auto` (default) | Learned when `SLOTIFY_RANKER_CHECKPOINT` points at a usable checkpoint and its normalizer; otherwise the heuristic, with the reason in the response. |

`auto` resolves to `heuristic` on a fresh clone with no checkpoint configured.
That is the honest default. The alternative — quietly labelling heuristic output
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
