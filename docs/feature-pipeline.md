# Phase 3 — the multimodal feature pipeline

Turns normalized episodes and generated candidates into the complete feature set
the future PyTorch ranker consumes. Everything runs locally on CPU. No paid API
is involved, no key is required, and the only network access is the one-time
Hugging Face model download.

**Phase 3 trains nothing.** It produces features. Any claim about model quality
belongs to a later phase.

---

## 1. Setup

Python **3.12.13** is the pinned ML interpreter. The system Python 3.14 must not
be used — several of the dependencies below have no 3.14 wheels.

```powershell
cd ml
# OneDrive-synced checkouts reject hardlinks (os error 396), so uv must copy.
$env:UV_LINK_MODE = "copy"
uv venv --python 3.12.13 .venv
uv pip install --python .\.venv\Scripts\python.exe -e ".[dev,label,features]"
```

`features` is an optional extra and is deliberately heavy (~1 GB of wheels), so
the Phase 1 parity checks and the Phase 2 dataset pipeline keep installing in
seconds without it.

### Resolved versions

Verified under Python 3.12.13 on Windows 11:

| Package | Version |
|---|---|
| torch | 2.13.0+cpu |
| transformers | 4.57.6 |
| sentence-transformers | 5.6.0 |
| librosa | 0.11.0 |
| numba | 0.66.0 |
| soundfile | 0.14.0 |
| scipy | 1.18.0 |
| numpy | 2.4.6 |

Two constraints are load-bearing:

* **`numba>=0.60`.** librosa pulls numba transitively and the resolver will
  otherwise backtrack to numba 0.53, which supports Python <3.10 only and fails
  the build with a confusing error attributed to `llvmlite`.
* **No alternate torch index.** On Windows the default PyPI `torch` wheel *is*
  the CPU build. Adding the CUDA index would download ~2.5 GB to run on a
  machine with no NVIDIA GPU.

### Compute

The development machine has Intel Iris Xe integrated graphics, no NVIDIA GPU and
no usable `nvidia-smi`. Everything is sized for CPU:

* `device: auto` resolves to CUDA when torch reports a usable device and CPU
  otherwise. An explicit `cuda` request on a machine without CUDA is an **error**,
  not a silent downgrade — otherwise an eight-hour run looks normal.
* `float16` is rejected on CPU: there are no accelerated kernels, so it is
  emulated and slower than `float32`.
* DirectML is deliberately not used. It would be the only route to the Iris Xe,
  and it would add a large Windows-only dependency to accelerate two models that
  already run acceptably.
* Each stage loads its model once, processes every episode, then releases it
  before the next stage loads its own. Peak memory is one model, not two.

**Measured CPU throughput** (19.1 s spoken-word fixture, 10 threads):

| Stage | Time | Rate |
|---|---|---|
| Transcription (whisper-tiny.en) | 7.0 s | ~2.7x real time |
| Audio encoder (whisper-tiny.en) | 1.3 s | ~15x real time |
| MiniLM (2 texts) | 0.4 s | — |
| Acoustic features | ~2 s/episode | — |

Extrapolating, one hour of audio costs roughly **22 minutes** of transcription
and **4 minutes** of audio encoding on this machine. Transcription dominates; it
is also the stage that caches most usefully, since it is invalidated only by the
audio or the model changing.

---

## 2. Artifacts

All git-ignored, under the data root (`SLOTIFY_DATA_ROOT`, default `<repo>/data`):

```text
data/transcripts/<episode>.transcript.json        timestamped transcript
data/features/handcrafted/<episode>.handcrafted.json   scalars + context text
data/features/audio/<episode>.audio.npy (+ .json)      pooled Whisper vectors
data/features/text/<episode>.text.npy   (+ .json)      MiniLM vectors
data/features/state/<stage>.json                       resumability ledger
data/manifests/features.jsonl                          the feature manifest
artifacts/features/*.json, feature_summary.md          reports (committed)
```

Embeddings are `.npy` plus a JSON sidecar. **Never pickle**: `numpy.save` with
`allow_pickle=False` is a stable format that loads without executing anything
from the file, and every load in the codebase passes that flag explicitly. The
sidecar carries the cache identity, the model id and revision, the row ids and
the dimension, so a stale array is detectable *before* it is used.

Rows resolve **by id** (`<candidate_id>#<kind>`), never by position. Implied
ordering is the classic way these files go wrong: the array and the manifest
drift by one row and every candidate is described by its neighbour's audio.

---

## 3. Transcription

`openai/whisper-tiny.en` via Transformers. Chosen over `openai-whisper` because
`from_pretrained` gives a pinnable revision (a cache identity needs one) and
because the encoder used for speech representations is the same object loaded
the same way — so the transcript and the audio embeddings provably come from the
same weights.

**Chunking.** 30 s windows with 5 s overlap. 30 s is Whisper's receptive field;
larger buys nothing (the model pads or truncates regardless) and much smaller
costs context. 5 s is about one spoken clause — long enough that a word
straddling a boundary is seen whole by at least one chunk. Peak memory tracks
chunk length, so a three-hour episode costs the same as a thirty-second one.

**Overlap reconciliation.** The overlap region is transcribed twice. Duplicates
are resolved by preferring the chunk whose *centre* is nearer: a boundary word
is better transcribed by the window that saw it with context on both sides.
Segments that overlap in time but say genuinely different things are both kept
and de-overlapped — silently discarding one side would delete speech.

**Decoding is greedy at temperature 0.** A non-zero temperature is rejected at
config load: sampled decoding is not reproducible, which would make a cached
transcript untrustworthy.

Timestamps are integer milliseconds throughout. Segments are clipped to the
audio (Whisper pads to 30 s and will emit timestamps inside the padding), and a
segment ending past the episode is a hard error — it is the signature of a chunk
offset applied twice, which otherwise reads perfectly.

**An empty transcript is a failure, not an empty success.** Storing one would
mark every candidate on that episode "no text" and quietly shrink the dataset.

---

## 4. Handcrafted features

110 named scalars per candidate, raw and unnormalized. **No train-set
normalization is fitted here** — doing so would leak validation and test
statistics into the features themselves. Standardization belongs in the training
pipeline, computed on the train split alone.

Every value is paired with a **missing mask**. A masked entry is `0.0` and
carries no information; consumers must read the mask.

### Windows

Three scales, because a breakpoint is a multi-scale phenomenon:

| Scale | Half-width | Captures |
|---|---|---|
| `short` | 1 s | the pause itself |
| `medium` | 3 s | the phrase either side |
| `context` | 10 s | whether the topic changed |

Windows are **clipped, never padded**. A candidate 2 s into an episode has no
10 s "before" context; the window is clipped and the surviving fraction is
recorded as `window_available_before_*`. Zero-padding the audio instead would
manufacture silence and make every early candidate look like it sits in a long
pause — a systematic bias towards the start of every episode.

### Acoustic family

Computed once per episode at a 25 ms frame / 10 ms hop (the standard speech
framing, and the same hop Whisper uses) and sliced per candidate. Naming is
`<family>_<statistic>_<window>`.

| Feature | Units | Break-like when |
|---|---|---|
| `rms_mean_before_*` / `rms_mean_after_*` | dBFS | lower |
| `rms_min_around_*` | dBFS | lower (a real trough) |
| `rms_max_around_*` | dBFS | lower |
| `rms_var_around_*` | dB² | lower (steady, not mid-word) |
| `rms_delta_across_*` | dB | near zero (symmetric) |
| `rms_slope_before_*` | dB/s | negative (decaying into it) |
| `rms_slope_after_*` | dB/s | positive (rising out of it) |
| `energy_discontinuity_*` | dB | higher (a real seam) |
| `silence_duration_ms` | ms | higher |
| `ms_since_prior_speech` / `ms_until_next_speech` | ms | higher |
| `zcr_mean_around_*` / `zcr_var_around_*` | ratio | lower variance |
| `spectral_centroid_mean_*` | Hz | lower (voice absent) |
| `spectral_centroid_delta_across_*` | Hz | larger magnitude |
| `spectral_bandwidth_mean_around_*` | Hz | lower |
| `spectral_rolloff_mean_around_*` | Hz | lower |
| `spectral_contrast_mean_around_*` | dB | lower |
| `onset_strength_mean_around_*` / `_max_around_*` | a.u. | lower |
| `dbfs_mean_before_short` / `_after_short` | dBFS | lower |

**Missing-value rule**: a window clipped to fewer than two analysis frames
cannot support a variance or a slope, so those are masked rather than reported
as a fabricated zero. A difference across the candidate is masked unless *both*
sides exist — a difference against an absent side is not a small difference.

**Tempo is deliberately excluded.** On spoken word `librosa.beat.tempo` returns
an unstable estimate driven by whatever periodicity the onset envelope happens
to contain, and it shifts materially with window length. It is a real feature
for the product's music path and noise here. Features were not added to reach a
target count.

### Structural family

Position (`normalized_episode_position`, `ms_from_episode_start`,
`ms_to_episode_end`, `edge_guard_violation`), generator provenance
(`source_silence`, `source_pause`, `source_rms_minimum`,
`source_transcript_segment_end`, `source_fixed_interval`,
`merged_source_count`, `merged_timestamp_count`), candidate timings, and the
Phase 1 heuristic's own opinion (`heuristic_component_{base,pause,mode,sentence,position,edge}`,
`heuristic_raw_total`, `heuristic_clamped`, `heuristic_total_score`).

The component scores are **not leakage**: they are computed from audio by
`heuristic_offline_v1` with no access to a label. What *is* excluded, and tested
for, is anything label-derived — no rating, no acceptability flag, no rank.

### Text family

`words_before/after`, `chars_before/after`, `prior_segment_duration_ms`,
`following_segment_duration_ms`, `transcript_gap_before_ms/after_ms`,
`transcript_inter_segment_pause_ms`, `transcript_sentence_end`,
`terminal_punctuation_ordinal`, plus `context_cosine_similarity` and
`semantic_change_score` (`1 - cosine`), which are computed at assembly because
they need the MiniLM vectors.

Availability indicators (`transcript_available`, `text_before_available`,
`text_after_available`) are **never masked** — a masked availability flag would
be circular.

---

## 5. Whisper speech representations

The frozen encoder of `openai/whisper-tiny.en`.

**Raw encoder hidden dimension: 384.** Not 768 — that is whisper-*base*'s width.
The dimension is read from the loaded model config, not from this document.

### The efficiency decision

Running the encoder once per candidate would re-encode overlapping 30 s windows
dozens of times per episode. Instead:

1. Each episode is cut into 30 s chunks with 5 s overlap.
2. The encoder runs **once per chunk**.
3. Time-indexed encoder states are kept with enough metadata to map frames back
   to episode-relative time.
4. Each candidate's windows are *pooled* out of that cache.

Cost becomes proportional to episode length, not candidate count.

### Temporal resolution — derived, not assumed

```text
seconds_per_encoder_frame = hop_length / sampling_rate x conv_stride
                          = 160 / 16000 x 2
                          = 0.02 s  (20 ms)
```

The Whisper feature extractor emits a log-mel frame every `hop_length /
sampling_rate` seconds (10 ms); the encoder's second convolution has stride 2, so
two mel frames become one encoder frame. `derive_frame_duration_ms` computes
this from the values the loaded model reports **and** cross-checks it against the
encoder's real output length (1500 frames for a 30 s window), raising if the two
disagree. A hardcoded "0.2 s resolution" would be wrong by a factor of ten and
would still produce plausible-looking embeddings.

### Padding is the trap

Whisper pads every input to a full 30 s, so the encoder emits 1500 frames whether
it saw 30 s of speech or 2 s. Frames past the end of the real audio describe
silence the model invented. Pooling is bounded to the valid region; without that,
a candidate near an episode end gets an embedding dominated by padding.

### Per candidate

| Name | Window | Dimension |
|---|---|---|
| `before` | medium (3 s before) | 384 |
| `after` | medium (3 s after) | 384 |
| `context` | context (±10 s) | 384 |
| `difference` | `after - before` | 384 |

`difference` is exactly derivable from the other two and therefore redundant for
a model that sees both; it is kept because it is cheap and makes the
discontinuity directly visible to a linear probe.

**No projection to 128 dimensions.** Projection is a learned layer and belongs
inside the ranker, where it can be trained. Doing it now with an untrained matrix
would destroy information for no benefit. **The model is never fine-tuned** — it
runs under `torch.no_grad` in `eval` mode, with the decoder discarded.

---

## 6. MiniLM transcript embeddings

`sentence-transformers/all-MiniLM-L6-v2`. **Native output dimension: 384.**

The constructed transcript vector is **1536**:

```text
[ before (384) | after (384) | |before - after| (384) | before * after (384) ]
```

**1536 is arithmetic, not a model property.** MiniLM produces 384; four
concatenated blocks produce 1536. The manifest records the native dimension
(`text_embedding_dimension = 384`) and the statistics record the constructed
width separately, so the two are never conflated.

The absolute difference and elementwise product are the standard sentence-pair
interaction features (InferSent, SBERT's classification head): the difference
captures *how much* the topic moved, the product captures *where* the contexts
agree. A model given only the concatenation cannot recover them — they are not
linear functions of the inputs.

### Context construction

For each candidate:

1. Find the transcript segments ending at or before the timestamp, and starting
   at or after it.
2. Take the nearest ones, bounded by `max_chars_*` (600), `max_segments_*` (4)
   and `max_gap_ms` (15 s).
3. Reassemble chronologically so the text reads correctly; truncate from the
   left so the words nearest the candidate survive.

Three rules matter:

* **Bounded gaps.** A candidate in a 90-second musical interlude has no
  "sentence before"; the nearest segment is a minute away and about something
  else. Beyond `max_gap_ms` the context is masked as absent rather than pulled in
  and labelled adjacent.
* **A candidate landing mid-segment belongs to neither side.** That is the
  clearest possible evidence of a bad breakpoint; splitting the sentence across
  both sides would disguise it as a clean boundary.
* **A missing side yields no constructed vector at all.** Zero-filling would make
  the product block all-zero and the difference block equal the present side,
  which a model reads as a strong signal rather than as absence.

Absence reasons are recorded explicitly: `no_transcript`, `episode_start`,
`episode_end`, `mid_segment`, `gap_too_large`, `empty_segments`.

---

## 7. Cache identity

An artifact is reusable only when **every input that could have changed its bytes
is unchanged**. Timestamps are never part of that judgement — a newer file is not
a correct file, and mtimes survive neither a OneDrive sync nor a fresh clone.

A `CacheIdentity` is a sorted mapping reduced to one SHA-256. Per stage:

| Stage | Identity inputs |
|---|---|
| transcript | episode id, normalized audio SHA-256, model id + revision, transcription config digest, transcription version, pipeline version, library versions |
| handcrafted | + candidate generation version, feature config digest, **feature spec version**, transcript identity |
| audio embedding | episode id, audio SHA-256, model id + revision + config digest, window config, pooling config, pipeline version, library versions |
| text embedding | + transcript-context config, **handcrafted identity** |

`normalized_audio_sha256` rather than the source checksum: the models read the
16 kHz render, so re-normalizing must invalidate the artifacts even when the
original file is untouched.

Note the chaining — text embeddings depend on the handcrafted identity because
the context strings come from that stage. Changing the feature spec therefore
correctly invalidates the acoustic **and** text stages while leaving the
transcription and audio-embedding caches alone.

**Pipeline settings are deliberately excluded.** Worker counts and retry limits
change how long the work takes, not what it produces; folding them in would
invalidate a corpus because someone passed `--workers 2`.

### Stage states

| State | Meaning | Behaviour |
|---|---|---|
| `missing` | never attempted | run it |
| `partial` | started and interrupted | outputs untrusted, rerun |
| `complete` | finished under a recorded identity | skip |
| `failed` | finished with a recorded error | skip until `--retry-failed` |
| `stale` | complete under a *different* identity | recompute |

`partial` is written before the work begins and cleared after, so a process that
dies mid-stage is detected on the next run. The ledger is a cache of *decisions*,
never the source of truth: deleting it costs a rescan, never correctness.

Writes are atomic throughout. `np.save` writes incrementally, so a crash
mid-write would otherwise leave a file with a valid header and truncated data —
which loads without error and yields silently wrong numbers.

---

## 8. Missing-modality policy

Real episodes have no transcript, incomplete transcripts, insufficient audio near
boundaries, or empty context on one side. The pipeline:

* records availability explicitly (`audio_embedding_available`,
  `text_embedding_available`, `transcript_available`);
* **never substitutes random values**, and uses deterministic zeros only when
  paired with a mask;
* distinguishes expected absence from processing failure (`feature_status` vs
  `failure_reason`);
* excludes only unrecoverable failures from training eligibility — partial
  records are usable training examples *with* a mask;
* reports every exclusion in the statistics.

| `feature_status` | Meaning |
|---|---|
| `complete` | both learned modalities present |
| `audio_only` | no usable transcript context |
| `text_only` | audio pooling failed or fell outside the episode |
| `handcrafted_only` | neither learned modality |
| `failed` | unrecoverable; excluded from training |

These are **never summed into one "has features" figure**.

---

## 9. Commands

Every command accepts `--data-root`, `--episode-id` (repeatable),
`--content-type`, `--split`, `--limit`, `--force` and `--retry-failed`.

```powershell
cd ml
$py = ".\.venv\Scripts\python.exe"

# Everything, in order, then validate and report.
& $py -m slotify_rank.cli pipeline features `
    --transcription-config configs/transcription_v1.yaml `
    --features-config configs/features_v1.yaml `
    --embeddings-config configs/embeddings_v1.yaml `
    --deep

# Progress without loading a model.
& $py -m slotify_rank.cli pipeline status
```

Individual stages:

```powershell
& $py -m slotify_rank.cli transcribe run
& $py -m slotify_rank.cli transcribe validate
& $py -m slotify_rank.cli features acoustic
& $py -m slotify_rank.cli embeddings audio
& $py -m slotify_rank.cli embeddings text
& $py -m slotify_rank.cli features assemble
& $py -m slotify_rank.cli features validate --deep
& $py -m slotify_rank.cli features stats
```

**One episode:**

```powershell
& $py -m slotify_rank.cli pipeline features --episode-id <EPISODE_ID> --deep
```

**One split:**

```powershell
& $py -m slotify_rank.cli pipeline features --split train --deep
```

`--force` recomputes a valid cache. It does **not** bypass integrity checks:
`features validate` does not accept the flag at all, and the pipeline still exits
non-zero on a validation error.

---

## 10. Validation

`features validate` exits non-zero on any error. Checks: unsupported schema
version, missing or unsorted or duplicated feature names, stale feature spec,
wrong vector dimension, mask-length mismatch, NaN/infinite scalars, unknown
candidate or episode, **synthetic candidates included**, timestamp mismatch,
timestamp out of episode bounds, audio checksum drift, split disagreement,
availability flags without references, `complete` without both modalities. Under
`--deep`: unreadable arrays, missing embedding rows, dimension disagreement
between a reference and its array, non-finite embeddings.

The failure mode this guards against is not a crash — it is a corpus that looks
fine and is quietly wrong.

---

## 11. How this feeds the PyTorch dataset (Phase 4, built)

This is what the Phase 4 loader (`slotify_rank.datasets.loader`) actually does,
per candidate — full detail in [`docs/model-training.md`](model-training.md):

1. Read one line of `data/manifests/features.jsonl`.
2. Take `handcrafted_feature_values` (110 floats) and
   `handcrafted_missing_mask`, positional against the header's
   `handcrafted_feature_names`.
3. Resolve `audio_embedding_reference[kind]` → read the `.npy`, look the
   `row_id` up in the sidecar's `row_ids`, gather each 384-vector, and
   concatenate `before`, `after`, `context`, `difference` → 1536.
4. Resolve `text_embedding_reference[before|after]` the same way and build the
   1536-vector (`before`, `after`, and their `difference` and elementwise
   `product` — arithmetic on cached MiniLM output, **not** a model re-run), or
   emit zeros **with the availability flag set false**.
5. Standardize the handcrafted block using statistics fitted on the **train
   split only** (`slotify_rank.datasets.normalizer`).
6. A *learned* per-modality projection (128-wide) sits in each model variant, so
   the 384-native vectors are projected inside the ranker, never here.

`dataset_split` is on every record and validated against the split manifest, so
pair generation groups without re-deriving it. Nothing in this path imports
Whisper or MiniLM — the embeddings are read from the cached arrays this pipeline
already wrote.

---

## 12. Known limitations

* **`sentence_end` from the candidate record is 100% missing.** Phase 2 generated
  candidates before transcripts existed, so it is `None` everywhere. The
  transcript-derived `transcript_sentence_end` covers the same signal. Regenerating
  candidates now that transcripts exist would populate it.
* **`model_revision: main`** is honest but not reproducible. Pin a commit SHA
  before generating a corpus whose transcripts must be reproducible months later.
* **Handcrafted intermediates are JSON.** Fine at the current scale (~220 KB for
  20 candidates); at 100k candidates this should become a binary matrix.
* **Word timestamps are off by default.** tiny.en's are not reliable enough to
  justify roughly doubling transcription time, and nothing in Phase 3 needs them.
* **`workers` is accepted but the stages are sequential.** The heavy work is
  already internally parallel through BLAS/torch threading.
