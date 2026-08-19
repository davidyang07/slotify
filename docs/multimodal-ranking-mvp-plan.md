# Multimodal Audio–Text Ranking Engine — MVP Plan

Status: **Phases 0–4 implemented and verified.** Phase 1 (reproducible heuristic
baseline), Phase 2 (dataset foundation), Phase 3 (multimodal feature pipeline)
and Phase 4 (PyTorch ranking training system) exist and are tested. Phases 2–4
have been exercised on smoke/synthetic data only — no real corpus is acquired
and no model-quality claim is made. Phase 5 (baseline-vs-model comparison,
ablations, human evaluation) is not started. See §19–24 for the training design
and [`docs/model-training.md`](model-training.md) for the built system.
Author: repository audit performed against commit `ef614ba` on branch `main`.
Date: 2026-07-21 (Phase 0), updated 2026-07-22 (locked decisions + Phase 1).

---

## 0. Locked decisions

These were approved after the Phase 0 audit and override anything below that
contradicts them.

### Compute and environment
* Laptop-only, **CPU-first**. No NVIDIA GPU is assumed anywhere.
* No DirectML, ROCm, or cloud training is introduced during Phase 1.
* The ML environment is pinned to **Python 3.12.13** in `ml/.venv`, created and
  managed independently of the backend's Python. The backend continues to run on
  its existing interpreter (verified: Python 3.14.2) and is unaffected.
* Windows setup commands are documented in `ml/README.md`.

### Dataset scope (targets, not results)
Final portfolio targets: **≥ 50 h** processed spoken-word audio, **≥ 10 000**
generated candidates, **≈ 8–12 h** manually reviewed subset, **≈ 1 500** manually
reviewed candidates.

The 50-hour corpus is **not** described as human-labelled anywhere. Five
quantities are tracked and reported separately:

| Quantity | Meaning |
|---|---|
| `processed_audio_hours` | audio decoded, transcribed and featurised |
| `unlabelled_candidate_count` | generated, never reviewed |
| `weakly_labelled_candidate_count` | labelled by heuristic or metadata, never by a person |
| `human_labelled_candidate_count` | reviewed by a person against the rubric |
| `held_out_evaluation_candidate_count` | human-labelled **and** in the test split |

Staged labelling (each stage is a checkpoint, not a separate pipeline):
1. **200–300** candidates — pipeline debugging.
2. **≈ 750** candidates — first trained model.
3. **≈ 1 500** candidates — final model selection and evaluation.

Phase 1 establishes only the schemas and interfaces needed for later
compatibility. No dataset is acquired or labelled.

### Corpus composition
The primary domain is **podcasts, interviews, narrated spoken word, and
conversational spoken-word recordings**. AMI is demoted from Phase 0's proposal:
it may be included later as a **supplemental** source for speaker changes and
conversational transitions, but it does not represent the target domain. The
**held-out test set must be real podcast or podcast-like content**, so the
reported ranking metrics describe the product's actual use case.

### Evaluation participants (Phase 6, not Phase 1)
* Blind human-preference evaluation requires **at least one evaluator other than
  the project author**; two additional evaluators preferred.
* The editing-time benchmark requires **at least three participants**; five where
  practical.

### Baseline policy
The deterministic offline heuristic is the **headline baseline** because it is
reproducible, free, credential-free and stable across runs. Reports distinguish
three families:

| Name | What it is | Requires credentials? |
|---|---|---|
| `heuristic_offline_v1` | **canonical baseline.** Silence candidates + the product scorer, selection and finalisation, run offline | no |
| `product_api_baseline_v1` | the same path *plus* the paid `whisper-1` transcript candidates and `gpt-4o-mini` enrichment | yes |
| trained model variants | Phases 4–5 | no |

**The paid path is never required to reproduce the headline comparison.**

### Frontend fallback policy
`frontend/src/App.tsx:453-482` invents three slots at ≈22 %, 48 % and 72 % of the
episode with hard-coded confidences `[92, 85, 78]` whenever fewer than three
suggestions arrive. This is **recorded as product technical debt** (§5, risk R6).
It must **never** enter model or heuristic evaluation — evaluation operates below
the UI layer, on `ml/`'s own pipeline, which never reads the frontend. Phase 7
replaces it with either clearly-labelled unscored suggestions or an explicit
insufficient-candidates state. It was **not** changed in Phase 1: it did not
block the golden fixture, since the fixture is generated from the scorer module
directly.

### Evidence policy
Intended final results: a trained multimodal PyTorch ranker; improved NDCG@3 over
a deterministic signal-based baseline; a processed dataset spanning 50+ hours and
10 000+ candidates; evaluation against a held-out human-labelled subset; ranked
breakpoints served through FastAPI into the existing React app.

No claim asserts that all 50+ hours or all 10 000+ candidates are manually
labelled. No positive improvement, human-agreement percentage or editing-time
reduction is assumed — every such value comes from a generated artifact.

---

## 1. Executive summary

Slotify today picks ad-insertion points with a hand-written scoring function that lives in
**TypeScript** (`backend/src/lib/candidates.ts:34` `scoreCandidate`) fed by a **Python** silence
detector (`backend/ad_inserter/analyze_cli.py:14` `_detect_podcast_silences`) and, when an OpenAI
key is present, Whisper-API segment ends (`backend/src/routes/insert-sections.ts:80-110`). There is
no dataset, no evaluation, no test suite, and no ML model. The frontend fabricates three fallback
slots with hard-coded confidences (`frontend/src/App.tsx:453-482`) whenever the backend returns
fewer than three points, so today's "AI-recommended timestamps" cannot be measured at all.

This plan converts that heuristic into a **measurable baseline** and builds a genuine learning-to-rank
system beside it, in a new self-contained Python package (`ml/`) that does not disturb the working
product. The learned ranker consumes three modalities — handcrafted acoustic/structural features
(≈70 scalars, an upper design estimate), frozen Whisper-encoder speech representations (encoder width
**384**, pooled over two 4 s windows into a 768-d construction), and MiniLM transcript context
embeddings (native width **384**, expanded to a 1536-d construction) — fuses them with learned
modality gates, and is trained with a pairwise `MarginRankingLoss` plus an auxiliary acceptability
head. See §17 for the four distinct dimension stages. It is served over FastAPI
(`POST /v1/rank-breakpoints`) and consumed by the existing Express route behind a feature flag, so
preview and export are untouched.

**Two hard environment constraints discovered during the audit shape every decision below:**

1. **There is no NVIDIA GPU on this machine.** `nvidia-smi` is absent; the only display adapter is
   `Intel(R) Iris(R) Xe Graphics`. RAM is 32 GB. Every training and extraction step must be
   CPU-feasible; CUDA support is written but exercised only opportunistically.
2. **The default interpreter is Python 3.14.2**, which PyTorch/Whisper/librosa do not reliably
   support. `py -0p` also lists **3.12.13** (uv-managed) and **3.10**. The `ml/` package must pin
   **3.12** in its own virtual environment, separate from whatever `backend/` uses.

Consequently the plan front-loads caching: Whisper and MiniLM run **once per episode**, results are
binned and stored, and training epochs touch only cached tensors. A full training run of the primary
model is expected to be minutes, not hours, on CPU.

Nothing in this document asserts a value for `X` (NDCG@3 lift), `Y` (human agreement), the editing-time
reduction, hours of audio, or candidate count. Every one of those is produced by a named command
into a named artifact, tracked in §33 and in `docs/evaluation-evidence.md`.

---

## 2. Current-state architecture

81 tracked files. Three layers, two process boundaries.

```
frontend/  React 19 + Vite + TS      (npm workspace, no tests)
   │  HTTP, VITE_API_BASE_URL default http://localhost:3001
backend/src/  Express 4 + TS via tsx  (npm workspace, no build, no tests)
   │  spawn(python, ["-m", "ad_inserter.<mod>"], cwd=BACKEND_DIR), JSON over stdout
backend/ad_inserter/  Python package  (no pyproject.toml, no tests)
```

### 2.1 Repository structure (actual)

| Path | Contents | LOC |
|---|---|---|
| `frontend/src/App.tsx` | entire UI flow: upload → analyze → clone → preview → export | 1540 |
| `frontend/src/components/` | `LandingHero`, `SlotifyLogo`, `SoundwaveBackground`, `SoundwaveIcon` | 283 |
| `backend/src/index.ts` | CORS + JSON middleware, mounts 8 routers | 56 |
| `backend/src/config.ts` | `BACKEND_DIR`, `port`, `allowedOrigins`, `pythonBin`, `apiBaseUrl` | 25 |
| `backend/src/types.ts` | `Candidate`, `ScoredCandidate`, `Slot`, `ProsCons`, `SponsorStatement` | 40 |
| `backend/src/lib/candidates.ts` | **the heuristic scorer** | 177 |
| `backend/src/lib/merge-filter.ts` | ffmpeg `filter_complex` builder for insert + preview | 54 |
| `backend/src/lib/text.ts` | `clamp`, `endsWithSentenceBoundary`, JSON field parsing | 36 |
| `backend/src/routes/*.ts` | 8 routers, each declaring its own full path | 1092 |
| `backend/src/services/*.ts` | `elevenlabs`, `ffmpeg`, `openai`, `python` | 480 |
| `backend/ad_inserter/*.py` | `analysis`, `analyze_cli`, `cli`, `insert_ad`, `llm`, `mix`, `tts` | 1405 |
| `backend/scripts/*.ts` | 4 dev scripts (`clone`, `speak`, `test-merge`, `test-preview-export`) | 521 |
| `backend/audio_tests/` | 13 mp3 files, **15.8 s – 28.0 s each, ≈ 4.6 minutes total** | — |
| `docs/` | `.gitkeep` only | — |

### 2.2 Node/Express API contracts (as implemented)

| Endpoint | File | Request | Response |
|---|---|---|---|
| `GET /api/health` | `routes/health.ts` | — | JSON status |
| `POST /api/clone` | `routes/clone.ts:1` | multipart `files[]`, `name` | `{ voiceId }` |
| `POST /api/insert-sections` | `routes/insert-sections.ts:35` | multipart `audio`, `count`, `mode`, `sponsors`/`statements` | `{ points[], confidences[], duration, source, slots[], sponsorStatements[] }` |
| `POST /api/related-products` | `routes/related-products.ts` | JSON | product suggestions |
| `POST /api/tts` | `routes/tts.ts:10` | JSON `voiceId`, `text`/`statements`/`sponsor`, `modelId`, `outputFormat`, `pauseMs` | `audio/mpeg` stream |
| `POST /api/speech` | `routes/tts.ts:85` | — | 307 redirect to `/api/tts` |
| `POST /api/generate` | `routes/generate.ts:11` | multipart `audio`, `voiceId`/`voiceIds`, `brand`, `productDesc` | `audio/mpeg` (runs `ad_inserter.cli`) |
| `POST /api/merge` | `routes/merge.ts:11` | multipart `audio`, `insert`; fields `insertAt`, `crossfade`, `pause`, `preview`, `previewSeconds` | `audio/mpeg` |
| `POST /ad/insert` | `routes/ad-insert.ts:11` | multipart `audio` + product/style/mode/voice fields | `audio/mpeg` (runs `ad_inserter.insert_ad`) |

`/api/insert-sections` is the **only** endpoint the ranking model needs to change.

### 2.3 Frontend flow

`App.tsx` holds ~30 `useState` hooks and three pages (`landing`/`upload`/`analyze`/`export`,
`timelineSteps` at `App.tsx:9`).

- `handleAnalyzeInsert` (`App.tsx:567`) POSTs `audio` + `count=3` to `/api/insert-sections`, reads only
  `data.points` and `data.confidences`, and stores `InsertSuggestion[]`. **It ignores `slots`,
  `pros`, `cons`, and `rationale` entirely** — the LLM enrichment in `routes/insert-sections.ts:226-258`
  is computed and discarded.
- `useEffect` at `App.tsx:453-482` turns suggestions into `Slot[]`. **If fewer than three suggestions
  arrive it substitutes `[0.22, 0.48, 0.72] × duration` with hard-coded confidences `[92, 85, 78]`.**
- `handleRequestAnalyze` → rights modal → `handleConfirmRights` (`App.tsx:643`) clones the uploaded
  voice via `/api/clone`.
- Preview and export share one function (`App.tsx:~690-812`): for each selected slot, POST `/api/tts`,
  then POST `/api/merge` with `preview=1&previewSeconds=3` (preview) or without (render). Slots are
  processed **latest-first** (`App.tsx:738`) so earlier insert offsets stay valid. Output is an object
  URL on an `<audio>` element.

---

## 3. Current heuristic and product workflow

The heuristic is **split across two languages and duplicated in two divergent code paths**. This is
the single most important audit finding.

### 3.1 Path A — the product path (`/api/insert-sections`)

1. **Candidate generation (Python)** — `analyze_cli.py:14` `_detect_podcast_silences`:
   `pydub.silence.detect_silence(min_silence_len=700, silence_thresh=audio.dBFS - 16)` (falls back to
   `-40` dBFS when `dBFS` is `-inf`). Each silence yields `{start_ms, end_ms, mid_ms, silence_ms}`;
   candidate time is the **midpoint** of the silence. Capped at `--max-candidates` (default 30; the Node
   caller does not override it). Song mode instead uses `analysis.analyze_song` (§3.3).
2. **Snippets (Python, optional)** — `analyze_cli.py:34` transcribes the **15 s preceding** the first
   `--snippet-count` candidates (Node passes `12`, `services/python.ts:20`) via
   `analysis.transcribe_snippet`, only if `import whisper` succeeds.
3. **Extra candidates (Node)** — if `OPENAI_API_KEY` is set, `services/openai.ts:196` `requestTranscription`
   calls the `whisper-1` API with `response_format=verbose_json`; segments whose text passes
   `endsWithSentenceBoundary` become candidates at `segment.end`, with `silenceMs` = gap to the next
   segment start (`routes/insert-sections.ts:90-105`).
4. **Merge (Node)** — `candidates.ts:9` `mergeCandidates(base, extra, minGapMs = 400)`: sort by time,
   collapse anything within 400 ms, keeping whichever has the larger `silenceMs` **or** ends on a
   sentence boundary when the incumbent does not.
5. **Score (Node)** — `candidates.ts:34` `scoreCandidate`, the formula to preserve verbatim:

   ```
   score = 0.40                                                     # base
         + min(0.40, (silenceMs / 2000) * 0.40)   if silenceMs      # pause reward, saturates at 2 s
         + 0.10                                   if mode == "song"
         + 0.30                                   if snippet ends with [.!?]  (and != "TRANSCRIPT_UNAVAILABLE")
         - 0.20                                   if snippet present but does not
         + 0.10                                   if 0.20 <= t/duration <= 0.80
         - 0.30                                   if t < 5 s or t > duration - 5 s
   score = clamp(score, 0, 1)
   ```

6. **Selection (Node)** — `candidates.ts:117` `selectTopSlots(candidates, duration, minSeparationSeconds=6, count)`:
   sort by score desc then time asc, greedily accept while ≥ 6 s from every accepted slot; if short,
   pad with `[0.22, 0.50, 0.78] × duration` at score `0.5`; if still short, pad at `minSeparation`
   multiples at score `0.4`. `insert-sections.ts:174-179` calls it with `count = max(3, count)` then
   `.slice(0, 3)`.
7. **Presentation (Node)** — `confidence_percent = round(clamp(70 + score * 25, 70, 95))`
   (`insert-sections.ts:207`). Note this **compresses [0,1] into [70,95]** — the displayed confidence
   is a cosmetic affine transform of the score, not a probability.
8. **Enrichment (Node)** — `services/openai.ts:84` `enhanceSlotsWithOpenAI` fills pros/cons/rationale
   under a strict JSON schema; fallback text comes from `candidates.ts:66` `buildFallbackProsCons`.
   Slots are then re-sorted by `confidence_percent` (`insert-sections.ts:260`).

### 3.2 Path B — the CLI path (`ad_inserter.cli`, used by `/api/generate`)

Divergent and **not** the product path:

- candidates from `analysis.py:46` `detect_podcast_candidates` — same idea but `min_silence_len=**500**`
  (not 700), midpoint of each silence;
- hard-coded `min_insert_ms = 30000`, `end_buffer_ms = 15000` (`cli.py:164-165`) via
  `analysis.filter_candidates`;
- default choice = first candidate ≥ 30 s (`analysis.py:146` `choose_default_insertion`);
- final choice delegated to **the LLM** (`llm.py:31` `_build_prompt` → `chosen_index`), re-validated
  against the offset window at `cli.py:244-255`.

There is no scoring function in Path B at all; the "semantic" decision is an LLM index pick from a
prompt containing the 15 s of transcript preceding each candidate.

### 3.3 Signal logic inventory

| Signal | Location | Parameters |
|---|---|---|
| Silence (product) | `analyze_cli.py:14` | `min_silence_len=700`, `thresh = dBFS - 16` |
| Silence (CLI) | `analysis.py:46` | `min_silence_len=500`, `thresh = dBFS - 16` |
| RMS minima | `analysis.py:55` `_rms_minima_times` | 50 ms frame, 25 ms hop, `top_k=12`, `argsort` ascending |
| Beat tracking | `analysis.py:77` `analyze_song` | `librosa.beat.beat_track`, `sr=44100` mono; each RMS minimum snapped to nearest beat |
| Sentence boundary | `lib/text.ts:35` | regex `/[.!?]["')\]]?\s*$/` |
| Loudness (LUFS) | `mix.py:27` | `pyloudnorm.Meter(frame_rate)`, integrated |
| Crossfade / ducking | `mix.py:68` | `crossfade_ms=250`, `duck_db=0.0` (ducking effectively off) |
| Context window | `mix.py:92` | ±4000 ms — reused as the loudness-match target |
| Diarization | `analysis.py:195` | optional `pyannote/speaker-diarization`, top-2 speakers → A/B |

Spectral features (centroid, bandwidth, contrast, roll-off, ZCR, onset) are **not computed anywhere
today** despite `librosa` being a dependency.

### 3.4 Transcription — two independent implementations

- **Remote:** `services/openai.ts:196`, `whisper-1`, `verbose_json`, segment-level, costs money,
  requires network, used only for extra candidates.
- **Local:** `analysis.py:118` `transcribe_snippet` — **calls `whisper.load_model(model_name)` inside
  the per-snippet loop**, so a 12-snippet analysis loads the model 12 times, and writes a temp WAV
  per snippet. This is the largest performance defect in the current code and is unusable at 50-hour
  scale.

### 3.5 Decoding, preview, export, storage

- Decode/standardize: `analysis.py:36-43` — `AudioSegment.from_file` then `set_frame_rate(44100).set_channels(2)`.
  librosa paths load at `sr=44100` **mono** (`analysis.py:80`). Two different canonical formats coexist.
- Preview: `/api/merge` with `preview=1`, `previewSeconds=3` → `lib/merge-filter.ts:22` emits
  `atrim` before/after windows + the ad, `concat=n=3:v=0:a=1`.
- Export: same filter without the preview window (`merge-filter.ts:41`), encoded `libmp3lame -q:a 2`.
- Storage: **none persistent.** `multer.memoryStorage()` (`middleware/upload.ts:4`) holds whole uploads
  in RAM; routes write to `os.tmpdir()` mkdtemp dirs and delete them in `finally`/stream-close handlers.
  Nothing is ever retained between requests.

---

## 4. Existing components to retain (unchanged)

| Component | Why |
|---|---|
| `lib/merge-filter.ts`, `routes/merge.ts` | preview + export work; none of the ranking results require touching them |
| `routes/tts.ts`, `routes/clone.ts`, `services/elevenlabs.ts` | ElevenLabs integration is the hackathon award; out of ML scope |
| `mix.py` (all of it) | loudness matching, room tone, crossfade — reused verbatim for labelling-clip generation |
| `analysis.py:186-245` diarization | reused as the optional speaker-change feature source |
| `ad_inserter/cli.py`, `insert_ad.py`, `llm.py`, `tts.py` | the ad-writing/voicing product; the ranker replaces only *placement*, not *ad generation* |
| `analysis.py:77` `analyze_song`, beat logic | the music path stays heuristic-only by design (§ locked scope) |
| Entire React UI shell, waveform canvas, preview/export handlers | integration is additive |

## 5. Components requiring refactoring

| Component | Problem | Action | Phase |
|---|---|---|---|
| `lib/candidates.ts` | scoring constants are literals inline; unreproducible from Python | ✅ **done (Phase 1)** — constants hoisted into `config/heuristic_offline_v1.json`, loaded by `lib/heuristic-config.ts`; output proven byte-identical before/after; ported to `ml/src/slotify_rank/candidates/heuristic.py` with a golden-parity test | 1 |
| `analyze_cli.py` vs `analysis.detect_podcast_candidates` | 700 ms vs 500 ms divergence | ✅ **done (Phase 1)** — `ad_inserter/heuristic_config.py` exposes `PRODUCT_V1` (700 ms) and `LEGACY_CLI_V1` (500 ms); each call site names its profile; the divergence is documented rather than reconciled | 1 |
| `routes/insert-sections.ts:181-260` slot finalisation | the deterministic finalisation (clamp → dedupe → confidence → sort) is inline in the route, interleaved with a non-deterministic OpenAI call, so it cannot be imported by the golden generator | **mirrored** in `scripts/dump-heuristic-golden.ts` and in the Python port for Phase 1 (route untouched). Extract a `finalizeSlots()` helper in Phase 7, when the route is being edited for ranker integration anyway. **Drift risk R17.** | 7 |
| `analysis.transcribe_snippet` | reloads the Whisper model per snippet | new `ml/.../transcribe/whisper_asr.py` with a module-level cached model and whole-episode transcription; the old function stays for backward compat but is no longer on the ML path | 3 |
| `App.tsx:453-482` | **fabricates 3 slots at ≈22/48/72 % with hard-coded confidences `[92,85,78]`** whenever fewer than three suggestions arrive | **Product technical debt.** Never enters evaluation: `ml/` computes rankings independently and never reads the frontend. Phase 7 replaces it with either (a) clearly-labelled *unscored* fallback suggestions or (b) an explicit insufficient-candidates state. Deliberately **not** touched in Phase 1 — it did not block the golden fixture, which is generated from the scorer module directly. | 7 |
| `App.tsx:590-616` | discards `slots[]`, pros/cons/rationale | read the richer payload so model explanations surface | 7 |
| `middleware/upload.ts` memory storage | a 2-hour episode ≈ 100 MB in RAM per request; dataset building must never go through the API | dataset pipeline reads from disk directly; API left as-is with a documented size caveat | 2 |
| `insert-sections.ts:207` confidence | affine rescale masquerading as confidence | when the model serves, emit the real `acceptability_probability`; keep the legacy field for compatibility | 7 |

## 6. Target architecture

```
frontend/ (React)                       unchanged flow, richer payload
    │  POST /api/insert-sections   (contract preserved, fields added)
backend/src/routes/insert-sections.ts
    │  RANKER_ENABLED=1 ?
    ├─ yes → services/ranker.ts ──HTTP──► ml/  FastAPI  POST /v1/rank-breakpoints
    │                                        ├─ candidate generation (multi-source, merged)
    │                                        ├─ feature + embedding pipeline (cached)
    │                                        ├─ trained gated multimodal ranker (PyTorch)
    │                                        └─ top-3 + acceptability + explanations
    └─ no  → existing Python analyze_cli + candidates.ts heuristic   (untouched fallback)
```

`ml/` is a **separate Python project with its own `pyproject.toml` and 3.12 venv**. It does not import
`ad_inserter`, and `ad_inserter` does not import it. The only coupling is a JSON heuristic-config file
so the ported baseline provably matches production.

## 7. End-to-end data flow (text)

**Offline (dataset → model):**

```
sources.yaml (URL, licence, series)
  → fetch.py            → data/audio/{episode_id}.{ext}  + checksum + licence record
  → decode              → 16 kHz mono float32 wav cache (feature canonical) + 44.1 kHz for clips
  → whisper_asr.py      → transcripts/{episode_id}.json   (words + segments + timestamps)
  → generators.py       → raw candidates (7 sources, high recall)
  → merge.py            → merged candidates (tolerance 0.75 s, source flags OR-ed)
  → heuristic.py        → heuristic_score + 5 component scores per candidate
  → acoustic.py         → 34 acoustic scalars per candidate
  → transcript.py       → 36 structural/contextual scalars per candidate
  → audio_embed.py      → per-episode binned Whisper-encoder tensor → 768-d per candidate
  → text_embed.py       → MiniLM before/after → 2 × 384-d per candidate
  → candidates.parquet + features/*.npy + manifest.jsonl
  → splits.py           → episode-level (and series-aware) train/val/test
  → labeling app        → labels.jsonl (human scores, resumable)
  → training/loop.py    → checkpoints + metrics.json
  → eval/evaluate.py    → artifacts/evaluation/*.json + ablation_results.csv
  → eval/report.py      → artifacts/reports/final_results.md  (computes X automatically)
```

**Online (request → top-3):**

```
upload → Express writes temp wav → ranker.ts POST /v1/rank-breakpoints (path or bytes)
  → decode 16 kHz mono → whisper transcribe (tiny.en) → generate candidates → merge
  → features + embeddings (no disk cache required; in-process) → model.forward (batch)
  → spacing constraint (≥ min_separation_s) + edge constraints → top-3
  → JSON: rank, timestamp, model_score, acceptability_probability, sources, explanation_features
```

---

## 8. Dataset acquisition and licensing strategy

**The existing `backend/audio_tests/` corpus is 13 files totalling ≈ 4.6 minutes** (longest 28.0 s),
three of which are music. It is a **smoke-test fixture only** and must never be described as a dataset.

Selection criteria: English, spoken word, redistributable or at minimum re-fetchable by URL, and
ideally shipped with transcripts or speaker turns.

**Domain priority (locked, §0):** podcasts, interviews, narrated spoken word and conversational
spoken-word recordings are the target domain. AMI meetings are **supplemental only** — meetings are
not podcasts, and a test set of meetings would produce metrics that do not describe the product.

| Tier | Source | Licence | Why | Target |
|---|---|---|---|---|
| **Primary** | **Internet Archive** spoken-word/podcast collections filtered to CC-BY / CC0 / public domain | per-item, recorded | genuine podcast-style long-form conversation — the actual target domain | 20–30 h |
| **Primary** | **US federal agency podcasts** (NASA, NIH, CDC) | US Gov works → PD | real podcast production conventions, interview format, unambiguous licensing | 10–15 h |
| **Primary** | **CC-licensed podcast feeds** discovered via the Podcast Index licence field | CC-BY / CC0 | independent podcasts with real ad-break conventions | 5–15 h |
| Secondary | **LibriVox / LibriSpeech** | PD | covers the "narrated spoken word" locked-scope category; trivially legal; caps how much narration can dominate | 5–10 h |
| **Supplemental** | **AMI Meeting Corpus** | CC-BY-4.0 | ground-truth speaker turns for the speaker-change features and conversational-transition study. **Capped, and excluded from the held-out test set.** | ≤ 10 h |
| Rejected | Spotify Podcast Dataset | distribution discontinued/restricted | cannot be re-fetched reproducibly | — |
| Rejected | GigaSpeech | signed agreement required | blocks a public portfolio repo | — |

**Test-set constraint (enforced in `splits.py`, asserted in `validate.py`):** every episode in the
held-out test split must have `content_type ∈ {podcast, interview, conversational}` and
`source != "ami"`. `split_statistics.json` reports the content-type composition of each split so the
constraint is visible, not merely asserted.

Rules:
- **No audio is ever committed.** `data/` is gitignored. `configs/sources.yaml` holds
  `{episode_id, url, sha256, licence, attribution, series_id, duration_s}`; `fetch.py` reconstructs
  the corpus and verifies checksums.
- Every episode record carries `licence` and `attribution`; `dataset_statistics.json` reports the
  breakdown by licence.
- MVP dev corpus (fast iteration): **8–12 h**. Portfolio corpus: **≥ 50 h**, reached by extending
  `sources.yaml` only — no code change. The dataset statistics script prints actual hours; the README
  and the evidence matrix quote that number, never a target.

### 8.1 The two dataset layers, kept strictly separate

| Layer | Target | Label status |
|---|---|---|
| **Processed corpus** | ≥ 50 h audio, ≥ 10 000 candidates | mostly **unlabelled**; some **weakly labelled** by heuristic/metadata |
| **Human-labelled subset** | ≈ 8–12 h audio, ≈ 1 500 candidates | **human**, against the §12 rubric |
| **Held-out evaluation subset** | the test-split portion of the above | **human**, and the only source of final test metrics |

Weak labels may be used for pretraining, sampling and bootstrapping. They may
**never** appear as ground truth in a test metric — `validate.py` fails the run
if a test-split relevance label has `label_source != "human"`.

## 9. Processed-dataset schema (`data/processed/`)

```
data/
  raw/audio/{episode_id}.{ext}              # fetched, gitignored
  processed/
    episodes.jsonl                          # one row per episode
    candidates.parquet                      # one row per candidate (§11)
    transcripts/{episode_id}.json           # whisper words+segments
    features/handcrafted/{episode_id}.npy   # (n_cand, 70) float32
    features/text/{episode_id}.npy          # (n_cand, 768) float16  [before‖after]
    features/audio/{episode_id}.npy         # (n_bins, 384) float16  binned encoder output
    clips/{candidate_id}.mp3                # labelling previews, generated on demand
  labels/labels.jsonl                       # append-only human labels (§10)
  splits/{train,val,test}.txt               # episode ids
```

`episodes.jsonl` row:

```
episode_id, series_id, source, url, sha256, licence, attribution,
duration_s, sample_rate_in, channels_in, language, content_type{podcast|interview|narrated|music},
transcript_model, transcript_version, n_candidates, n_labelled,
preprocessing_version, fetched_at, processed_at
```

## 10. Human-labelled dataset schema (`data/labels/labels.jsonl`, append-only)

```
label_id, candidate_id, episode_id,
annotator_hash,          # sha256(annotator_name + salt)[:12] — no PII
task_type,               # "pointwise" | "pairwise"
quality_score,           # 1..5, pointwise only
is_acceptable,           # bool, pointwise only
pairwise_opponent_id,    # candidate_id, pairwise only
pairwise_choice,         # "a" | "b" | "tie", pairwise only
listened_ms, n_replays, decision_ms,
label_source,            # "human" | "weak_heuristic" | "metadata_derived" | "unlabelled"
rubric_version, ui_version, created_at
```

Append-only JSONL with `fsync` per write gives crash safety and free resumability (§12).
Weak labels are written to a **separate** file `data/labels/weak_labels.jsonl` with the identical
schema and `label_source != "human"`, so the two can never be conflated by a join mistake.

## 11. Candidate schema (`candidates.parquet`)

| Field | Type | Notes |
|---|---|---|
| `episode_id` | str | |
| `candidate_id` | str | `{episode_id}:{timestamp_ms:08d}` — deterministic |
| `timestamp_seconds` | float64 | merged representative time |
| `candidate_sources` | list[str] | any of `silence`, `whisper_segment_end`, `whisper_word_gap`, `rms_minimum`, `spectral_change`, `speaker_turn`, `fixed_interval` |
| `n_sources` | int8 | |
| `heuristic_score` | float32 | port of `scoreCandidate` |
| `heuristic_components` | struct(5) | `silence_term`, `sentence_term`, `position_term`, `edge_penalty`, `mode_term` |
| `silence_ms`, `pause_ms` | float32 | |
| `dist_to_sentence_boundary_s` | float32 | |
| `audio_window_reference` | struct | `{path, bin_start, bin_end, bin_seconds}` into `features/audio/*.npy` |
| `transcript_before`, `transcript_after` | str | ≤ 400 chars each |
| `sentence_boundary_features` | struct | `is_sentence_end`, `punct_type`, `is_incomplete`, `n_words_before/after` |
| `handcrafted_audio_features` | struct | index into `features/handcrafted/*.npy` row |
| `audio_embedding_reference` | struct | row/slice reference |
| `text_embedding_reference` | struct | row reference into `features/text/*.npy` |
| `quality_score` | float32 \| null | human, 1–5 |
| `is_acceptable` | bool \| null | human |
| `label_source` | str | `human` / `weak_heuristic` / `metadata_derived` / `unlabelled` |
| `annotator_hash` | str \| null | |
| `dataset_split` | str | `train` / `val` / `test` / `unassigned` |
| `preprocessing_version` | str | e.g. `prep-v1.0.0` — bumping invalidates caches |
| `candidate_config_version` | str | e.g. `cand-v1.0.0` |

Parquet, one file, partitioned by `episode_id` if it exceeds ~500 MB.

## 12. Labelling rubric and interface

**Rubric v1** (shown in the UI on every screen):

| Score | Meaning | Test to apply |
|---|---|---|
| 1 | Disruptive — inside a word, phrase, or mid-clause | Would a listener think the file broke? |
| 2 | Possible but clearly unnatural — clause boundary, thought unfinished | Sentence trails into the next |
| 3 | Acceptable — sentence ends, no real pause or topic shift | Neutral; you'd tolerate an ad |
| 4 | Strong natural break — sentence ends **and** a real pause | You'd expect a beat here |
| 5 | Highly natural transition — topic/segment change, speaker hand-off, "we'll be right back" energy | An editor would choose this |

`is_acceptable` is collected **separately** from the 1–5 score (not derived by threshold), so the
auxiliary head trains on an independent human signal. The relationship between them is reported in
the dataset statistics as a sanity check.

**Interface** — `ml/src/slotify_rank/labeling/app.py`, a FastAPI app serving one static HTML page
(no build step, no npm; deliberately not part of the React app so the product stays untouched):

- plays **8–12 s before** and **8–12 s after** the candidate (`clip_seconds` configurable, default 10),
  as two separate `<audio>` elements plus a joined "as it would sound" clip — replay each side
  independently;
- shows `transcript_before` / `transcript_after` with the boundary marked, transcript hidden behind a
  toggle for a first listen-only pass;
- keyboard: `1–5` score, `A`/`U` acceptable/unacceptable, `R` replay, `Space` play/pause, `Enter` submit;
- optional pairwise mode: two candidates from the **same episode**, randomized order;
- clips are generated on demand with `pydub` via existing `mix.context_window` semantics and cached in
  `data/clips/`;
- **hides `heuristic_score`, model scores, and candidate source flags** — the annotator must not be
  primed by the systems under test;
- resume: the queue is `candidates_to_label - labels.jsonl(candidate_id, annotator_hash)`, recomputed
  at startup, so killing the process loses at most the in-flight item;
- sampling for the queue: stratified over heuristic-score deciles and episode position, so labels do
  not concentrate on what the heuristic already likes (this bias would inflate the baseline).

**Volume, stated honestly.** At a realistic 20–30 s per pointwise label, 2 000 labels is **11–17 hours**
of human work. MVP target: **1 200–1 500 pointwise labels over 15–25 episodes (10–15 h audio)**, with a
pairwise fast-mode (~8 s per judgement) used to densify ranking signal cheaply. Scaling to 3 000 needs
no code change.

## 13. Human-evaluation methodology

Three distinct quantities. **`Y` is defined as A1 wherever it is reported** and the definition is
printed verbatim next to the number in `final_results.md`, the model card, and the README.

- **A1 — top-3 acceptability hit rate (primary, "human agreement").**
  Over held-out **test** episodes: the fraction of episodes in which **at least one** of the model's
  top-3 predictions is a candidate a human labelled `quality_score ≥ 4`, matched with a **±3 s
  timestamp tolerance** (configurable `agreement_tolerance_s`). Computed for the heuristic too.
- **A2 — top-1 match rate.** Fraction of test episodes where the model's rank-1 prediction lies within
  the tolerance of *the* human-preferred candidate (highest human score in that episode; ties count
  as a match against any tied candidate).
- **A3 — blind pairwise preference vs the heuristic.** The defensible head-to-head:
  1. On held-out test episodes, take the heuristic's rank-1 and the model's rank-1; skip episodes where
     they agree within tolerance (report the skip rate — it is itself informative).
  2. Render both 20 s context clips identically, present in **randomized order**, system identity hidden,
     file names anonymised.
  3. Evaluator answers: *A more natural / B more natural / equivalent*.
  4. **At least one evaluator other than the project author is required**; two additional evaluators
     preferred. With ≥ 2 evaluators on an overlapping subset, report inter-rater agreement
     (Cohen's κ); with one, report a bootstrap CI on the win rate and state `n_evaluators = 1`
     prominently as a limitation.
  5. Report `model_win_rate`, `heuristic_win_rate`, `tie_rate`, `n_comparisons`, `n_evaluators`, `kappa`.

Guard rails, enforced in code: `human_eval.py` refuses to run on episodes not in the test split, and
`evaluate.py` raises if any test-set relevance label has `label_source != "human"`. **No human-preference
number may be computed from heuristic- or model-generated labels.**

## 14. Editing-time benchmark methodology

Within-subject, counterbalanced, run through the real product.

- **Participants: ≥ 3 required, 5 targeted where practical.** **Episodes:** ≥ 6 unseen (test-split)
  episodes, 10–40 min each. If fewer than 3 participants are available the result is reported as a
  pilot with `n` stated, not as a headline claim.
- **Design:** Latin square over (participant × episode × workflow) so no participant does the same
  episode twice and each episode is done under both workflows an equal number of times. Order of
  workflow is alternated.
- **Manual arm:** ranker disabled (`RANKER_ENABLED=0`) and the suggestion panel hidden via
  `?workflow=manual`; the user scrubs the waveform, picks a point, previews, exports.
- **Assisted arm:** ranker enabled; the user reviews the top-3, selects, previews, exports.
- **Instrumentation:** a small event log written by the frontend to `POST /api/benchmark/event`
  (new, dev-only, disabled unless `BENCHMARK_MODE=1`) capturing `session_id, participant_hash,
  episode_id, workflow, event, t_ms`. Timer starts at first interaction after load, stops at export
  completion. Also recorded: `n_preview_attempts`, `n_rejected_candidates`, `final_placement_accepted`
  (self-reported yes/no), `episode_duration_s`, `workflow_version`, `model_version`.
- **Statistic:** per-pair `100 * (manual_time - assisted_time) / manual_time`; report **median** and
  mean with a bootstrap 95 % CI (n resamples = 10 000), plus the per-episode table.
- Raw rows → `artifacts/benchmarks/editing_time_raw.csv`; aggregate → `editing_time_results.csv`.

**The measured number is reported as measured.** If it is 45 %, every report says 45 %. The benchmark
is specified before any data is collected precisely so it cannot be tuned toward 80 %.

---

## 15. Candidate-generation algorithm

Optimised for **recall**; the model supplies precision.

```
for episode e:
    y16 = decode(e, sr=16000, mono)                    # canonical feature audio
    tr  = whisper_transcribe(e)                        # words[], segments[]
    C = []
    # 1 sentence/segment ends
    C += [seg.end for seg in tr.segments if endswith_terminal_punct(seg.text)]
    # 2 word-gap pauses
    C += [w_i.end + gap/2 for consecutive words where gap = w_{i+1}.start - w_i.end >= 0.35 s]
    # 3 silence regions  (pydub parity with production heuristic)
    C += [mid(s) for s in detect_silence(min_silence_len=400, thresh=dBFS-16)]
    # 4 local RMS minima  (generalisation of analysis._rms_minima_times to podcasts)
    C += [argrelmin(rms, order=k) filtered to rms < percentile(rms, 15)]
    # 5 energy change
    C += [t where |d/dt smoothed_rms| is a local max above percentile 90]
    # 6 spectral change
    C += [t where cosine distance between adjacent 1 s MFCC means > percentile 90]
    # 7 speaker turns (when diarization/ground truth available)
    C += [turn boundaries]
    # 8 fixed-interval fallback, applied only where a 90 s stretch has no candidate
    C += [uncovered_gap_midpoints at 45 s spacing]

    C = merge(C, tolerance = 0.75 s)      # union of source flags, representative = median time
    C = drop(t < edge_guard_s or t > duration - edge_guard_s)     # edge_guard_s = 20
    C = cap_per_episode(C, max_per_minute = 8, keep = highest n_sources then longest pause)
```

**Merge rule:** cluster by single-linkage within `tolerance`; representative timestamp = the median of
the cluster; `candidate_sources` = union of all member flags; `silence_ms`/`pause_ms` = max over members.
All source flags survive, as required.

**Yield model (for the 10 000-candidate claim):** after merging and capping, expect **3–6 candidates per
minute** of conversational audio. At 4/min, 50 h → **12 000 candidates**. At the pessimistic 3/min,
50 h → 9 000, and the cap/`max_per_minute` knob or 60 h of audio closes the gap. Because the number is
computed and written to `dataset_statistics.json`, the plan never has to guess.

Every parameter above lives in `configs/candidates_v1.yaml` and is stamped into
`candidate_config_version`.

## 16. Feature definitions

**Windows.** Unless noted, "before" = `[t - 3.0, t)`, "after" = `[t, t + 3.0)`, plus a micro window
`[t - 0.5, t + 0.5]`. All computed on 16 kHz mono float32.

### 16.1 Handcrafted acoustic — 34 dims

| # | Feature | Rationale |
|---|---|---|
| 1–4 | `rms_before` mean/min/max/var | is the speaker winding down? variance separates steady speech from a trailing off |
| 5–8 | `rms_after` mean/min/max/var | does speech resume immediately (bad) or gently (good) |
| 9 | `rms_local_min` (±0.5 s) | depth of the actual gap at the cut |
| 10 | `rms_delta` = after_mean − before_mean | energy step across the boundary |
| 11 | `rms_log_ratio` | scale-invariant version of 10 |
| 12 | `silence_duration_s` | the primary signal the current heuristic uses; must be available to the model |
| 13 | `pause_duration_s` (word-gap) | transcript-derived pause, more precise than dBFS silence |
| 14 | `time_since_prior_speech_s` | interruption risk on the left |
| 15 | `time_until_next_speech_s` | interruption risk on the right |
| 16–17 | `zcr_before/after` mean | fricative/noise vs voiced; catches cuts inside sibilants |
| 18–19 | `spectral_centroid_before/after` | timbre continuity across the cut |
| 20–21 | `spectral_bandwidth_before/after` | as above |
| 22–23 | `spectral_rolloff_before/after` | as above |
| 24–25 | `spectral_contrast_mean_before/after` | speech vs music/bed distinction |
| 26 | `spectral_flux_at_boundary` | abrupt timbre change = a real content boundary |
| 27–28 | `onset_strength_at_t`, `onset_strength_max_window` | onset right at `t` means we are cutting into an attack |
| 29–31 | `lufs_before`, `lufs_after`, `lufs_delta` | perceptual loudness — directly relevant to how jarring the ad entry is; reuses `mix.measure_lufs` |
| 32 | `dbfs_at_t − episode_dbfs` | candidate quietness relative to the episode's own level |
| 33 | `nearest_speech_distance_s` | distance to the closest voiced frame |
| 34 | `speech_rate_wps_before` | words/second before the cut; fast delivery = mid-thought |

Music-specific tempo/beat features are **excluded** from the podcast model (locked scope); the field
exists in the spec registry but is masked off by `configs/features_v1.yaml`.

### 16.2 Transcript, structural and contextual — 36 dims

| # | Feature |
|---|---|
| 35 | `is_sentence_end` (regex parity with `lib/text.ts:35`) |
| 36–40 | `punct_type` one-hot: period / question / exclamation / comma / none |
| 41 | `is_incomplete_sentence` — trailing conjunction, determiner, preposition, or filler |
| 42–43 | `n_words_before`, `n_words_after` (in window) |
| 44–45 | `utterance_duration_before`, `utterance_duration_after` |
| 46 | `cosine_sim_across_boundary` (MiniLM before vs after) |
| 47 | `semantic_change_magnitude` = 1 − (46) |
| 48–49 | `speaker_change_flag`, `speaker_change_confidence` (0 when unavailable, with an explicit `speaker_available` mask at 50) |
| 50 | `speaker_available` |
| 51 | `position_ratio` = t / duration |
| 52–53 | `distance_from_start_s`, `distance_from_end_s` (log1p) |
| 54 | `episode_duration_s` (log1p) |
| 55 | `timestamp_s` (log1p) |
| 56 | `dist_to_nearest_other_candidate_s` |
| 57 | `candidate_density_60s` |
| 58–64 | seven `candidate_source` binary flags |
| 65 | `n_sources` |
| 66 | `heuristic_score` (production parity port) |
| 67–70 | heuristic components: `silence_term`, `sentence_term`, `position_term`, `edge_penalty` |

**`D_hand ≈ 70` is an upper design estimate, not a requirement.** The enumerations above are a
candidate list; the implemented set may be smaller. Features are not retained to inflate a count, and
later ablations (§23) decide whether whole groups earn their place.

Every **implemented** feature must carry, in `features/spec.py`:

| Property | Requirement |
|---|---|
| stable name | snake_case, never reused for a different meaning |
| definition | one-line formula or precise prose |
| extraction window | explicit (e.g. `[t−3.0, t)`, `[t−0.5, t+0.5]`, whole episode) |
| missing-value policy | the sentinel and its companion `*_available` mask, or the documented default |
| normalisation policy | `standardise` / `log1p_then_standardise` / `none` (for binary flags) |
| feature-version metadata | inclusion in `FEATURE_SPEC_VERSION` and the spec hash |

The realised count is written into the spec hash and asserted by `tests/test_feature_spec.py`, so the
model's input width follows the spec rather than a number typed in two places.

Normalisation: per-feature standardisation with **statistics fitted on the training split only**,
persisted to `artifacts/models/{run_id}/feature_stats.json`, applied at serving time.

## 17. Expected tensor dimensions

Every dimension below is stated at one of four explicitly distinguished stages.
Conflating them is what produced the errors corrected in this revision.

### 17.1 Audio: four distinct stages

| Stage | Dimension | Notes |
|---|---|---|
| **Raw encoder hidden dimension** | **384** | `openai/whisper-tiny.en` encoder `d_model = 384`. This is the model's native width and is **not** 768. `whisper-base.en` is 512. |
| **Temporal downsampling** | 1500 frames → 150 bins per 30 s | encoder emits 50 Hz frames (20 ms); mean-pooled into 0.2 s bins. Dimension is unchanged at 384; only the time axis shrinks. |
| **Candidate-window pooling** | 384 → **768** | mean-pool the bins in `[t−4, t)` and `[t, t+4)` separately, then concatenate: `2 × 384 = 768`. The 768 is a *construction*, not the encoder width. |
| **Projected model dimension** | 768 → **128** | the audio branch projection (unchanged from Phase 0; approved). |

### 17.2 Text: two distinct stages

| Stage | Dimension | Notes |
|---|---|---|
| **Native sentence-embedding dimension** | **384** | `sentence-transformers/all-MiniLM-L6-v2` outputs 384. It is **not** 1536. |
| **Constructed transcript feature** | **1536** | built as below. |

```text
concat(
    before_embedding,             # 384
    after_embedding,              # 384
    abs(before - after),          # 384
    before * after                # 384
)                                 # total: 1536
```

Projected model dimension: 1536 → **128** (approved).

### 17.3 Full tensor table

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| handcrafted `x_hand` | `(B, D_hand)` | float32 | standardised; `D_hand ≈ 70` is an upper design estimate (§16), read from the feature spec, never hard-coded |
| text before/after | `(B, 384)` each | float32 | MiniLM native output |
| text feature `x_text` | `(B, 1536)` | float32 | constructed as in §17.2 |
| per-episode audio cache | `(n_bins, 384)` | float16 | encoder width 384, 0.2 s bins |
| audio pooled `x_audio` | `(B, 768)` | float32 | `[pool(t−4,t) ‖ pool(t,t+4)]`, each 384 |
| branch projections | `(B, 128)` each | float32 | three branches |
| gates | `(B, 3)` | float32 | sigmoid, one per modality |
| fused | `(B, 128)` | float32 | |
| `score` | `(B,)` | float32 | unbounded scalar |
| `accept_logit` | `(B,)` | float32 | sigmoid → probability |

With `whisper-base.en` the cache becomes `(n_bins, 512)` and `x_audio` becomes `(B, 1024)`. Every
branch input dimension is read from the cache header at construction time, never hard-coded.

**Parameter count** ≈ 0.35 M — deliberately small enough to train on this CPU in minutes.

## 18. Embedding and cache strategy

The naive approach (run the Whisper encoder per candidate) costs 10 000 × one 30 s forward pass and is
the difference between a 40-minute job and an overnight one on this hardware. Instead:

1. Split the episode into **non-overlapping 30 s chunks** (the encoder's fixed input length).
2. Run the frozen `whisper-tiny.en` encoder once per chunk → `(1500, 384)` frames at 50 Hz.
3. **Mean-pool into 0.2 s bins** → `(150, 384)` per chunk, cast to float16, concatenate per episode →
   `features/audio/{episode_id}.npy`, ~115 KB per 30 s ⇒ **≈ 550 MB for 50 hours**.
4. Any candidate's ±4 s representation is a **slice-and-pool of the cached bins** — free at training time.
5. Chunk boundaries: candidates within 4 s of a chunk edge read across the concatenated per-episode
   array, so no boundary artefacts.

Text: MiniLM-L6-v2 over `transcript_before` / `transcript_after` (≤ 400 chars each), batched, float16
→ `(n_cand, 768)`, ≈ 15 MB for 10 000 candidates.

Cache invalidation: every artifact directory carries a `_meta.json` with
`{preprocessing_version, candidate_config_version, feature_spec_hash, model_name, git_sha}`. A stage
recomputes an episode **iff** any of those changed or the output is missing. `--force` overrides.
Resume is therefore free and idempotent: re-running a partially completed extraction skips finished
episodes.

CPU/GPU: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` everywhere;
`torch.autocast` enabled only on CUDA (CPU autocast with bf16 is guarded behind a config flag and off
by default); batch size configurable and auto-halved on `RuntimeError: out of memory`;
`torch.set_num_threads` from config.

## 19. Model architectures

All implement `ml/src/slotify_rank/models/base.py::Ranker` — `score(batch) -> (B,)` plus optional
`accept_logit(batch) -> (B,)` — so training, evaluation, ablation and serving share one interface.
The heuristic implements the same interface, which is what makes the comparison honest.

| Key | Model | Inputs |
|---|---|---|
| `heuristic` | `HeuristicRanker` — the ported `scoreCandidate`, no learned parameters | rule inputs only |
| `mlp_hand` | 70 → 128 → 64 → 1 MLP (+ GBDT variant via scikit-learn `HistGradientBoostingRegressor` for a non-neural sanity check) | handcrafted |
| `text_only` | 1536 → 256 → 128 → heads | text |
| `audio_only` | 768 → 256 → 128 → heads (+ the 34 acoustic scalars, per the "learned speech representations and waveform features" definition) | audio |
| `concat` | three projections → concat(384) → 128 → heads | all three |
| `gated` | **primary** | all three |

**Gated multimodal ranker.** The input widths below are the **constructed** feature widths from §17,
not native model widths (`whisper-tiny.en` encoder is 384 wide; MiniLM outputs 384). `D_hand` is read
from the feature spec rather than hard-coded.

```
h_hand  = MLP(D_hand ≈ 70 → 128) : Linear → LayerNorm → GELU → Dropout(p) → Linear → LayerNorm
h_text  = MLP(1536 → 256 → 128)   same block structure
h_audio = MLP(768  → 256 → 128)   same block structure

g = sigmoid(Linear(concat[h_hand, h_text, h_audio] (384) → 3))      # one learned gate per modality
z = LayerNorm( g0*h_hand + g1*h_text + g2*h_audio )                 # 128
z = Dropout(p)(GELU(Linear(128 → 128)(z))) + z                      # residual block

score        = Linear(128 → 1)(z).squeeze(-1)
accept_logit = Linear(128 → 1)(z).squeeze(-1)
```

Gates are **logged per episode at evaluation time** — "the text gate dominates on narrated content,
the audio gate on conversational" is exactly the kind of finding that explains a result, and it comes
free from this design.

Defaults (`configs/model_gated_v1.yaml`): `hidden=128`, `dropout=0.2`, `activation=gelu`,
`norm=layer`, `aux_weight=0.25`.

## 20. Pair-generation algorithm

```
for episode e in split:
    L = [c for c in candidates(e) if c.label_source == "human" and c.quality_score is not None]
    pairs = [(a, b) for a, b in combinations(L, 2)
             if abs(a.quality_score - b.quality_score) >= min_score_gap]     # default 1.0
    # balance: cap per episode, stratify over gap size so easy 5-vs-1 pairs don't dominate
    pairs = stratified_sample(pairs, key=gap_bucket, n=max_pairs_per_episode)  # default 200
    y = +1 if a.quality_score > b.quality_score else -1
```

Invariants, enforced by `tests/test_pairs.py`:
- **both members always come from the same episode** — cross-episode pairs are meaningless because
  score scale is only comparable within an episode, and they would leak across splits;
- no pair may span two dataset splits (impossible by construction, asserted anyway);
- pair order is shuffled with the run seed; `gap_bucket` stratification prevents the loss being
  dominated by trivially separable pairs;
- if an episode yields zero pairs it contributes only to the auxiliary loss.

## 21. Training objective

```
ranking_loss = MarginRankingLoss(margin=1.0)(score_a, score_b, y)
acceptability_loss = BCEWithLogitsLoss(pos_weight=w)(accept_logit, is_acceptable)
total_loss = ranking_loss + aux_weight * acceptability_loss        # aux_weight = 0.25, configurable
```

`pos_weight = n_negative / n_positive` on the training split handles class imbalance; pair imbalance
is handled by the stratified cap in §20.

Training pipeline requirements, all in `training/loop.py`:
deterministic seeding (`random`, `numpy`, `torch`, `PYTHONHASHSEED`, `torch.use_deterministic_algorithms(True)`);
YAML config validated by a pydantic model; train/val/test manifests resolved from `data/splits/`;
checkpoint every epoch + `best.pt` selected on **validation NDCG@3**; early stopping (patience 15);
`clip_grad_norm_(1.0)`; AdamW (`lr 3e-4`, `weight_decay 1e-2`), cosine schedule with warmup;
CUDA autodetect with CPU fallback; AMP only on CUDA; `--resume` from the last checkpoint including
optimizer/scheduler/RNG state; metrics appended to `artifacts/models/{run_id}/metrics.jsonl`;
optional W&B behind `if config.wandb.enabled` — **local JSON/CSV/Markdown records are the source of
truth and the project runs fully offline.**

## 22. Evaluation metric definitions

All metrics are computed **per episode and then averaged over episodes** (macro), because episodes
differ wildly in candidate count.

- **Relevance / gain.** `rel = max(0, quality_score - 1)` maps 1–5 → 0–4; `gain = 2^rel - 1`.
- **NDCG@3** (headline) — `DCG@3 = Σ_{i=1..3} gain_i / log2(i + 1)`; `IDCG@3` from the same episode's
  human labels; episodes with `IDCG@3 == 0` are excluded and the exclusion count is reported.
- **Precision@3** — fraction of the top-3 with `quality_score ≥ 4`.
- **Recall@3** — of that episode's `quality_score ≥ 4` candidates, the fraction appearing in the top-3
  (capped at 3, so it is reported as `min(n_relevant, 3)`-normalised recall; the raw definition is
  stated in the artifact).
- **Binary F1** — on the auxiliary `is_acceptable` head at threshold 0.5, plus the threshold-free AP.
- **MRR** — reciprocal rank of the first `quality_score ≥ 4` candidate.
- **Pairwise ranking accuracy** — over held-out pairs with gap ≥ 1.0, fraction where the model orders
  them as the human did.
- **Per-episode table** — every metric per episode, written to CSV, so outliers are visible.
- **Latency** — mean and p95 wall-clock per candidate and per episode for the full serving path
  (transcribe → candidates → features → score), measured on this CPU and labelled with the hardware.
- **Confidence intervals** — bootstrap over **episodes** (10 000 resamples), 95 %, on NDCG@3 and on the
  relative improvement.

**Headline number, computed by `eval/report.py`, never typed by hand:**

```
ndcg_improvement_pct = 100 * (model_ndcg_at_3 - baseline_ndcg_at_3) / baseline_ndcg_at_3
```

written to `artifacts/evaluation/model_results.json` and interpolated into
`artifacts/reports/final_results.md`. The README and the evidence matrix link to the artifact rather
than restating the number.

## 23. Baseline and ablation matrix

**Model comparison** (all on the same held-out human-labelled test episodes):

| # | System | Modalities |
|---|---|---|
| 0 | `product_api_baseline_v1` — the paid path (whisper-1 transcript candidates + gpt-4o-mini enrichment). Reported **separately** as a product reference; never the headline denominator, because it needs credentials and is not stable across runs. | rules + paid APIs |
| 1 | **`heuristic_offline_v1`** — the canonical deterministic baseline and the headline denominator | rules |
| 2 | Handcrafted-feature MLP (+ GBDT variant) | hand |
| 3 | Transcript-only ranker | text |
| 4 | Audio-only ranker | audio (+ acoustic scalars) |
| 5 | Multimodal concatenation | all |
| 6 | **Gated multimodal ranker** | all |

**Required ablations** (each is model 6 with one change):

| Key | Change |
|---|---|
| `no_audio_embed` | drop the Whisper-encoder branch |
| `no_text_embed` | drop the MiniLM branch |
| `no_handcrafted` | drop the 70-d branch |
| `no_heuristic_score` | zero out features 66–70 (heuristic score + components) |
| `no_aux_head` | `aux_weight = 0` |
| `concat_fusion` | replace gated fusion with plain concatenation |

Each ablation runs with **3 seeds**; the table reports mean ± std. Output:
`artifacts/evaluation/ablation_results.csv` with columns
`variant, seed, ndcg@3, p@3, r@3, mrr, f1, pairwise_acc, latency_ms, n_test_episodes, git_sha, run_id`.

## 24. Episode-level leakage prevention

- Splitting is done by **episode id**, never by candidate (`data/splits.py`); the function's only input
  is the episode list.
- **Series-aware splitting is the default.** `episodes.jsonl.series_id` groups episodes from the same
  show/meeting series; the splitter uses `GroupShuffleSplit` on `series_id` so a host's voice, recording
  chain, and topics cannot appear on both sides. A `--group-by episode` escape hatch exists but emits a
  loud warning and is recorded in `split_statistics.json`.
- Target 70 / 15 / 15 by **duration** (not episode count), with a greedy balancing pass so no split is
  starved of labelled candidates.
- Feature normalisation statistics are fitted on train only (§16).
- `data/validate.py` runs as a hard gate before training and fails on: any `candidate_id` in two splits;
  any `series_id` in two splits (unless explicitly allowed); any test candidate with
  `label_source != "human"` used as ground truth; any candidate whose `preprocessing_version` differs
  from the manifest; any timestamp outside `[0, duration]`; any duplicate `candidate_id`.
- Audio windows: because candidates are episode-scoped and splits are episode-scoped, no window can
  straddle splits — asserted in `tests/test_splits.py`.

## 25. FastAPI contract

`ml/src/slotify_rank/serving/api.py`

```
POST /v1/rank-breakpoints
  multipart: audio=<file>            (or JSON {"audio_path": "..."} for local/offline use)
  fields: top_k=3, min_separation_s=30, edge_guard_s=20,
          content_type=podcast, include_explanations=true

200 →
{
  "episode_id": "req-8f2c1a",
  "model_version": "gated-v1.0.0+run_20260721_142233",
  "processing_time_ms": 4213,
  "n_candidates_considered": 187,
  "candidates": [
    {
      "rank": 1,
      "timestamp_seconds": 412.84,
      "model_score": 2.71,
      "acceptability_probability": 0.93,
      "candidate_sources": ["whisper_segment_end", "silence", "speaker_turn"],
      "explanation_features": {
        "pause_duration_s": 1.42,
        "is_sentence_end": true,
        "semantic_change_magnitude": 0.61,
        "speaker_change": true,
        "position_ratio": 0.38,
        "heuristic_score": 0.80,
        "modality_gates": {"hand": 0.41, "text": 0.78, "audio": 0.55}
      }
    }
  ]
}

GET  /v1/health   → {status, model_version, device, torch_version, warm}
GET  /v1/model-info → model card summary, feature spec hash, training run id
```

Errors: `400` unreadable/unsupported audio, `413` over `max_audio_mb`, `422` validation (pydantic),
`503` model not loaded. Structured JSON logs with a per-request id. Model and encoders load **once**
at startup (`lifespan`), never per request.

## 26. Existing-application integration strategy

Additive and reversible:

1. **New** `backend/src/services/ranker.ts` — `rankBreakpoints(audioPath, opts)` POSTs to
   `RANKER_URL` (default `http://localhost:8000`) with an `AbortSignal` timeout (`RANKER_TIMEOUT_MS`,
   default 120 000).
2. **Modify** `backend/src/routes/insert-sections.ts` — when `RANKER_ENABLED === "1"`, call the ranker
   and map its candidates onto the **existing `Slot` shape**; on any error or timeout, log and fall
   through to the current heuristic path unchanged. Response gains `model_version`,
   `acceptability_probability` per slot, and `source: "model" | "heuristic" | "fallback"`. `points`,
   `confidences`, `slots`, `sponsorStatements` keep their current shapes.
3. **Modify** `frontend/src/App.tsx` — read `slots[]` (rationale, pros/cons, acceptability) instead of
   only `points`/`confidences`; tag fabricated fallback slots as such in the UI.
4. Preview, merge, export, TTS, cloning: **zero changes.**
5. Backward compatibility contract: with `RANKER_ENABLED` unset, the app must be byte-for-byte
   behaviourally identical to today. `tests/integration/test_legacy_path.py` asserts this against the
   golden heuristic outputs from Phase 1.

## 27. Proposed directory structure

```
ml/
  pyproject.toml                 # python = "3.12", torch, transformers, sentence-transformers,
  README.md                      #   librosa, pydub, pyloudnorm, pyarrow, fastapi, uvicorn,
  configs/                       #   pydantic, typer, scikit-learn, pytest
    sources.yaml   candidates_v1.yaml   features_v1.yaml
    train_gated_v1.yaml   train_concat_v1.yaml   train_text_only_v1.yaml
    train_audio_only_v1.yaml   train_mlp_hand_v1.yaml
    ablations.yaml   eval_v1.yaml   smoke.yaml
  src/slotify_rank/
    __init__.py  cli.py                       # typer: slotify-rank <subcommand>
    config/settings.py  config/versions.py
    data/  sources.py fetch.py decode.py manifest.py splits.py stats.py validate.py
    transcribe/ whisper_asr.py
    candidates/ schema.py generators.py merge.py heuristic.py
    features/ spec.py acoustic.py transcript.py audio_embed.py text_embed.py cache.py normalize.py
    models/ base.py heuristic_ranker.py mlp.py text_only.py audio_only.py concat.py gated.py registry.py
    training/ pairs.py dataset.py loop.py checkpoint.py seed.py
    eval/ metrics.py evaluate.py ablations.py human_eval.py benchmark.py report.py
    labeling/ app.py store.py clips.py static/index.html
    serving/ api.py inference.py schemas.py
  tests/ unit/ integration/ fixtures/
data/            # gitignored
artifacts/       # gitignored except *.md / *.json / *.csv reports
docs/
  multimodal-ranking-mvp-plan.md      (this file)
  evaluation-evidence.md
  model-inference.md   dataset-card.md      labelling-guide.md
backend/
  ad_inserter/heuristic_config.py     # NEW: named, versioned heuristic profiles
  src/services/ranker.ts              # NEW
  scripts/dump-heuristic-golden.ts    # NEW: emits TS golden outputs for the parity test
```

## 28. Exact files to create, modify, move, delete

**Created in Phase 1 (actual):**

| File | Purpose |
|---|---|
| `config/heuristic_offline_v1.json` | **the canonical, language-neutral baseline configuration** — one file, three thin loaders, no duplicated constants |
| `ml/pyproject.toml` | package metadata, `requires-python = ">=3.12,<3.13"`, dependency-light (`pyyaml` only) |
| `ml/README.md` | Windows + bash setup, commands, parity workflow |
| `ml/configs/heuristic_v1.yaml` | **run** settings only (profile name, metric thresholds, output paths) — never baseline constants |
| `ml/src/slotify_rank/__init__.py` | `__version__` |
| `ml/src/slotify_rank/jsnum.py` | `js_round` / `js_to_fixed` — ECMAScript rounding, required for exact parity |
| `ml/src/slotify_rank/config/settings.py` | canonical-config loader + validation + repo-root discovery |
| `ml/src/slotify_rank/config/versions.py` | `PACKAGE_VERSION`, `CANDIDATE_SCHEMA_VERSION`, reserved Phase 2/3 stamps |
| `ml/src/slotify_rank/candidates/schema.py` | canonical candidate/episode schema (§11) |
| `ml/src/slotify_rank/candidates/heuristic.py` | **port of `mergeCandidates`, `scoreCandidate`, `selectTopSlots` and the route's slot finalisation**, with component-score breakdown |
| `ml/src/slotify_rank/evaluation/metrics.py` | NDCG@k, P@k, R@k, F1, MRR, pairwise accuracy, per-episode + macro aggregate |
| `ml/src/slotify_rank/cli.py` | `version`, `config show`, `heuristic rank`, `evaluate` |
| `ml/tests/conftest.py` | guarded pytest temp-root workaround (risk R19) |
| `ml/tests/test_heuristic.py` | golden parity + every scoring component, merge, spacing, tie-breaking, malformed input |
| `ml/tests/test_metrics.py` | every metric against hand-computed values, plus undefined-case policy |
| `ml/tests/test_cli.py` | CLI smoke: exit codes, machine-readable output, Windows paths |
| `ml/tests/fixtures/heuristic_golden.json` | **generated only by the TypeScript scorer**, committed |
| `backend/src/lib/heuristic-config.ts` | thin TS loader for the canonical config (necessary addition, not in the Phase 0 list) |
| `backend/ad_inserter/heuristic_config.py` | thin Python loader; profiles `PRODUCT_V1` (700 ms) and `LEGACY_CLI_V1` (500 ms) |
| `backend/scripts/dump-heuristic-golden.ts` | golden-fixture generator over 9 synthetic episodes covering all 11 required scenarios |

**Modified in Phase 1 (actual) — all behaviour-preserving:**

| File | Change | Behaviour |
|---|---|---|
| `backend/src/lib/candidates.ts` | constants read from `HEURISTIC_V1`; arithmetic and operation order untouched | **verified byte-identical** (SHA-256 of the golden output matched pre/post) |
| `backend/ad_inserter/analyze_cli.py` | silence params from `heuristic_config.PRODUCT_PROFILE` | **verified identical** on real audio |
| `backend/ad_inserter/analysis.py` | `detect_podcast_candidates` reads `LEGACY_CLI_PROFILE`; docstring records the divergence | **verified identical** (500 / −16 / −40) |
| `.gitignore` | `data/`, `artifacts/**` (reports exempted), `ml/.venv/`, `ml/.pytest-tmp/`, `*.parquet`, `*.npy` | — |

**Left unchanged, deliberately:** all of `frontend/` (including the fabricated-fallback debt),
`routes/insert-sections.ts`, `ad_inserter/cli.py`, `insert_ad.py`, `llm.py`, `mix.py`, `tts.py`, and
every other route and service.

**Later phases** (created, not in Phase 1): everything else under `ml/src/slotify_rank/`;
`backend/src/services/ranker.ts` and the `insert-sections.ts` / `App.tsx` edits (Phase 7);
`docs/model-inference.md`, `docs/dataset-card.md`, `artifacts/models/model_card.md` (Phase 7).

**Move / delete: none.** `backend/audio_tests/` stays where it is and becomes the smoke-test fixture
directory (documented as such, not as a dataset).

## 29. Local GPU and memory strategy

Given **no CUDA device** on this machine:

- Device selection is dynamic (`cuda` → `cpu`); CI/smoke path is CPU-only and is the tested path.
- `whisper-tiny.en` is the **default** encoder and transcriber. `whisper-base.en` is config-selectable
  and documented as "only with a GPU or an overnight budget".
- Transcription cost estimate for planning: `tiny.en` on CPU runs roughly 8–15× realtime ⇒ **50 h of
  audio ≈ 3.5–6 h wall clock**, run once, resumable, overnight. Encoder-embedding extraction is a
  second pass at a similar order and is fused into the same job to avoid decoding twice.
- Training touches **only cached tensors**: ~1 500 labelled candidates × (70 + 1 536 + 768) floats is
  a few MB. Full training of the gated model: **single-digit minutes on CPU**. This is why the caching
  design in §18 is not optional — it is what makes the project trainable on this laptop.
- `torch.set_num_threads` configurable (default `os.cpu_count() - 2`); batch sizes small by default
  (train 64, extraction 8) and auto-halved on OOM; float16 storage for all cached embeddings.
- Peak RSS budget: keep every stage under 6 GB so the 32 GB machine can run extraction while the dev
  server and browser are open.

## 30. Testing strategy

| Level | Tests |
|---|---|
| Unit | heuristic parity vs TS golden; metrics vs hand-computed NDCG/MRR/P@k fixtures; candidate merge (overlapping clusters, flag union, tolerance edges); pair generation invariants (same-episode, gap threshold, no cross-split); feature spec dimension `== 70`; normalisation fit/transform round-trip; config validation rejects unknown keys |
| Property | merge is idempotent; merge output is sorted and strictly increasing; NDCG ∈ [0,1]; NDCG of the ideal ordering == 1 |
| Integration | full pipeline on the 13 `audio_tests/` files (`smoke.yaml`): fetch-skip → transcribe → candidates → features → train 2 epochs on synthetic labels → evaluate → report, all under 5 minutes on CPU; FastAPI `TestClient` end-to-end on one 18 s file; Express→FastAPI round trip with the ranker up and, separately, down (fallback path) |
| Regression | `test_legacy_path` asserts `/api/insert-sections` with `RANKER_ENABLED` unset reproduces the Phase-1 golden outputs |
| Data | `validate.py` invoked as a pytest case over the real manifest |
| Manual | labelling UI checklist in `docs/labelling-guide.md`; the human-eval and benchmark protocols |

`pytest -m "not slow"` is the default; anything touching real model weights is marked `slow`.
Coverage target for `ml/src/slotify_rank/{candidates,features,eval,training}`: **≥ 80 %**. Node side
gains no test framework in this project (out of scope); the parity golden is generated by a plain
`tsx` script.

## 31. Reproducibility strategy

- One command per phase, all through `slotify-rank <subcommand>` (typer), documented in the README with
  copy-pasteable PowerShell **and** bash forms (the repo is developed on Windows/OneDrive paths —
  every path is quoted and constructed with `pathlib`).
- Global `--seed` (default 42) seeding `random`, `numpy`, `torch`, `PYTHONHASHSEED`;
  `torch.use_deterministic_algorithms(True)` with the documented exception list.
- Every artifact embeds `{git_sha, dirty_flag, run_id, config_hash, seed, python_version, torch_version,
  device, timestamps}` in a `_provenance` block. `run_id = {UTC timestamp}_{config_hash[:8]}`.
- Dataset determinism: `episodes.jsonl` sorted by `episode_id`; candidate ids derived from
  `(episode_id, timestamp_ms)`; splits computed from a hash of `series_id` + seed, so adding episodes
  does not reshuffle existing assignments.
- Dependency pinning: `pyproject.toml` ranges + a committed `ml/requirements.lock` (`uv pip compile`).
- No network at training or evaluation time — HF models are pre-downloaded into a local cache by
  `slotify-rank prepare-models` and `HF_HUB_OFFLINE=1` is set for the training entrypoint.

## 32. Artifact and experiment versioning

```
artifacts/
  dataset/     dataset_statistics.json   split_statistics.json   candidate_statistics.json
  models/      {run_id}/ {config.yaml, best.pt, last.pt, metrics.jsonl, feature_stats.json, _provenance.json}
               model_card.md
  evaluation/  baseline_results.json  model_results.json  ablation_results.csv
               per_episode_results.csv  human_preference_results.json
  benchmarks/  editing_time_raw.csv   editing_time_results.csv
  reports/     final_results.md   {run_id}_report.md
```

`artifacts/**` is gitignored **except** `*.md`, `*.json`, `*.csv` report files, which are committed so
the repository itself is the evidence trail. `final_results.md` is generated by `eval/report.py` from
the JSON — **no metric is ever hand-copied into documentation**, and a test asserts that the numbers
in `final_results.md` match the JSON they claim to come from.

## 33. Capability-to-evidence matrix

Maintained canonically in `docs/evaluation-evidence.md`. Summary:

| Capability | Required evidence | Generating command | Artifact | Status |
|---|---|---|---|---|
| **Reproducible deterministic baseline** | TS↔Python parity on golden fixtures; serialized component scores; metrics module | `npx tsx scripts/dump-heuristic-golden.ts` + `pytest` | `ml/tests/fixtures/heuristic_golden.json`, test run | ✅ **verified (Phase 1)** |
| Multimodal PyTorch ranker | architecture + trained checkpoint + param count + modality gates | `slotify-rank train --config configs/train_gated_v1.yaml` | `artifacts/models/{run_id}/best.pt`, `model_card.md` | not started |
| X % NDCG@3 improvement over `heuristic_offline_v1` | baseline + model test metrics on held-out **human** labels, bootstrap CI | `slotify-rank evaluate --config configs/eval_v1.yaml` | `artifacts/evaluation/{baseline,model}_results.json`, `reports/final_results.md` | not started |
| Modality contribution | 6-variant comparison + 6 ablations × 3 seeds | `slotify-rank ablate --config configs/ablations.yaml` | `artifacts/evaluation/ablation_results.csv` | not started |
| `processed_audio_hours` ≥ 50 | measured duration by source, licence and content type | `slotify-rank dataset-stats` | `artifacts/dataset/dataset_statistics.json` | not started |
| `generated_candidate_count` ≥ 10 000 | measured counts, broken out by label source | `slotify-rank candidate-stats` | `artifacts/dataset/candidate_statistics.json` | not started |
| `human_labelled_candidate_count` ≈ 1 500 over `human_labelled_audio_hours` ≈ 8–12 | label ledger by `label_source` and annotator | `slotify-rank dataset-stats` | `artifacts/dataset/dataset_statistics.json` | not started |
| `held_out_evaluation_candidate_count` | human-labelled candidates in the test split only | `slotify-rank validate` + `dataset-stats` | `artifacts/dataset/split_statistics.json` | not started |
| Y % human agreement (A1) | blind evaluation, ≥1 non-author evaluator, A1/A2/A3 with definitions | `slotify-rank human-eval --split test` | `artifacts/evaluation/human_preference_results.json` | not started |
| Editing-time reduction (measured) | timed counterbalanced study, ≥3 participants | `slotify-rank benchmark-report` | `artifacts/benchmarks/editing_time_results.csv` | not started |
| FastAPI + product integration | end-to-end test upload → rank → preview → export | `pytest ml/tests/integration -m e2e` | test report + `docs/model-inference.md` | not started |

**No claim is marked complete until its command succeeds and its artifact exists in the repo.**

## 34. Risks and mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **No GPU** (Intel Iris Xe only) | high | tiny.en default; bin-and-cache embeddings (§18); 0.35 M-param model; overnight one-shot extraction; every step resumable |
| 2 | **Python 3.14 is the default interpreter**; torch/whisper wheels unavailable | high | `ml/` pins 3.12 in its own venv (`py -V:3.12 -m venv ml/.venv`); `slotify-rank doctor` fails loudly on a wrong interpreter |
| 3 | **Labelling is the real bottleneck** — 1 500 labels ≈ 11–17 h of human work | high | keyboard-first UI, pairwise fast mode, stratified queue, resumable sessions; MVP target 1 200–1 500 with a documented path to 3 000 |
| 4 | Too few labelled candidates → noisy NDCG@3 with wide CIs | high | report bootstrap CIs always; if the CI includes zero, say so; increase test episodes before increasing model complexity |
| 5 | Labelling bias: annotator is also the author | high | blind the UI to all system scores; blind pairwise for the headline human claim; recruit ≥ 2 evaluators for A3 and report κ |
| 6 | Weak labels contaminating test metrics | high | separate files, `label_source` on every row, `validate.py` hard-fails on non-human test labels |
| 7 | Series leakage (same host in train and test) | med | series-aware `GroupShuffleSplit` by default; leakage assertions in `validate.py` |
| 8 | Baseline weakened accidentally during the port | high | golden-file parity test to 1e-9 against the live TS implementation; parity test is a release gate |
| 9 | Candidate yield below 10 000 | med | 7 generators + fixed-interval coverage fill; `max_per_minute` knob; the number is measured, not assumed, and the corpus can grow |
| 10 | Licensing on scraped podcast audio | med | audio never committed; per-episode licence recorded; only PD/CC sources in `sources.yaml` |
| 11 | Whisper transcription errors propagate into features | med | store word confidences; add `transcript_confidence` to features; report metrics on a high-confidence subset as a robustness check |
| 12 | AMI is meetings, not podcasts — domain gap | med | mix in Internet Archive podcasts + narrated LibriVox; report per-`content_type` metrics so the gap is visible rather than hidden |
| 13 | Editing benchmark n is small | med | within-subject counterbalanced design maximises power; report per-pair values and CI; state n prominently |
| 14 | Serving latency (transcribe on request) makes the demo feel broken | med | measure and report; cache by audio sha256; stream progress; document that a 30-min episode is a background job on CPU |
| 15 | Scope creep into rewriting the working product | med | integration is one new service file + one flag; preview/export untouched; regression test on the legacy path |
| 16 | OneDrive path sync-locks during long extraction jobs | low | `data/` and `artifacts/` documented as candidates for a non-synced location; all paths quoted, `pathlib`-built |
| 17 | **Baseline drift between the route and the golden generator.** `insert-sections.ts:181-260` is mirrored, not imported, by `scripts/dump-heuristic-golden.ts`. An edit to the route's finalisation would silently desynchronise the baseline from the product. | med | The mirrored block cites the exact route lines it reproduces; §5 schedules extraction of `finalizeSlots()` in Phase 7. Until then, treat any edit to `insert-sections.ts:136-260` as a baseline change requiring fixture regeneration. |
| 18 | OneDrive rejects hardlinks, so `uv` installs fail with `os error 396` | low | `UV_LINK_MODE=copy` documented in `ml/README.md`; verified working |
| 19 | `%TEMP%\pytest-of-danda` on this machine denies enumeration (`WinError 5`), breaking every `tmp_path` test | low | `ml/tests/conftest.py` redirects pytest's temp root to `ml/.pytest-tmp` **only when** the default is unusable; a no-op on healthy machines. System ACLs deliberately left untouched. |
| 20 | Pre-existing frontend lint/build failures (4 unused-variable errors in `App.tsx`) block `npm run build` | low | Pre-dates Phase 1 and is unrelated to it (`git diff -- frontend` is empty). Left unfixed to avoid unrelated cleanup; scheduled with the Phase 7 frontend work. |

## 35. MVP scope vs later improvements

**In the MVP:** silence/transcript/RMS/spectral/speaker candidate generation; 70 handcrafted features;
frozen tiny.en + MiniLM embeddings; six model variants; six ablations; pairwise + auxiliary training;
NDCG@3-led evaluation with CIs; labelling app; blind human eval; editing benchmark; FastAPI service;
Express feature-flag integration; dataset/model cards; evidence matrix.

**Explicitly later:** fine-tuning any pretrained encoder; listwise losses (ListNet/LambdaRank);
cross-modal attention or transformer fusion; music-specific learned ranking; multilingual; sponsor-aware
(topic/tone-conditioned) placement; active learning for label selection; ONNX/quantised serving;
multi-ad-per-episode joint optimisation; Docker/CI/CD.

**Never (out of scope by instruction):** Kubernetes, distributed training, feature stores, cloud
infrastructure, extra microservices, architectural rewrites.

## 36–37. Implementation phases, acceptance criteria, and commands

Commands assume `cd ml` with the 3.12 venv active. PowerShell users: `ml\.venv\Scripts\Activate.ps1`.

### Phase 0 — Audit and architecture ✅
Deliverables: this document + `docs/evaluation-evidence.md`.
**Acceptance:** plan cites real paths/symbols; no product behaviour changed. — **met.**

### Phase 1 — Reproducible heuristic baseline ✅
Files: §28. Verified 2026-07-22.

```powershell
# 1. Regenerate the golden fixture from the TypeScript product scorer
cd backend
npx tsx scripts/dump-heuristic-golden.ts --out ..\ml\tests\fixtures\heuristic_golden.json
npm run typecheck

# 2. Python environment (once)
cd ..\ml
$env:UV_LINK_MODE = "copy"
uv venv --python 3.12.13 .venv
uv pip install --python .\.venv\Scripts\python.exe -e ".[dev]"

# 3. Tests, including TypeScript/Python parity
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest --cov=slotify_rank --cov-report=term-missing

# 4. CLI
.\.venv\Scripts\python.exe -m slotify_rank.cli version
.\.venv\Scripts\python.exe -m slotify_rank.cli config show
.\.venv\Scripts\python.exe -m slotify_rank.cli heuristic rank --input episodes.json --output rankings.json
.\.venv\Scripts\python.exe -m slotify_rank.cli evaluate --predictions rankings.json --labels labels.json
```

**Acceptance criteria — met:**

| Criterion | Result |
|---|---|
| Product-path rankings reproduced deterministically | ✅ 9/9 golden episodes, exact equality (tolerance 0.0) |
| Every candidate and score component serialized | ✅ all 13 required schema fields + component breakdown |
| Saved rankings evaluable against labelled fixtures | ✅ `evaluate` command, 6 metrics |
| TypeScript ↔ Python comparison | ✅ 45 parity assertions across merge, score, selection, full pipeline |
| Baseline drift detectable | ✅ parity suite fails on any divergence; config version stamped into every record |
| Tests run without paid APIs, network or model downloads | ✅ 169 tests, 3.0 s, offline |
| `npm run typecheck` clean | ✅ |
| Product behaviour unchanged | ✅ golden output SHA-256 identical pre/post refactor; `analyze_cli` identical on real audio |

### Phase 2 — Dataset and labelling workflow ✅ (workflow complete; corpus not acquired)

Verified 2026-07-22. Final command names differ from the sketch above; these are the real ones.

```powershell
cd ml
uv pip install --python .\.venv\Scripts\python.exe -e ".[dev,label]"

python -m slotify_rank.cli dataset import-local --sources configs/sources.yaml
python -m slotify_rank.cli dataset fetch --sources configs/sources.yaml   # only networked command
python -m slotify_rank.cli dataset probe
python -m slotify_rank.cli dataset normalize
python -m slotify_rank.cli candidates generate --config configs/dataset_v1.yaml
python -m slotify_rank.cli dataset split --config configs/splits_v1.yaml
python -m slotify_rank.cli dataset validate --deep
python -m slotify_rank.cli dataset stats
python -m slotify_rank.cli label serve --port 8000
python -m slotify_rank.cli label export
```

**Acceptance criteria — met:**

| Criterion | Result |
|---|---|
| An episode goes import → probe → normalize → candidates → split → validate → stats | ✅ 7 smoke fixtures, end to end, all exit 0 |
| Statistics distinguish human / weak / unlabelled | ✅ eight quantities tracked separately in `label_statistics.json` |
| `dataset_statistics.json` and `split_statistics.json` exist | ✅ plus `candidate_statistics.json`, `label_statistics.json`, `dataset_summary.md`, `validation_report.json` |
| Killing the labeller mid-session loses at most one item | ✅ loses **zero** — every rating is committed on submit; resumption verified across a simulated restart |
| Splits are leakage-safe | ✅ series-grouped; episode-level and series-level leakage each have a test that makes validation fail |
| Synthetic product padding excluded | ✅ pinned by the schema constructor, refused by the label DB, failed by `validate`, excluded from every statistic |
| Tests offline, no GPU, no model downloads, no paid APIs | ✅ 379 tests, ~37 s, 91 % coverage |
| Product behaviour unchanged | ✅ no frontend, route, service or `ad_inserter` change; `npm run typecheck` clean; golden parity re-verified |

**Deliberately deferred to Phase 3:** transcription (no Whisper download), learned embeddings, and
any PyTorch dependency. `transcript_segment_end` consumes a supplied timestamped transcript but never
generates one, and every generation report states when none was available.

**Not achieved, and not claimed:** the 50-hour / 10 000-candidate / 1 500-label targets. The smoke
dataset is ~4.6 minutes over 7 fixtures yielding 20 candidates and 0 labels. See
`docs/evaluation-evidence.md` for what is populated versus merely implemented.

### Phase 3 — Feature and embedding pipeline (complete)

Shipped commands differ from the sketch above: stages are named subcommands under the
existing parser rather than one `features --all` flag, so each can be run, retried and
cached independently.

```powershell
cd ml; $py = ".\.venv\Scripts\python.exe"
& $py -m slotify_rank.cli pipeline features --deep    # transcribe -> ... -> validate -> stats
& $py -m slotify_rank.cli pipeline features --deep    # second run: all cache hits
& $py -m slotify_rank.cli pipeline status
```

**Acceptance — met.** One command builds every feature; the second identical run recomputed
nothing (4/4 model stages cache-hit, 102 s → 1.2 s); an interrupted stage is detected via a
`partial` ledger entry and reprocessed; a `FEATURE_SPEC_VERSION` bump forced recompute of
exactly the affected stages (acoustic + text embeddings) and correctly left transcription and
audio embeddings cached.

**One correction to the sketch.** Cache validity is *not* verified by mtime assertions. Timestamps
are never part of a cache decision: a newer file is not a correct file, and mtimes survive neither
a OneDrive sync nor a fresh clone. Validity is a SHA-256 over every input that determines the
artifact's bytes — audio checksum, model id and revision, config digests, library versions. See
`docs/feature-pipeline.md` §7.

Dimensions recorded: whisper-tiny.en encoder **384** (not 768), MiniLM native **384**, constructed
transcript vector **1536** = 384 × 4. 110 handcrafted scalars, each with a missing mask.

**No model was trained.** Phase 3 produces features only.

### Phase 4 — Training
```bash
slotify-rank train --config configs/train_mlp_hand_v1.yaml
slotify-rank train --config configs/train_text_only_v1.yaml
slotify-rank train --config configs/train_audio_only_v1.yaml
slotify-rank train --config configs/train_concat_v1.yaml
slotify-rank train --config configs/train_gated_v1.yaml
slotify-rank train --config configs/train_gated_v1.yaml --resume    # resumability check
```
**Acceptance:** every variant trains from a config file on CPU and emits a checkpoint plus
`metrics.jsonl`; best checkpoint is selected on validation NDCG@3; the same seed reproduces the same
validation metric.

### Phase 5 — Evaluation and ablations
```bash
slotify-rank evaluate --config configs/eval_v1.yaml --models heuristic,mlp_hand,text_only,audio_only,concat,gated
slotify-rank ablate --config configs/ablations.yaml --seeds 3
slotify-rank report
```
**Acceptance:** `baseline_results.json`, `model_results.json`, `ablation_results.csv`,
`per_episode_results.csv` and `reports/final_results.md` are produced; the NDCG@3 improvement is
computed by code; bullet 1's evidence exists end to end.

### Phase 6 — Human and workflow evaluation
```bash
slotify-rank human-eval --split test --mode blind-pairwise --port 8011
slotify-rank human-eval-report
slotify-rank benchmark-run --participants 3 --episodes 6     # prints the protocol + session URLs
slotify-rank benchmark-report
```
**Acceptance:** `human_preference_results.json` contains A1/A2/A3 with their definitions, n, and CIs;
`editing_time_results.csv` contains raw and aggregate rows; both numbers are whatever was measured.

### Phase 7 — Product integration and polish
```bash
slotify-rank serve --host 127.0.0.1 --port 8000
cd backend && RANKER_ENABLED=1 npm run dev        # PowerShell: $env:RANKER_ENABLED="1"; npm run dev
cd frontend && npm run dev
pytest ml/tests/integration -m e2e
cd frontend && npm run lint && npm run build
```
**Acceptance:** upload → ML-ranked points → preview → export works in the browser; with the ranker
down the legacy path still works and matches the Phase-1 golden; `docs/model-inference.md`,
`docs/dataset-card.md`, `artifacts/models/model_card.md`, and an updated README exist; every completed
row of the evidence matrix links to a real artifact.

## 38. Remaining questions

### Resolved by the locked decisions (§0)

| Question | Resolution |
|---|---|
| Compute budget / external GPU | **Laptop-only, CPU-first.** No GPU assumed; no cloud training. |
| Corpus choice — AMI as primary? | **No.** Podcasts/interviews/narrated/conversational are primary; AMI is supplemental and excluded from the test split. |
| Benchmark participants | **≥ 3 required, 5 targeted.** |
| Second evaluator | **≥ 1 non-author evaluator required, 2 preferred.** |
| Paid-API baseline | Evaluate both; **`heuristic_offline_v1` is the headline denominator**, `product_api_baseline_v1` is reported separately. |
| Labelling volume | **≈ 1 500 human-labelled candidates over ≈ 8–12 h**, in three stages (200–300 → 750 → 1 500). |
| Music path | Confirmed out of scope for the learned model; the heuristic keeps serving `mode=song`. |
| Frontend fabricated fallback | Product debt; excluded from all evaluation; replaced in Phase 7. |

### Still open

1. **Annotator time budget in calendar terms.** ≈1 500 pointwise labels is 11–17 hours of human work
   at 20–30 s each. Confirming *when* that happens determines whether Phase 4 starts against the
   750-label checkpoint or waits for the full set.
2. **Who the non-author evaluator(s) will be**, and whether they can be briefed on the rubric without
   being told which system produced which clip.
3. **Is 50 h hard?** If curating 50 h of properly-licensed *podcast* audio proves slower than
   expected, is a smaller, higher-quality corpus with an honest smaller number acceptable? This is
   now more likely than under the Phase 0 plan, because AMI's easy ~100 h is no longer primary.
4. **Podcast Index / CC-feed licence verification** — feed-level licence metadata is often wrong.
   How much manual verification is acceptable per episode before a source is dropped?
5. **Deployment target.** Anything beyond localhost? *Assumption:* no — local demo plus a video.
6. **Whether `ml/.venv` and `data/` should live outside OneDrive.** Sync locks already forced
   `UV_LINK_MODE=copy` (risk R18); long extraction jobs in Phase 3 may make relocation worthwhile.
</content>
