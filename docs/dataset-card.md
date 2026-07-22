# Dataset card — Slotify breakpoint corpus

**Status: workflow implemented, corpus not yet acquired.** As of Phase 2 the repository contains the
complete pipeline for building this dataset and a ~4.6-minute *smoke* dataset used to test it. The
measured numbers below are read from `artifacts/dataset/` and are not targets.

Every quantity in this document comes from a generated artifact. Nothing here is typed by hand.

```powershell
cd ml
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset stats
```

---

## What this dataset is

Candidate advertisement breakpoints in spoken-word audio, each with a timestamp, the signals that
proposed it, the production heuristic's score for it, and — for a reviewed subset — a human
naturalness rating on a 1–5 rubric.

It exists to train and evaluate a ranker that orders candidate breakpoints better than the
deterministic `heuristic_offline_v1` baseline that ships in the product today.

## Two layers, tracked separately

| Layer | What it is | Target | Fully labelled? |
|---|---|---|---|
| **Processed corpus** | Audio decoded, normalized, and turned into a candidate pool | 50+ hours, 10 000+ candidates | No |
| **Human-reviewed subset** | Candidates a person rated against the rubric | ~1 500 candidates over ~8–12 hours | Yes, by definition |

Collapsing these into one "labelled dataset" figure would overstate the human effort by roughly an
order of magnitude, so the statistics artifacts never do. The eight tracked quantities are:

```text
processed_audio_hours                 audio decoded and normalized
processed_episode_count               episodes at status 'normalized'
generated_candidate_count             real candidates (synthetic padding excluded)
human_labelled_candidate_count        candidates a person rated
human_labelled_audio_hours            duration of the episodes containing those labels
weakly_labelled_candidate_count       never added to the human count
unlabelled_candidate_count            the remainder
held_out_evaluation_candidate_count   human-labelled AND in the test split
```

## Target domain

Podcasts, interviews, conversational recordings and narrated spoken word.

- **Music is out of domain** and is excluded from headline hours and from the test partition, even
  though the product supports a song mode.
- **Meeting corpora (AMI and similar) are supplemental only.** They may pad the training corpus but
  must never dominate it, and `dataset validate` fails if the test partition contains no
  podcast-like audio.

`TARGET_DOMAIN_CONTENT_TYPES` in `ml/src/slotify_rank/data/schema.py` is the machine-readable
definition: `podcast`, `interview`, `conversational`, `narrated`.

---

## Sources and licensing

Three source types, declared in `ml/configs/sources.yaml`:

| Type | Where it comes from | Licence requirement |
|---|---|---|
| `local_file` | Anywhere on your machine; copied into `data/raw/` | **None assumed.** Private, never redistributed. |
| `direct_download` | One direct file URL | **`license_name` and `license_url` are mandatory.** Enforced at parse time. |
| `existing_repository_fixture` | Already in the checkout | None assumed. Smoke testing only. |

There is no scraping path and there will not be one. Remote sources require a direct audio URL;
landing pages, feeds, credentialed URLs and URLs with token query parameters are all rejected by
`load_sources`. The registry is committed, so it is also checked for anything that looks like a
credential.

**Your licensing responsibilities.** Registering a URL here asserts that you have the right to
download and process that file, and that the licence you named is the real one. The code enforces
that you *state* a licence; it cannot verify that the statement is true. If you are unsure, download
the file yourself and register it as a `local_file` — private local use, no redistribution, no
licence claim.

Audio is **never committed to Git.** `data/` is ignored in its entirety.

---

## How the audio is processed

1. **Import / fetch** — SHA-256 computed and recorded. Downloads verify against `expected_sha256`
   when declared; a mismatch deletes the partial file and aborts. Existing files are never silently
   replaced.
2. **Probe** — `ffprobe` measures duration, sample rate, channel count and format. Nothing is
   inferred from the file extension. Zero-length files, files with no audio stream, and files with
   no positive duration are rejected.
3. **Normalize** — `ffmpeg` renders **16 kHz mono PCM s16 WAV** into `data/normalized/`. Writes are
   atomic. The cache is keyed on `(source sha256, preprocessing_version)`, so an unchanged file is
   never re-rendered and a version bump invalidates every render at once.

4. **Transcribe** (Phase 3) — local `openai/whisper-tiny.en` produces a timestamped transcript in
   `data/transcripts/`, in integer milliseconds. No paid API and no key; the only network access is
   the one-time model download. Long audio is decoded in overlapping 30 s chunks and the overlap is
   reconciled. An empty transcript is recorded as a **failure**, never stored as an empty success.
5. **Features** (Phase 3) — 110 handcrafted scalars per candidate plus frozen Whisper speech
   representations (384-d) and frozen MiniLM transcript-context embeddings (384-d native, 1536-d
   constructed). Every artifact is keyed on a cache identity covering its audio checksum, model id
   and revision, config digests and library versions. See `docs/feature-pipeline.md`.

The original stays outside Git and is untouched. **The product's own upload, preview and export
paths are unaffected** — they still operate on the user's original file at its own sample rate.
Phase 3 is likewise offline only: nothing in it touches the Node API, the React app, or the
production ranking path.

---

## Candidates

Five generators, then deterministic merging:

| Generator | Question it answers |
|---|---|
| `silence` | Is there a product-length pause here? Uses the canonical `heuristic_offline_v1` thresholds. |
| `pause` | Is there a *short* pause the product would reject? The informative near-misses. |
| `rms_minimum` | Is this locally quiet without being silent? |
| `transcript_segment_end` | Does a transcript segment end here? Only if you supply a timestamped transcript. |
| `fixed_interval` | A regular grid, ignoring the audio. Most land mid-sentence — that is the point, they are the negative class. |

Merging collapses candidates within `merge_tolerance_ms` (400 ms) using the *product's* survivor
rule, so a dataset candidate sits where a product candidate would. Source flags are unioned onto the
survivor, so nothing about provenance is lost.

### Product padding is not a candidate

When the product cannot find three real candidates it invents slots at fixed ratios. Those
timestamps are not evidence about the audio. They are recorded (with `--include-product-padding`)
purely for auditing, and every one carries:

```text
is_synthetic = true
eligible_for_labelling = false
eligible_for_evaluation = false
```

The schema's constructor rejects any other combination, the labelling database refuses to register
them, and `dataset validate` fails if one is ever marked eligible. They never enter a count, a
training set, or a metric.

---

## Splits

Grouped by `series_id`, never by episode or candidate. Two episodes of one show share hosts, room,
mic chain and vocabulary; training on one and testing on the other measures memorisation.

- Algorithm `split-grouped-greedy-v1.0.0`, deterministic under a configurable seed.
- Balanced on **duration**, not episode count.
- Target 70 / 15 / 15.
- Out-of-domain material is kept out of the test partition.
- **Below 6 independent groups the splitter refuses to pretend.** It emits a single `development`
  partition and marks the result degraded, so no one reports a "held-out" metric from a smoke
  dataset.
- A split manifest is immutable for its version. Changing it requires a version bump or `--force`.

---

## Labels

SQLite at `data/labels/labels.sqlite3`, exported to versioned JSONL. One active label per
(annotator, candidate), enforced by a unique index; re-rating updates in place. Annotator ids are
pseudonymous and no personal information is stored. See [`labelling-guide.md`](labelling-guide.md)
for the rubric.

---

## Known limitations

- **The corpus does not exist yet.** The current dataset is seven repository fixtures totalling
  ~4.6 minutes. It tests the pipeline; it is not a training corpus and the statistics say so.
- **No transcripts.** Phase 2 does not generate them. `transcript_segment_end` contributes nothing
  until you supply timestamped transcripts, and every generation report states this explicitly.
- **Silence detection is an independent reimplementation** of pydub's semantics against the same
  canonical thresholds — not a bit-exact port. Bit-exact parity is asserted for the *scorer*, which
  is the part that must match the product.
- **Single-annotator labels have no inter-rater agreement.** Any agreement figure requires a second
  annotator.
- **Content type is operator-declared.** Nothing verifies that a file marked `podcast` is one.

## What is committed

| Committed | Ignored |
|---|---|
| `ml/configs/*.yaml` (source registry, generation, split settings) | `data/` — all audio, manifests, databases, exports |
| `artifacts/dataset/*.json` and `dataset_summary.md` | normalized renders, clip cache |
| the code and tests | `.venv`, coverage, caches |
