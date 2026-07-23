# Human-labelling workflow (Phase 5A)

This document is the operator's guide to the Phase 5A pipeline: register a real
target-domain corpus, process it into candidates and features, build a
stratified labelling queue, run resumable human-labelling sessions, and check
whether the Phase 5B experiment gate is met.

Nothing here trains a model. Phase 5B (the real baseline-vs-model comparison)
begins **only** when `experiment readiness` reports the gate as met — and it
cannot be met until a person has actually labelled candidates. The repository
ships with **zero human labels**, so the terminal step below is starting the
pilot labelling session.

All commands run from the `ml/` directory with its own 3.12 virtual environment:

```powershell
cd ml
.\.venv\Scripts\python.exe -m slotify_rank.cli <command>
```

CPU-only throughout. The only networked step is `dataset fetch`; the only model
download is the one-time Whisper/MiniLM fetch during `pipeline features`.

---

## 1. The round-1 corpus

`ml/configs/sources_real_v1.yaml` registers **six real, public-domain,
target-domain series** (ten episodes), every one verified against the Internet
Archive metadata API — real identifier, real file, real `licenseurl`:

| Series | Content type | Licence | Episodes |
|---|---|---|---|
| `art-of-war` | narrated | Public Domain (LibriVox) | 1 |
| `two-br-02-b` | conversational (dramatic reading) | Public Domain (LibriVox) | 1 |
| `short-nonfiction-013` | narrated | Public Domain (LibriVox) | 1 |
| `short-nonfiction-072` | narrated | Public Domain Mark 1.0 | 3 |
| `short-nonfiction-064` | narrated | Public Domain Mark 1.0 | 2 |
| `short-nonfiction-067` | narrated | Public Domain Mark 1.0 | 2 |

Only public-domain audio is used. To add CC-BY podcast/interview feeds later,
append one `direct_download` entry per episode with `license_name` and
`license_url` set — `load_sources` rejects any remote entry that omits them, and
no code change is needed.

## 2. Acquire and process the corpus

Fetch a subset (or all) of the registered episodes, then run the standard
Phase 2 → Phase 3 pipeline. Every stage is resumable and skips work already
done, so re-running after an interruption costs only what was lost.

```powershell
# Fetch (networked). Omit --source-id to fetch every registered episode.
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset fetch `
    --sources configs\sources_real_v1.yaml `
    --source-id snf013-towrite --source-id snf072-newmadrid `
    --source-id snf064-dayofinfamy --source-id snf067-ramanujan `
    --source-id art-of-war-0304 --source-id two-br-02-b-01

.\.venv\Scripts\python.exe -m slotify_rank.cli dataset probe
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset normalize
.\.venv\Scripts\python.exe -m slotify_rank.cli candidates generate --config configs\dataset_v1.yaml

# Phase 3 features (downloads whisper-tiny.en + MiniLM once; CPU).
.\.venv\Scripts\python.exe -m slotify_rank.cli pipeline features

# Freeze the series-aware split for the real corpus (v2; v1 was the smoke split).
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset split --config configs\splits_v2.yaml
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset validate --split-version v2
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset stats    --split-version v2
```

Round-1 processed corpus (six real episodes, one per series), as measured by
`dataset stats` and written to `artifacts/dataset/dataset_statistics.json`:
~0.8 h of audio and ~575 real eligible candidates, of which the feature pipeline
assembled a complete multimodal record for the large majority (the rest are
`audio_only`, where a segment had no usable transcript context). These are
**generated, unlabelled** candidates — not human labels.

## 3. Build the stratified labelling queue

The queue is deterministic and stratified so labels are not concentrated on
whatever the heuristic already likes. Hard strata are (dataset split ×
heuristic-score tertile); episode and series spread is enforced by round-robin
and a per-episode cap; every other dimension is reported as coverage.

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli label queue `
    --config configs\labelling_queue_v1.yaml --split-version v2
# -> data\labels\queue_v1.json
```

The artifact records the score-tertile boundaries, the candidate- and
split-manifest hashes it was built from, and a coverage breakdown across split,
score stratum, series, episode, candidate source, position, silence duration,
sentence-boundary status and transcript availability. It carries four stages:

* **pilot** (~24): a diverse first pass to validate the rubric and the context
  window before the bulk run;
* **primary** (~256): the rest of the unique set;
* **overlap** (~40): a subset flagged for a *second* annotator, for
  inter-annotator agreement later;
* **consistency** (~12): re-presented under a distinct `presentation_id` to
  measure intra-annotator consistency — never counted as unique candidates.

The queue is immutable once labels are collected against it: to change it, bump
`version` in the config (a byte-different queue of the same version is refused).

> **Note.** `signal_disagreement` coverage is 0 in round 1 because candidates
> were generated before transcripts existed, so their `sentence_end` field is
> null. Regenerating candidates with transcript-aware sentence boundaries would
> populate it; transcript *availability* is already 100% on the selected set.

## 4. Run the labelling session (the terminal Phase 5A step)

Start the local UI, restricted to the queue. The interface hides the heuristic
score, candidate source, split and repeat status by default, so the annotator is
never primed by the systems under test. Sessions are resumable: killing the
process loses at most the in-flight item.

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli label serve `
    --queue data\labels\queue_v1.json --port 8000
# Open http://127.0.0.1:8000/ and enter an annotator pseudonym.
```

Score each candidate 1–5 on the naturalness rubric (`docs/labelling-guide.md`),
mark acceptable / unusable, optionally add a note. Start with the ~24 pilot
candidates, confirm the rubric and context feel right, then continue into the
primary set. Target **250–300 unique labels** for the first genuine experiment.

Check quality and export as you go (both are safe to re-run):

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli label check  --queue data\labels\queue_v1.json --split-version v2
.\.venv\Scripts\python.exe -m slotify_rank.cli label export --dataset-version v1
```

`label check` reports invalid ratings, orphaned labels, acceptability
inconsistencies, candidates labelled outside the queue, split drift and manifest
drift — as **warnings**, never by altering a human judgement.

## 5. Check the Phase 5B gate

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment readiness `
    --queue data\labels\queue_v1.json --split-version v2
```

Phase 5B may begin only when this reports `Phase 5B ready: True`. The gate
requires ≥ 200 human-labelled unique candidates, ≥ 8 episodes, ≥ 6 series
(≥ 4 train / ≥ 1 validation / ≥ 1 test), usable within-episode pairs in train
and validation, at least one graded-relevant candidate in test, all labels
passing integrity checks, and a complete multimodal feature record for every
labelled candidate. Add `--require-ready` to make a failing gate exit non-zero
(so a training script can depend on it). The thresholds are fixed in
`ReadinessGate` and are not to be weakened to advance the phase.

When the gate passes, freeze an immutable snapshot before training:

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment freeze `
    --snapshot-version v1 --split-version v2 --queue data\labels\queue_v1.json
```

## 6. Where things are

| Artifact | Location | Committed? |
|---|---|---|
| Source registry | `ml/configs/sources_real_v1.yaml` | yes |
| Queue config | `ml/configs/labelling_queue_v1.yaml` | yes |
| Split config (real) | `ml/configs/splits_v2.yaml` | yes |
| Fetched audio, normalized WAV | `data/raw/`, `data/normalized/` | no (git-ignored) |
| Candidate / feature manifests | `data/manifests/` | no |
| Queue artifact | `data/labels/queue_v1.json` | no |
| Label store | `data/labels/labels.sqlite3` | no |
| Label export | `data/labels/labels_v1.jsonl` (+ `.meta.json`) | no |
| Frozen snapshot | `data/labels/label_snapshot_v1.json` | no |
| Readiness report | `artifacts/experiments/readiness_report.json` | yes (no private data) |
| Dataset statistics | `artifacts/dataset/*.json` | yes |

---

## Current status

The machinery is complete and tested; the corpus is processed; the queue is
built. **There are zero human labels**, so the readiness gate reports Phase 5B
**blocked**, and Phase 5A stops here. The single remaining action is human work:
start the pilot labelling session with the `label serve --queue …` command in
§4.
