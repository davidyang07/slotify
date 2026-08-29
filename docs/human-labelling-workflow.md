# Human-labelling workflow

This document is the operator's guide to the labelling pipeline: acquire a real
target-domain corpus, process it into candidates and features, build a
stratified queue, run resumable labelling sessions, and check whether the
experiment gate is met.

Nothing here trains a model. The held-out comparison begins **only** when
`experiment readiness` reports the gate as met, and it cannot be met until a
person has actually labelled candidates.

> **If you only want to label, skip to [section 3](#3-label).** One command
> prepares everything mechanical and one command opens the session. Sections 1
> and 2 explain what those commands do and how to run the stages individually.

All commands run from the `ml/` directory with its own 3.12 virtual environment:

```powershell
cd ml
.\.venv\Scripts\python.exe -m slotify_rank.cli <command>
```

CPU-only throughout. The only networked steps are `dataset discover` and
`dataset fetch`; the only model download is the one-time Whisper/MiniLM fetch
during `pipeline features`.

---

## 0. The whole thing, in two commands

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset prepare-resume-experiment
.\.venv\Scripts\python.exe -m slotify_rank.cli label resume-experiment
```

The first runs every stage in sections 1 and 2 in order and ends with a
*measured* readiness summary. The second pre-cuts every clip, reports how many
labels remain, and opens the UI. Everything below is what those two commands do.

---

## 1. The corpus

Two committed source registries, both generated or verified against the Internet
Archive metadata API — real identifier, real file, real licence:

| Registry | What it holds |
|---|---|
| `ml/configs/sources_real_v1.yaml` | The first six LibriVox series (ten episodes), hand-written and verified. |
| `ml/configs/sources_resume_v1.yaml` | **Generated** by `dataset discover` from `ml/configs/corpus_resume_v1.yaml`: one real interview podcast, ten multi-voice dramatic readings and ten narrated volumes. |

Only openly licensed audio is used, and every generated entry records *how* its
licence was established — see [`dataset-card.md`](dataset-card.md#sources-and-licensing).
To add a show, add it to the corpus plan and re-run:

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset discover
```

`--check` re-resolves the plan and fails instead of writing, which is what CI
runs weekly to notice an item being withdrawn or made access-restricted.

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

# Freeze the series-aware split (v3 is the current corpus; v1 was the smoke
# split and v2 the first six real series -- a manifest is immutable per version).
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset split --config configs\splits_v3.yaml
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset validate --split-version v3
.\.venv\Scripts\python.exe -m slotify_rank.cli dataset stats    --split-version v3
```

The measured size of the processed corpus is written to
`artifacts/dataset/dataset_statistics.json` and
`artifacts/dataset/candidate_statistics.json` by `dataset stats`, and read from
there by every report. Whatever those files say, the candidates they count are
**generated and unlabelled** — not human labels, and never added to a human
count.

## 3. Build the stratified labelling queue

The queue is deterministic and stratified so labels are not concentrated on
whatever the heuristic already likes. Hard strata are (dataset split ×
heuristic-score tertile); episode and series spread is enforced by round-robin
and a per-episode cap; every other dimension is reported as coverage.

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli label queue `
    --config configs\labelling_queue_resume_v1.yaml --split-version v3 `
    --output data\labels\queue_resume_v1.json
```

The artifact records the score-tertile boundaries, the candidate- and
split-manifest hashes it was built from, and a coverage breakdown across split,
score stratum, series, episode, candidate source, position, silence duration,
sentence-boundary status and transcript availability. It carries four stages:

* **pilot** (30): a diverse first pass to validate the rubric and the context
  window before the bulk run;
* **primary**: the rest of the unique set;
* **overlap** (60): a subset flagged for a *second* annotator, for
  inter-annotator agreement later. Inert with one annotator; it costs nothing to
  reserve;
* **consistency** (60): re-presented under a distinct `presentation_id` to
  measure intra-annotator consistency. Never counted as unique candidates, never
  exported as labels, and indistinguishable from a first showing in the payload
  — not a field, not even the clip URL.

The queue is immutable once labels are collected against it: to change it, bump
`version` in the config (a byte-different queue of the same version is refused).

> **Note.** `signal_disagreement` coverage is 0 in round 1 because candidates
> were generated before transcripts existed, so their `sentence_end` field is
> null. Regenerating candidates with transcript-aware sentence boundaries would
> populate it; transcript *availability* is already 100% on the selected set.

## 4. Label

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli label resume-experiment
# -> pre-cuts every clip, prints what is left, serves http://127.0.0.1:8000/
```

Enter an annotator pseudonym and start. The session is built for a few thousand
items:

| Key | What it does |
|---|---|
| `1`–`5` | Rate **and advance**. One keystroke per item; there is no confirm step. |
| `Space` | Play / pause |
| `R` / `B` | Replay the window / replay around the break |
| `U` | Mark the clip broken — a judgement, and recorded as one |
| `S` | Skip for now — **not** a judgement; it never reaches the export, the gate or a model |
| `N` / `Esc` | Write a note / leave the note field |

The next several items and their audio are already in the browser, so advancing
is a repaint rather than a round trip. Progress reads `1234 / 2400` against the
round's target. Sessions resume: nothing is buffered client-side, so killing the
process loses at most the item on screen.

The annotator never sees the heuristic score, the candidate's source flags, the
split, or whether an item is a repeat. Transcript context either side of the
break is resolved from the cached episode transcript with the same selection the
feature pipeline uses, so the annotator reads what the model reads.

Score each candidate 1–5 on the naturalness rubric
([`labelling-guide.md`](labelling-guide.md)). Run the pilot stage first with
`--stage pilot`, confirm the rubric and the context window feel right, then run
the rest. The full pilot walkthrough is in [`pilot-labelling.md`](pilot-labelling.md).

To run only part of the queue, or to serve without the pre-cutting step,
`label serve --queue ... --stage ...` is still there and behaves identically.

Check quality and export as you go (both are safe to re-run):

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli label check  `
    --queue data\labels\queue_resume_v1.json --split-version v3
.\.venv\Scripts\python.exe -m slotify_rank.cli label export --dataset-version resume-v1
```

`label check` reports invalid ratings, orphaned labels, acceptability
inconsistencies, candidates labelled outside the queue, split drift, manifest
drift, and **measured intra-annotator agreement** once the session has reached
its first blind repeats — as warnings, never by altering a human judgement.

## 5. Check the gate

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment readiness `
    --queue data\labels\queue_resume_v1.json --split-version v3
```

The comparison may begin only when this reports ready. The gate
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
    --snapshot-version resume-v1 --split-version v3 `
    --queue data\labels\queue_resume_v1.json

# Then pin the whole experiment -- split hash, label hash, model and baseline
# configs -- before the test split is read.
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment manifest --require-ready
```

See [`resume-experiment.md`](resume-experiment.md) for what happens next.

## 6. Where things are

| Artifact | Location | Committed? |
|---|---|---|
| Corpus plan | `ml/configs/corpus_resume_v1.yaml` | yes |
| Source registries | `ml/configs/sources_real_v1.yaml`, `sources_resume_v1.yaml` (generated) | yes |
| Queue config | `ml/configs/labelling_queue_resume_v1.yaml` | yes |
| Split config | `ml/configs/splits_v3.yaml` | yes |
| Experiment definition | `ml/configs/experiment_resume_v1.yaml` | yes |
| Fetched audio, normalized WAV | `data/raw/`, `data/normalized/` | no (git-ignored) |
| Candidate / feature manifests | `data/manifests/` | no |
| Queue artifact | `data/labels/queue_resume_v1.json` | no |
| Label store | `data/labels/labels.sqlite3` | no |
| Label export | `data/labels/labels_resume-v1.jsonl` (+ `.meta.json`) | no |
| Frozen snapshot | `data/labels/label_snapshot_resume-v1.json` | no |
| Experiment manifest | `artifacts/experiments/experiment-resume-v1.json` | yes |
| Readiness report | `artifacts/experiments/readiness_report.json` | yes (no private data) |
| Dataset statistics | `artifacts/dataset/*.json` | yes |

---

## Current status

The machinery is complete and tested, the corpus is processed and the queue is
built. The one thing no command can do is form judgements, so the current human
label count is whatever the label store holds — read it from
`artifacts/dataset/label_statistics.json`, or from the checklist in
`artifacts/reports/resume_evidence.md`, rather than from this sentence.

Until that count clears the gate, the readiness report says so and the held-out
comparison refuses to publish. The single remaining action is `label
resume-experiment` (§4).
