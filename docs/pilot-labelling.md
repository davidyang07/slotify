# Pilot labelling session (Phase 5A.1)

This is the operator's guide to the **pilot**: a controlled first pass of
~20–30 real candidates, labelled through the local UI, reviewed before any bulk
labelling begins. Its purpose is to validate the rubric, the audio/transcript
context and the candidate generator on genuine data — not to train anything.

The repository ships with **zero human labels**. Phase 5B (the real
baseline-vs-model comparison) stays blocked until ≥ 200 genuine labels exist, so
the pilot is where the human work starts. See
[`human-labelling-workflow.md`](human-labelling-workflow.md) for how the corpus
and queue were built, and [`labelling-guide.md`](labelling-guide.md) for the
full rubric.

All commands run from the `ml/` directory with its own Python 3.12 virtual
environment. CPU-only; nothing here reaches the network or a paid API.

---

## 1. Launch the pilot

```powershell
cd ml
.\.venv\Scripts\Activate.ps1        # or call .\.venv\Scripts\python.exe directly

.\.venv\Scripts\python.exe -m slotify_rank.cli label serve `
    --queue ..\data\labels\queue_v1.json `
    --stage pilot `
    --host 127.0.0.1 --port 8000
```

| Setting | Value |
|---|---|
| Working directory | `ml/` |
| Python environment | `ml/.venv` (activate `.\.venv\Scripts\Activate.ps1`) |
| Command | `slotify_rank.cli label serve` |
| Queue argument | `--queue ..\data\labels\queue_v1.json` |
| Stage argument | `--stage pilot` (serves **only** the 24 pilot candidates) |
| Host / port | `127.0.0.1` / `8000` (localhost only) |
| Local URL | <http://127.0.0.1:8000/> |
| Annotator id | `david-pilot-v1` (a pseudonym — enter it in the page) |
| Labels saved to | `data/labels/labels.sqlite3` (git-ignored) |
| Stop the service | `Ctrl+C` in the terminal |

`--stage pilot` restricts the session to the queue's pilot stage. The service
serves exactly those 24 candidates, `Progress` counts against 24, and a label
for any candidate **outside** the pilot is rejected with `404`. Drop `--stage`
(or pass `--stage all`) later to continue into the full 280-candidate queue;
`--stage primary` serves the 256 non-pilot candidates.

Open the URL, type the annotator id `david-pilot-v1`, and press **Start /
resume**. No name or personal information is stored — the id is the only
identifier on a label.

---

## 2. The rubric

Judge **one question**: if an audio ad were inserted at this exact timestamp,
how natural would the transition sound? Not whether the content deserves an ad,
not whether the audio is interesting — only the seam.

```text
1 — Clearly disruptive or inside speech.
2 — Technically possible, but noticeably unnatural.
3 — Acceptable breakpoint.
4 — Strong natural breakpoint.
5 — Highly natural transition.
```

Acceptability is derived, not entered: `is_acceptable = quality_score >= 3`.

Edge cases (the full list is in [`labelling-guide.md`](labelling-guide.md)):

- A long silence is **not** automatically strong if it interrupts a thought —
  judge what the pause *means*, not how long it is.
- A completed sentence is **not** automatically strong without enough
  perceptual separation from what follows.
- Topic changes and speaker changes **strengthen** a candidate.
- Intro, outro, continuous music, corrupted audio and insufficient context are
  marked **unusable** (tick the box; still give a score of 1 — `is_unusable`
  is a separate flag).
- Rate insertion **naturalness**, not how interesting the content is.
- Do **not** consider what the heuristic or a model might think — the UI hides
  those signals for exactly this reason.
- Apply the same standard to every candidate.

The rubric and edge cases are always reachable from the UI via the
[`labelling-guide.md`](labelling-guide.md) link on the page.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `1`–`5` | Rate and advance |
| `R` | Replay the full context window |
| `B` | Replay the last few seconds before the break |
| `U` | Toggle "unusable" |
| `Space` | Play / pause |

Each clip is ~10 s before and ~10 s after the proposed break, played as one
continuous window; the UI labels where in the clip the break falls. Transcript
text either side of the break is shown when a transcript exists (all 24 pilot
candidates have one), and a clean "rate from the audio alone" message when it
does not.

---

## 3. Save and resume

- **Every rating saves immediately.** Nothing is buffered in the browser;
  closing the tab loses at most the in-flight item.
- **Sessions resume.** Your candidate order is a stable shuffle keyed on your
  annotator id, so re-running the launch command and entering `david-pilot-v1`
  returns you to the next unlabelled candidate.
- **Re-rating updates in place.** One active label per (annotator, candidate);
  a second rating overwrites the first rather than appending.

Resume is just the same launch command again — there is no separate resume flag.

---

## 4. Export, validate, and post-pilot review

Safe to re-run at any time.

```powershell
# Export human labels to versioned JSONL (+ .meta.json sidecar).
.\.venv\Scripts\python.exe -m slotify_rank.cli label export --dataset-version v1

# Quality check: invalid ratings, orphans, out-of-queue labels, manifest drift.
.\.venv\Scripts\python.exe -m slotify_rank.cli label check `
    --queue ..\data\labels\queue_v1.json --split-version v2

# Phase 5B readiness gate (reports zero labels as blocked, honestly).
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment readiness `
    --queue ..\data\labels\queue_v1.json --split-version v2
```

Once the pilot has ≥ 20 genuine usable labels, rerun the **Phase 5A.1
post-pilot review**: `label check`, pilot statistics and the pilot acceptance
check, then freeze the snapshot and generate the primary queue.

---

## 5. Pilot acceptance criteria

The pilot is accepted when all of these hold:

```text
valid usable labels        >= 20
episodes represented       >= 3
series / sources           >= 3
quality-score levels        >= 2
both acceptable and unacceptable candidates exist
usable within-episode ranking pairs >= 10
unusable rate < 30% (unless clearly explained)
no critical playback or persistence defect
```

If any criterion fails, do **not** generate the full primary queue. Identify the
blocking reason, make only justified workflow or candidate-generation
corrections, version any changed configuration, and generate a pilot *supplement*
rather than discarding valid labels.

If accepted, freeze an immutable snapshot before anything else:

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment freeze `
    --snapshot-version v1 --split-version v2 --queue ..\data\labels\queue_v1.json
```

---

## 6. The primary queue

The primary 200–250-candidate queue is generated **only after the pilot is
accepted**, and it already exists inside `queue_v1.json` as the `primary` stage
(256 candidates, disjoint from the 24 pilot candidates). Serve it with
`--stage primary`. It preserves the frozen series-aware v2 split and is
stratified by split × heuristic-score tertile with round-robin episode/series
spread; overlap (40) and consistency (12) presentations are reserved for
inter- and intra-annotator agreement and are never counted as unique training
examples. Regenerating the queue requires a version bump — a byte-different
queue of the same version is refused, so labels already collected are never
orphaned.

---

## 7. Why evaluation stays blocked

`experiment readiness` will report **Phase 5B ready: False** until the label
database genuinely contains enough human judgements. The gate requires ≥ 200
unique labelled candidates, ≥ 8 episodes, ≥ 6 series (≥ 4 train / ≥ 1 validation
/ ≥ 1 test), usable within-episode pairs in train and validation, at least one
graded-relevant test candidate, all labels passing integrity checks, and a
complete multimodal feature record for every labelled candidate. The thresholds
are fixed in `ReadinessGate` and are **not** to be weakened to advance the
phase. Until they are met, no model is trained and no NDCG comparison against
`heuristic_offline_v1` is computed or implied. A pilot of ~24 labels is a
workflow validation, not a dataset.

---

## Pilot queue at a glance

Read from `data/labels/queue_v1.json` (git-ignored; regenerated by
`label queue`). These are **generated, unlabelled** candidates — not labels.

| Field | Value |
|---|---|
| Queue version | `v1` (schema `labelling-queue-v1.0.0`) |
| Pilot candidates | 24 unique |
| Episodes / series | 6 / 6 |
| Split spread | train 9 · validation 9 · test 6 |
| Score strata | high 9 · medium 9 · low 6 |
| Candidate sources | silence 18 · rms_minimum 4 · pause 2 |
| Episode position | spread across p0–p4 (early/middle/late) |
| Transcript / feature completeness | 24/24 · 24/24 |
| Synthetic / missing-media | 0 · 0 |

The queue's content and manifest hashes are recorded inside the artifact; a
freeze pins exactly this queue.
