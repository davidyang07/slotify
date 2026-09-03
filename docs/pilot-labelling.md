# Pilot labelling session

This is the operator's guide to the **pilot**: a controlled first pass of ~30
real candidates, labelled through the local UI and reviewed before any bulk
labelling begins. Its purpose is to validate the rubric, the audio and
transcript context and the candidate generator on genuine data — not to train
anything.

Do the pilot first. The bulk round is a few thousand items; discovering after
two thousand of them that the context window is too short, or that the rubric's
middle grades are ambiguous, means discarding real human effort. Thirty items
costs ten minutes.

See [`human-labelling-workflow.md`](human-labelling-workflow.md) for how the
corpus and queue were built, [`evaluation-evidence.md`](evaluation-evidence.md)
for what the labels are for, and [`labelling-guide.md`](labelling-guide.md) for
the full rubric.

All commands run from the `ml/` directory with its own Python 3.12 virtual
environment. CPU-only; nothing here reaches the network or a paid API.

---

## 1. Launch the pilot

```powershell
cd ml
.\.venv\Scripts\Activate.ps1        # or call .\.venv\Scripts\python.exe directly

# Pre-cuts every clip, reports what is left, then serves the UI.
.\.venv\Scripts\python.exe -m slotify_rank.cli label run-experiment `
    --stage pilot --host 127.0.0.1 --port 8000
```

| Setting | Value |
|---|---|
| Working directory | `ml/` |
| Python environment | `ml/.venv` (activate `.\.venv\Scripts\Activate.ps1`) |
| Command | `slotify_rank.cli label run-experiment` |
| Queue argument | defaults to `data/labels/queue_full-v2.json` under the data root |
| Stage argument | `--stage pilot` (serves **only** the 30 pilot candidates) |
| Host / port | `127.0.0.1` / `8000` (localhost only) |
| Local URL | <http://127.0.0.1:8000/> |
| Annotator id | `david-pilot-v1` (a pseudonym — enter it in the page) |
| Labels saved to | `data/labels/labels.sqlite3` (git-ignored) |
| Stop the service | `Ctrl+C` in the terminal |

`--stage pilot` restricts the session to the queue's pilot stage. The service
serves exactly those 30 candidates, `Progress` counts against 30, and a label
for any candidate **outside** the pilot is rejected with `404`. Drop `--stage`
(or pass `--stage all`) later to continue into the full 2,400-candidate queue;
`--stage primary` serves the 2,370 non-pilot candidates. The counts come from
the queue itself — `artifacts/labelling/queue_summary.json` records them — so a
requeue changes them here too.

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
text either side of the break is shown when a transcript exists — every queued
candidate has one, because the queue requires a complete feature record — and a
clean "rate from the audio alone" message when it does not.

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
.\.venv\Scripts\python.exe -m slotify_rank.cli label export --dataset-version full-v2

# Quality check: invalid ratings, orphans, out-of-queue labels, manifest drift.
.\.venv\Scripts\python.exe -m slotify_rank.cli label check `
    --queue ..\data\labels\queue_full-v2.json --split-version v4

# Readiness gate (reports zero labels as blocked, honestly).
.\.venv\Scripts\python.exe -m slotify_rank.cli experiment readiness `
    --queue ..\data\labels\queue_full-v2.json --split-version v4
```

Once the pilot has at least 20 genuine usable labels, run the post-pilot
review below before continuing into the bulk round.

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
    --snapshot-version full-v2 --split-version v4 `
    --queue ..\data\labels\queue_full-v2.json
```

---

## 6. The rest of the queue

The bulk stages live in the same artifact and are generated at the same time;
the pilot is not a separate build. Serve them with `--stage primary`, or drop
`--stage` entirely to work the whole queue.

The queue preserves the frozen series-grouped split and is stratified by
(split × heuristic-score tertile) with round-robin episode and series spread.
Its four stages, with the sizes the committed config declares:

| Stage | Size | What it is for |
|---|---|---|
| `pilot` | 30 | the controlled first pass above |
| `primary` | the remainder (2,370 under `full-v2`) | the bulk round |
| `overlap` | 60 | reserved for a *second* annotator, for inter-annotator agreement |
| `consistency` | 60 | blind repeats of candidates already in the set, interleaved far from their first showing, for intra-annotator agreement |

Overlap and consistency presentations are **never** counted as unique
candidates and never enter the label export. Regenerating the queue requires a
version bump — a byte-different queue of the same version is refused — so labels
already collected are never orphaned.

---

## 7. Why evaluation stays blocked

`experiment readiness` reports **not ready** until the label database
genuinely contains enough human judgements. The gate requires at least 200
unique labelled candidates, 8 episodes and 6 series (4 train / 1 validation /
1 test), usable within-episode pairs in train and validation, at least one
graded-relevant test candidate, all labels passing integrity checks, and a
complete multimodal feature record for every labelled candidate. The thresholds
are fixed in `ReadinessGate` and are **not** to be weakened to advance past
them.

the benchmark experiment sets a second, higher bar on top of that one: 2,400 unique
human labels, declared in `ml/configs/experiment_v2.yaml`. Until both are
met, no model is trained on human labels and no NDCG comparison against
`heuristic_offline_v1` is computed or implied. A pilot of thirty labels is a
workflow validation, not a dataset.

---

## Pilot queue at a glance

The queue artifact records exactly what it selected: the score-tertile
boundaries, the candidate- and split-manifest hashes it was built from, and a
coverage breakdown across split, score stratum, series, episode, candidate
source, position bucket, silence bucket, sentence-boundary status and transcript
availability.

Read the current numbers from the artifact rather than from here — they change
whenever the corpus does:

```powershell
.\.venv\Scripts\python.exe -c "import json,sys; q=json.load(open(r'..\data\labels\queue_full-v2.json')); c=q['coverage']; print(q['queue_version'], q['unique_candidate_count']); print(c['by_split']); print(c['by_score_stratum']); print(c['by_primary_source'])"
```

Whatever they say, these are **generated, unlabelled** candidates — not labels,
and never counted as any. A freeze pins exactly this queue by its content hash.
