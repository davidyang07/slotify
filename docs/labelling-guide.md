# Labelling guide

You are judging **one question**: if an audio advertisement were inserted at this exact timestamp,
how natural would the transition sound to a listener?

You are *not* judging whether the content deserves an ad, whether the ad would sell anything, or
whether the surrounding audio is interesting. Only the seam.

Rubric version: `rubric-v1.0.0`. It is stored on every label, so labels collected under a future
rubric are never pooled with these.

---

## The scale

| Score | Meaning |
|---|---|
| **1** | Clearly disruptive, or inside speech. Cuts a word, a phrase, or a thought in half. |
| **2** | Technically possible but unnatural. Lands in a gap, but the gap is not a real boundary — mid-list, mid-argument, between a question and its answer. |
| **3** | Acceptable. A listener would notice the break but would not be annoyed by it. |
| **4** | Strong natural breakpoint. A real end of a thought, a completed exchange, a clean handoff. |
| **5** | Highly natural transition. A topic actually ends here; an ad would feel like it belongs in the seam. |

The scale is **graded**, not binary. NDCG consumes `quality_score` directly, and
the equivalent 0–4 relevance grade is `quality_score - 1` — a 1 contributes no
gain at all, which is why "clearly disruptive" and "possible but unnatural" have
to stay distinguishable rather than collapsing into "bad".

**`is_acceptable` is derived, not entered.** The default rule is:

```text
is_acceptable = quality_score >= 3
```

The threshold is configurable (`--acceptable-threshold`) and the value in force is stored on every
label and recorded in the export metadata, so changing it later cannot silently rewrite history.

---

## How to run a session

```powershell
cd ml
.\.venv\Scripts\python.exe -m slotify_rank.cli label run-experiment
```

That pre-cuts every clip so no rating waits on FFmpeg, prints how many labels
remain, and serves the UI. Add `--stage pilot` for the controlled first pass.
`label serve --queue …` is still there for a session that should not re-cut
clips.

Open <http://127.0.0.1:8000/>, enter an annotator id, and start. Notes:

- **Your annotator id may be a pseudonym.** `annotator-a` is fine. No personal information is
  requested or stored anywhere in the schema.
- **A rating is one keystroke.** `1`–`5` saves *and* advances; there is no confirm step. The next
  several clips are already loaded, so the gap between items is a repaint.
- **Every rating saves immediately.** Nothing is buffered in the browser, so closing the tab loses
  at most the item on screen. If a save fails you are told and the item comes back.
- **Transcript context is shown when available.** The text just before and just after the break is
  the same context the model reads; when the episode has no transcript the UI says so and you rate
  from the audio alone.
- **Sessions resume.** Your candidate order is a stable shuffle keyed on your annotator id, so you
  return to exactly where you stopped. Two annotators get different orders, which is what removes
  position effects from inter-rater agreement.
- **The heuristic's opinion is hidden.** You will not see the baseline score or which generator
  proposed the timestamp. This is deliberate: if you knew the baseline liked a candidate you would
  rate it higher, and the labels would partly be a measurement of the baseline rather than of the
  audio. (`--reveal-hints` exists for debugging; never use it for labels you intend to train on.)

### Keyboard shortcuts

The session needs no mouse.

| Key | Action |
|---|---|
| `1`–`5` | Rate **and advance** |
| `Space` | Play / pause |
| `R` | Replay the full context window |
| `B` | Replay the last few seconds before the break |
| `U` | Toggle "broken clip" — a judgement, and recorded as one |
| `S` | Skip for now — **not** a judgement; the item returns when you clear skips |
| `N` / `Esc` | Write a note / leave the note field |

`U` and `S` are different things and the difference matters. A broken clip is
information about the audio and is exported as a label with `is_unusable` set. A
skip is you deferring, and never reaches the export, the readiness gate or a
model.

---

## What to do about specific situations

### Long silence
A long silence is *evidence* of a boundary, not proof of one. A five-second pause because the host
is thinking is not a topic end — score it **2–3**. A five-second pause because a segment finished
is a **4–5**. Judge what the silence *means*, not how long it is. The pipeline already prefers long
pauses; your job is to tell it when that preference is wrong.

### Incomplete speech
If a word, name, number, or clause is cut, it is a **1**. No exceptions, however long the pause
around it looks.

### Filler words
A break immediately after "um", "uh", "like", "you know" is usually **2**: it is a gap, but the
speaker was mid-thought. If the filler clearly closes a turn ("...yeah, so, anyway.") it can be a
**3–4**.

### Topic transitions
The strongest signal available. A genuine change of subject is **5**. Score it that way even if the
pause is short — a clean topic change with a 400 ms gap beats a two-second breath mid-topic.

### Speaker changes
A handoff between speakers is a natural seam: **4** by default. Drop to **2** if the second speaker
is answering a question the first just asked — splitting a question from its answer is jarring even
though the gap is real.

### Music, intros and outros
A break at the edge of a music bed, sting, or theme is usually **4–5** — that is exactly where real
ads go. A break *inside* continuous music is **1–2**: it chops the music. If the file is entirely
music (the repository's `drake.mp3`, `ryan-rap.mp3`), it is out-of-domain material kept only to
exercise the pipeline; label it if it appears, but it does not count toward target-domain results.

### Repeated or duplicate candidates
You should not see the same moment twice — merging collapses candidates within 400 ms. If two
candidates *do* feel like the same moment, rate both consistently and add a note. Do not try to
compensate by scoring the second one differently.

### Corrupted audio
Silence where there should be speech, digital noise, an abrupt truncation, a clip that does not
play: tick **unusable**. Still give a score (use **1**); `is_unusable` is a separate flag, and
unusable candidates are reported on their own line rather than being deleted.

### Insufficient context
If the clip starts or ends so close to the break that you cannot tell what is happening — most often
near the beginning or end of an episode — tick **unusable** and note "insufficient context". The
5 s edge guard should prevent this; if it happens often, the guard needs raising.

### When you genuinely cannot decide
Score **3** and add a note. Do not press `S`: a systematically skipped *kind* of candidate biases
the dataset more than an uncertain label does. Skipping is for interruptions — the phone rang, you
need to re-listen with better headphones — not for hard cases.

### You are shown some clips twice
Deliberately, and you are not told which. A small blind subset is re-presented much later in the
session under a different id, and the agreement between your two judgements is the only measurement
of how consistent you are. Rate the repeat as you find it. If you recognise one, rate it as you
would have anyway rather than trying to recall your first answer — a remembered answer measures
memory, which is not the thing being measured.

---

## Consistency

- **Calibrate first.** Label 20–30 candidates, then re-listen to your 1s and your 5s and check they
  still feel a scale apart. Adjust *before* going further, not afterwards.
- **Do not adjust to hit a distribution.** If most candidates in an episode are bad, most scores
  should be low. A forced spread is worse than an honest skew.
- **Take breaks.** Fatigue compresses ratings toward the middle. 30–45 minutes at a time.

---

## Staged milestones

Labelling is staged so problems surface before hours are sunk into them.

| Stage | Labels | Purpose |
|---|---|---|
| Pilot | 30 | Prove the loop works and the rubric is usable. Expect to revise the rubric here — that is what it is for. |
| Readiness gate | 200 | The generic minimum for any held-out comparison: enough episodes, series and within-episode pairs to be meaningful. |
| the benchmark experiment | 2 400 | The gate declared in `ml/configs/experiment_v2.yaml`. Roughly 7–10 hours of attention at 10–15 s an item. |

These are **targets, not results.** The measured count lives in
`artifacts/dataset/label_statistics.json` as `human_labelled_candidate_count`, and it is the only
number that may be quoted. The progress counter in the UI reads against the round's target so you
can see the distance; it is not a claim about anything until the round finishes.

---

## Exporting

```powershell
.\.venv\Scripts\python.exe -m slotify_rank.cli label export --dataset-version full-v2
```

Writes `data/labels/labels_full-v2.jsonl` plus a `.meta.json` sidecar recording the rubric
version, the label schema version, the acceptability rule in force, the graded-relevance rule, any
orphaned labels, and how many blind repeats were excluded. Every exported row is a human judgement:
weak and heuristic labels are never written to this file, and neither are the consistency
repeats — pooling those would turn a quality control into extra supervision.

The database (`data/labels/labels.sqlite3`) and the exports are git-ignored. They are your data.
