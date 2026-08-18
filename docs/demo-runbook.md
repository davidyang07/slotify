# Demo runbook

A 2–3 minute walkthrough for a technical interview, and what to do when
something breaks. Read the recovery section before you need it.

---

## Before the call

```bash
npm run preflight
```

Every REQUIRED line must read `[OK]`. `[WARN]` lines are fine — they are the
optional paid APIs, and the demo is designed to work without them.

Pick your clip. **60–120 seconds of speech.** Longer is worse: Whisper
transcription is the dominant cost and runs at roughly real time on CPU, so a
10-minute episode is a 90-second wait in front of an interviewer.
`backend/audio_tests/rogan-test1.mp3` (19 s) is the safe default and is already
in the repository.

Warm the caches once, so the first run of the day is not the demo:

```bash
cd ml
python -m slotify_rank.cli infer rank \
  --audio ../backend/audio_tests/rogan-test1.mp3 \
  --checkpoint ../artifacts/training/gated-d8ed976101aa4c3b/best_checkpoint.pt \
  --top 3
```

Then start it:

```bash
npm run demo
```

Have a second terminal open at the repository root. You will want it for the
evidence report if the conversation goes that way.

---

## The walkthrough

**00:00 — Frame it.**
"Slotify places ads inside podcasts. The interesting part isn't the audio
stitching, it's deciding *where* the ad goes — that's a ranking problem inside
one episode, and the signal is multimodal: a two-second pause is a good break if
the topic also changed, and a bad one if the host just drew breath mid-sentence."

**00:20 — Upload.**
Drag the clip onto the upload card. The waveform renders client-side. Click
**Analyze & recommend slots**.

Say while it runs: "This needs no API keys. Everything from here to the ranking
is local."

**00:30 — Candidate generation.**
"First a deterministic pass over the audio: silence runs, pause detection, RMS
valleys, and transcript sentence ends. That produces candidates — it does not
rank them. Generation and ranking are deliberately separate."

**00:45 — Multimodal features.**
"Each candidate gets three blocks: 110 handcrafted acoustic and structural
scalars, a Whisper encoder representation of the audio either side, and a MiniLM
embedding of the transcript either side plus their difference and product. Both
encoders are frozen — I'm learning the ranking, not the representations."

**01:00 — The ranking.**
The analyze page shows the ranked slots. Point at:

- the **provenance chip** — "Learned ranker (gated)", so you can always tell
  which system produced the ordering;
- the **placement score** — "That's a ranking score out of 100, not a
  probability. The product used to show '92% confidence' from
  `clamp(70 + score * 25, 70, 95)`, which reads like a calibrated number and was
  calibrated against nothing. I removed it.";
- the **signals** under each slot — every one is measured. If no transcript was
  available, it says so rather than filling the space.

Note the slot count. **If it shows two slots, say so:** "It found two defensible
points, so it shows two. It used to pad up to three with slots at 22, 48 and 72
percent of the duration carrying hard-coded scores. That's the first thing I
fixed."

**01:20 — The honest bit.** This is the strongest moment; do not skip it.

"The model shipping here is trained on weak labels derived from the heuristic
baseline — a distillation. It proves the architecture and the serving path. It
is not evidence of ranking quality, and the system won't let me pretend
otherwise: the API attaches that warning to every response, and the evaluation
command refuses to publish an improvement against the baseline, because the
baseline is its teacher. I have zero human labels so far, and the readiness gate
is blocked, correctly."

**01:40 — Selection and generation.**
Select a slot. If `ELEVENLABS_API_KEY` is set, click **Preview** and let the
sponsor read play. If it is not set, the buttons are disabled with the reason —
which is itself worth pointing at: "Generation is gated on the key. Placement
isn't. That was a refactor: cloning used to run *before* analysis, so the whole
ML demo depended on a paid API."

**02:00 — Export.**
Render, play the merged audio, download.

**02:20 — If they ask about evidence.**

```bash
npm run evidence
```

Show `artifacts/reports/resume_evidence.md`. Every number in it is read from a
generated artifact; `NOT YET SUPPORTED` means nothing produced that number, and
a measured zero prints as `0`. The two are deliberately different strings.

**If they ask about architecture**, the README's Mermaid diagram is the fastest
answer, and `docs/model-inference.md` traces a single request end to end.

---

## Recovery

| Symptom | Cause | Do this |
| --- | --- | --- |
| Analyze spins for a long time | Whisper is transcribing; ~1× real time on CPU | Say so, and use a shorter clip next time |
| `placementStatus: "unavailable"` | ffmpeg could not decode the upload | Use a `.mp3` or `.wav`; check `ffmpeg` on PATH |
| "No insertion points to show" | Analysis ran and found nothing eligible | This is the honest empty state. Show it — it is a feature. Use a longer clip |
| Learned ranker times out | Long audio | `SLOTIFY_RANKER_SKIP_TRANSCRIPTION=1` (text modality masked, and the response says so), or restart with `--heuristic` |
| ElevenLabs error | Key missing, out of quota, or the service is down | Skip generation. Placement is the demo; ad rendering is the encore |
| Server will not start | Port in use | `npm run preflight` names the port; change `PORT` |
| Everything is on fire | — | See the offline path below |

### The offline path

If the network is gone, the paid APIs are down, or you have thirty seconds:

```bash
npm run demo -- --heuristic
```

The baseline needs no network, no key and no model, and returns in about seven
seconds. The provenance chip will read "Heuristic baseline" — which is the point:
the product tells you which system spoke, so the fallback is not a lie.

If even that fails, the CLI is a complete demo on its own:

```bash
cd ml
python -m slotify_rank.cli infer rank \
  --audio ../backend/audio_tests/rogan-test1.mp3 \
  --checkpoint ../artifacts/training/gated-d8ed976101aa4c3b/best_checkpoint.pt \
  --top 3
```

That prints the whole provenance-stamped ranking document, including the model
identity, the score scale, `is_calibrated_probability: false`, and per-stage
timings.

---

## Questions worth preparing for

**"Is the model actually better than the heuristic?"**
Unknown, and the repository says unknown. That needs human labels on held-out
episodes; there are zero so far. The infrastructure to answer it — labelling UI,
readiness gate, frozen label snapshot, held-out comparison — is built and tested,
and `evaluation compare` blocks the headline until the conditions hold.

**"Why weak labels at all, then?"**
To prove the inference path on real audio with real weights before the labelling
round. It is a bootstrap, stamped as one in the checkpoint, the API response,
the preflight output and the evidence report.

**"Why is the placement score relative?"**
The learned model's output is an unbounded score whose ordering is meaningful
only within an episode. Presenting it on an absolute scale would imply a
cross-episode comparison the model was never trained to make. The scale is named
in the response (`episode_relative_min_max`) so a client cannot misread it.

**"Where's the FastAPI?"**
Hosting the human-labelling UI (`slotify-rank label serve`), not the product API.
The product API is Express.

**"Why a subprocess instead of a model service?"**
The model loads in ~0.15 s against a feature pass that takes tens of seconds, so
a persistent service would optimise 0.4% of the latency and add a process that
can be down during a demo. `RankerPredictor` is written to be constructed once
and reused, so a service is a new entry point rather than a rewrite.
