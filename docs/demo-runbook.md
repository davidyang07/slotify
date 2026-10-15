# Demo runbook

A 2–3 minute technical walkthrough of the running system, and what to do when
something breaks. Read the recovery section before you need it.

---

## Before the call

```bash
npm run preflight
```

Every REQUIRED line must read `[OK]`; any REQUIRED failure exits non-zero and
`npm run demo` will refuse to start. `[WARN]` lines are fine — they are the
optional paid APIs and the learned-ranker extras, and the demo is designed to
work without them. The ones that catch people:

- **Python (ad_inserter)** imports pydub, numpy, pyloudnorm, librosa and
  soundfile with the interpreter the API will use (`PYTHON_BIN`, default
  `python`). If it fails, every analysis would fail too:
  `<that python> -m pip install -r backend/requirements.txt`, or point
  `PYTHON_BIN` at an interpreter that has them.
- **port 3001 (backend)** and **port 5173 (frontend)** must be free. A busy port
  is a failure that names the process holding it — usually a previous
  `npm run demo` or `npm run dev`. Stop it with the command shown. The API port
  can move instead (`PORT=3002 npm run demo`); the UI port cannot, because Vite
  runs with `--strictPort`.
- **Node.js** must be 22 or newer.

Preflight reads `backend/.env` the way the API does, so `PYTHON_BIN`, `PORT` and
the API keys set there count.

Pick your clip. **60–120 seconds of speech.** Longer is worse: Whisper
transcription is the dominant cost and runs at roughly real time on CPU, so a
10-minute episode is a 90-second wait in front of an audience.
`backend/audio_tests/ryan-test1.mp3` (23 s) is the safe default and is already
in the repository.

Warm the caches once, so the first run of the day is not the demo:

```bash
cd ml
python -m slotify_rank.cli infer rank \
  --audio ../backend/audio_tests/ryan-test1.mp3 \
  --checkpoint ../artifacts/training/gated-d8ed976101aa4c3b/best_checkpoint.pt \
  --top 3
```

Then start it:

```bash
npm run demo
```

This starts the API on `PORT` (3001) and Vite on 5173 with `--strictPort`, and
prints `Open http://localhost:5173/` once Vite reports it is serving — open that
URL, not one from memory. The browser talks only to Vite, which proxies `/api`
to the API (`VITE_PROXY_TARGET`, which the demo points at `PORT`), so there is no
CORS setup to get wrong. Ctrl-C stops both; if either one dies, the demo stops
the other and exits non-zero.

Have a second terminal open at the repository root.

### Optional: tokens and rate limits

The token matters only if the API will be reachable from other machines (a
tunnel, a shared network); on localhost, leave it unset. The rate limits are on
whether or not you set them, at the defaults below — generous for one presenter.

| Variable | Where | Effect |
| --- | --- | --- |
| `SLOTIFY_API_TOKEN` | `backend/.env` | Every route except health and capabilities (GET or HEAD) requires `Authorization: Bearer <token>` |
| `VITE_API_TOKEN` | `frontend/.env.local` | The token the UI sends. Must match `SLOTIFY_API_TOKEN`, or every call is refused. It is compiled into the page, so it keeps out passers-by, not anyone who can load the UI |
| `SLOTIFY_RATE_LIMIT_COMPUTE_PER_MIN` | `backend/.env` | Requests per client IP per minute to the compute-heavy routes, analysis included (default 30; `0` disables) |
| `SLOTIFY_RATE_LIMIT_PAID_PER_MIN` | `backend/.env` | Requests per client IP per minute to the routes that spend OpenAI or ElevenLabs credit (default 10; `0` disables) |
| `SLOTIFY_MAX_CONCURRENT_JOBS` | `backend/.env` | Analyses, pipelines and merges running at once (default half the CPUs, at least 1). More wait in a queue of `SLOTIFY_MAX_QUEUED_JOBS` (default 8); past that the API answers 503 with `Retry-After` |
| `SLOTIFY_MAX_AUDIO_MINUTES` | `backend/.env` | Longest upload accepted (default 180); longer is a 413 |
| `SLOTIFY_MAX_SONG_MINUTES` | `backend/.env` | Longest upload song mode analyses (default 10); its beat tracking needs ~180 MB of memory per minute |
| `SLOTIFY_VOICE_TTL_MIN` | `backend/.env` | A cloned voice idle this long is deleted from the ElevenLabs account (default 30). The UI also releases its voice when the file is replaced or the page is closed |
| `SLOTIFY_MAX_LIVE_VOICES` | `backend/.env` | Cloned voices kept at once, least recently used deleted first (default 3) |
| `SLOTIFY_TTS_MAX_STATEMENTS`, `SLOTIFY_TTS_MAX_CHARS` | `backend/.env` | Per-request text-to-speech caps, checked before ElevenLabs is called (defaults 8 and 1,500) |
| `SLOTIFY_MAX_CLONE_SAMPLES`, `SLOTIFY_MAX_CLONE_SAMPLE_MB` | `backend/.env` | Voice-clone upload caps (defaults 3 files, each up to `SLOTIFY_MAX_UPLOAD_MB`) |
| `SLOTIFY_CLONE_SAMPLE_SECONDS` | `backend/.env` | How much of each clone sample is sent to ElevenLabs (default 180 s, skipping the first 30 s of a long file) |
| `SLOTIFY_OPENAI_MODELS` | `backend/.env` | OpenAI models `/ad/insert` may be asked for (default `gpt-4o-mini`) |
| `SLOTIFY_KILL_GRACE_MS` | `backend/.env` | Grace between SIGTERM and SIGKILL for a stopped child process (default 5000) |

Restart `npm run demo` after changing any of them; Vite reads `VITE_*` only at
startup.

---

## The walkthrough

**00:00 — Frame it.**
"Slotify places ads inside podcasts. The interesting part isn't the audio
stitching, it's deciding *where* the ad goes — that's a ranking problem inside
one episode, and the signal is multimodal: a two-second pause is a good break if
the topic also changed, and a bad one if the host just drew breath mid-sentence."

**00:20 — Upload.**
Drag the clip onto the dropzone. Analysis starts on drop; the waveform renders
client-side while the break list shows progress. Type the sponsor name into the
**Sponsor read** panel while it runs.

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
The break list fills in, ranked. Point at:

- the **provenance chip** — "Learned ranker (gated)", so you can always tell
  which system produced the ordering;
- the **placement score** — "That's a ranking score out of 100, not a
  probability. The product used to show '92% confidence' from
  `clamp(70 + score * 25, 70, 95)`, which reads like a calibrated number and was
  calibrated against nothing. I removed it.";
- the **reasons** on each row (expand a row for all of them) — every one is
  measured. If no transcript was available, it says so rather than filling the
  space.

Note the break count. **If it shows two, say so:** "It found two defensible
points, so it shows two. It used to pad up to three with slots at 22, 48 and 72
percent of the duration carrying hard-coded scores. That's the first thing I
fixed."

**01:40 — Selection and generation.**
Tick a break (or click its marker on the waveform). If `ELEVENLABS_API_KEY` is
set, press the row's ▶ to preview the sponsor read in place. If it is not set,
the controls are disabled with the reason —
which is itself worth pointing at: "Generation is gated on the key. Placement
isn't. That was a refactor: cloning used to run *before* analysis, so the whole
ML demo depended on a paid API."

**02:00 — Render.**
Press **Render N ad breaks** in the panel, play the merged audio, download.

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
| Server will not start | Port in use | `npm run preflight` names the port and the process holding it; stop that process, or change `PORT` (API only) |
| `[ui] Error: Port 5173 is already in use` | Another Vite (often an earlier demo) holds 5173 | Stop it; Vite will not move to 5174 on its own, by design |
| Every analysis fails with `ModuleNotFoundError` | `PYTHON_BIN` lacks backend/requirements.txt | `npm run preflight` says which module; install the requirements into that interpreter |
| UI says the API is not reachable | The API is down or on another port | Check the `[api]` lines; if you changed `PORT` outside the demo, set `VITE_PROXY_TARGET` to match |
| 401 on every request | `SLOTIFY_API_TOKEN` is set but `VITE_API_TOKEN` is missing or different | Make them match, or unset both, and restart |
| 429 Too Many Requests | A per-minute rate limit, usually from rapid repeat clicks | Wait a minute, or raise `SLOTIFY_RATE_LIMIT_COMPUTE_PER_MIN` (analysis) or `SLOTIFY_RATE_LIMIT_PAID_PER_MIN` (generation) and restart |
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
  --audio ../backend/audio_tests/ryan-test1.mp3 \
  --checkpoint ../artifacts/training/gated-d8ed976101aa4c3b/best_checkpoint.pt \
  --top 3
```

That prints the whole provenance-stamped ranking document, including the model
identity, the score scale, `is_calibrated_probability: false`, and per-stage
timings.

---

## Questions worth preparing for

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
