# The `ad_inserter` pipeline

The hackathon build's audio analysis, mixing and two-speaker insertion, moved
out of the README when that was rewritten around the ranker. The modules are the
same; the behaviour notes below (splice insertion, candidate selection, voice
cleanup, callback auth) describe them as they are now.

Run every command from the `backend/` directory so `python -m ad_inserter.*`
resolves the package.

Where this sits relative to the rest of the system: `ad_inserter` does candidate
*detection* and audio *mixing*. Candidate *ranking* is `ml/src/slotify_rank`
(see [`model-inference.md`](model-inference.md)). The two are separate on
purpose — mixing has nothing to learn, and ranking has nothing to render.

## Ad Inserter backend module
- `__init__.py` exposes the package modules (analysis, llm, mix, tts, insert_ad) and version
- `analysis.py` handles audio analysis: ffmpeg check, loading/standardizing audio, the reduced analysis decode (see [Memory](#memory)), silence-based candidate detection for podcasts, beat/RMS analysis for songs, optional Whisper transcription, and building candidate payloads
- `analyze_cli.py` exposes a CLI helper that runs analysis and returns JSON for the Node API (`POST /api/insert-sections`)
- `runtime.py` is the process contract every CLI shares: `--work-dir`, the private temp dir, and the SIGTERM/SIGINT handlers (see [Running under the Node API](#running-under-the-node-api))
- `cli.py` provides the single-speaker CLI workflow: parse args, pick candidates, call LLM to write promo/choose insertion, loudness match + room tone, then hand the splice to the API's `/api/merge` (plus debug artifacts)
- `insert_ad.py` handles two-speaker insertion (A/B/DUO), optional diarization, and optional voice cloning; it splices the ad in locally with `mix.insert_with_crossfade`
- `llm.py` builds the prompt and calls OpenAI to generate promo text and choose insertion index; parses JSON response into `LLMResult`. A malformed reply (not JSON, not an object, missing `promo_text`/`chosen_index`, a non-integer or out-of-range index) falls back to the product name and the heuristic insertion point instead of crashing. It also enforces the model allow-list and the output caps (see [LLM configuration](#llm-configuration))
- `mix.py` does audio mixing utilities: LUFS measurement, loudness matching, looping room tone, splice insertion with crossfaded joins, and context window extraction
- `tts.py` requests sponsor reads from the API's `/api/tts` (ElevenLabs behind it)
- `http_client.py` is the one place the pipeline makes HTTP requests: it adds the callback credential and timeout (see [Callbacks to the API](#callbacks-to-the-api)) and encodes multipart bodies
- `heuristic_config.py` loads `config/heuristic_offline_v1.json`, the constants shared with the TypeScript scorer and the ML baseline

### How is the ad inserted?
`mix.insert_with_crossfade` is a splice, not an overlay: the output is
`original[:t] + ad + original[t:]`. The original pauses for the ad and resumes
exactly where it stopped, so nothing plays under the ad and the ad is never cut
short (an insertion point past the end appends it). Each join is a crossfade of
`--crossfade-ms` (default 250), limited by the audio on either side and to half
the ad so the two joins never overlap. The output is therefore
`len(original) + len(ad)` minus the two crossfade overlaps. There is no ducking:
the `--duck-db` flag has been removed from both CLIs.

Before splicing, the ad is loudness-matched to the 8 s of original around the
insertion point. Matching is skipped when either measurement is not finite
(digital silence, or a clip shorter than the meter's 400 ms block), and the gain
is clamped to +/-20 dB, so a silent context can no longer mute the ad.

`cli.py` does not splice locally; it posts the original and the processed promo
to `/api/merge` with `insertAt`, `crossfade` (`--merge-crossfade`, seconds,
default 0.08) and `pause` (`--merge-pause`, seconds, default 0.2).

### How is insertion point chosen?
Semantic context:
- Uses Whisper locally (if installed) to transcribe short context windows around candidate insertion points before evaluating topic transitions and sentence boundaries. The model is loaded once per process and reused for every snippet
- If Whisper is not available, fall back to silence-based insertion
- `cli.py` transcribes the chosen point on demand when it was not among the pre-transcribed candidates; without Whisper the debug transcript is `TRANSCRIPT_UNAVAILABLE`

Podcast candidates (`analyze_cli --mode podcast`):
- Detects every silence of at least 700 ms (`heuristic_offline_v1.silence_detection`)
- Keeps the `--max-candidates` **longest** silences (ties go to the earlier one) and reports them in time order, so a long episode's candidates are not all bunched at its start
- Transcribes snippets for the `--snippet-count` longest of those
- Both flags default to `silence_detection.max_candidates` and `silence_detection.snippet_count` in `config/heuristic_offline_v1.json`; the Node API passes the same values explicitly

Rhythmic/syntactic context (song mode):
- Uses librosa to estimate tempo and beat times
- Finds low-energy (RMS) valleys, quietest first, at least 6 s apart and at least 5 s from either end of the track (both waived only if the track is too short to leave any frame), so candidates spread through the song instead of collapsing into a quiet intro or outro
- Snaps each valley to the nearest beat and returns the candidates in time order

### Test without deploying full app

Run from the `backend/` directory so `python -m ad_inserter.cli` can find the package.

Podcast example:
```bash
python -m ad_inserter.cli \
  --main path/to/main.mp3 \
  --voice-id <elevenlabs-voice-id> \
  --product-name "Sparrow Notes" \
  --product-desc "A calmer note-taking app for busy teams" \
  --product-url "https://sparrow.example" \
  --mode podcast \
  --out output.mp3 \
  --debug-dir debug
```

Song example:
```bash
python -m ad_inserter.cli \
  --main path/to/song.mp3 \
  --voice-id <elevenlabs-voice-id> \
  --product-name "Pulse Water" \
  --product-desc "Electrolytes without the sugar crash" \
  --mode song \
  --out song_with_ad.mp3
```

## Two-speaker ad insertion
This feature inserts an AI-written ad into a two-person conversation. It can speak as Speaker A, Speaker B, or a short back-and-forth.

### Required env vars
- `OPENAI_API_KEY` for ad script generation (unless `--llm-provider none`)
- `ELEVENLABS_API_KEY` with `--clone-voices`, to delete a cloned voice directly when the API cannot release it; the run refuses to start (exit 2) without it. TTS itself goes through the API's `/api/tts`, so the pipeline needs no ElevenLabs key for it
- A voice for each speaker the ad mode uses: `--voice-id-a`/`ELEVENLABS_VOICE_ID_A` for A, `--voice-id-b`/`ELEVENLABS_VOICE_ID_B` for B, or `ELEVENLABS_DEFAULT_VOICE_ID` as a fallback. `A_ONLY` needs only A, `B_ONLY` only B, `DUO` both

Optional diarization (enables DUO mode and voice cloning):
- Install `pyannote.audio` separately
- Set `HUGGINGFACE_TOKEN` (or `PYANNOTE_TOKEN` / `HF_TOKEN`) for model access; `HF_HOME` / `HF_HUB_CACHE` / `TORCH_HOME` relocate the model caches

### CLI example
```bash
python -m ad_inserter.insert_ad \
  --input path/to/conversation.mp3 \
  --product-name "Notion" \
  --product-blurb "AI-powered productivity workspace" \
  --ad-style casual \
  --ad-mode DUO \
  --out out.mp3
```

Optional voice cloning (requires diarization + ElevenLabs API key). Only the
speakers the ad mode uses are cloned, through the API's `/api/clone`, which
creates a *permanent* ElevenLabs voice and records it in the API's voice
registry. Once the ad is rendered, and also when rendering fails, the run hands
every voice it cloned back with `POST <--clone-url>/{voice_id}/release`
(internal token, 10 s timeout); the API deletes the voice and drops it from the
registry, so it stops counting toward the registry's live-voice cap. Only if
that release fails or answers 404 (a voice the registry does not know) does the
run delete the voice itself, with
`DELETE https://api.elevenlabs.io/v1/voices/{voice_id}` (header `xi-api-key:
$ELEVENLABS_API_KEY`, 30 s timeout). A failed cleanup is logged to stderr and
does not fail the run; check the ElevenLabs voice library for leftover
`Speaker A Clone` / `Speaker B Clone` voices if you see that warning. A speaker
whose diarized sample is empty falls back to the configured voice.
```bash
python -m ad_inserter.insert_ad \
  --input path/to/conversation.mp3 \
  --product-name "Notion" \
  --product-blurb "AI-powered productivity workspace" \
  --ad-style casual \
  --ad-mode A_ONLY \
  --clone-voices \
  --out out.mp3
```

### API example
```bash
curl -X POST http://localhost:3001/ad/insert \
  -F "audio=@path/to/conversation.mp3" \
  -F "productName=Notion" \
  -F "productBlurb=AI-powered productivity workspace" \
  -F "adStyle=casual" \
  -F "adMode=DUO" \
  --output out.mp3
```

## LLM configuration
- `--llm-provider openai` (default `openai`); `none` skips the LLM (product name / blurb, heuristic insertion point)
- Set `OPENAI_API_KEY` in your environment
- `--llm-model` must be in the allow-list `SLOTIFY_OPENAI_MODELS`: a comma-separated list, entries trimmed, empty entries dropped, exact match; unset or empty means `gpt-4o-mini`. It is the variable and parsing the Node API enforces on `/ad/insert`; the pipeline checks it again before decoding anything, and `cli.py`/`insert_ad.py` exit 2 with `OpenAI model '<name>' is not allowed; SLOTIFY_OPENAI_MODELS permits: ...`. Without `--llm-model` the model is `gpt-4o-mini` if allowed, else the first allowed one
- Every OpenAI call sends `max_tokens=400` (`llm.OPENAI_MAX_TOKENS`); the JSON around a ~20 s read is well under 200 tokens. A reply cut off at the limit is malformed and takes the usual fallback. Allow only chat models that accept `max_tokens` (o-series models do not)
- Each script segment becomes one TTS callback carrying the internal token, so the script is capped before any TTS: at most 6 segments (`MAX_AD_SEGMENTS`) and 400 characters in all (`MAX_SCRIPT_CHARS`, a ~20 s read with headroom), cut on a word boundary; segments past the budget are dropped. The same 400-character cap applies to the single-voice promo and to the fallback scripts built from `--product-name`/`--product-blurb`

## Running under the Node API
The Node API spawns one CLI per request (`cwd` = `backend/`) and passes each
one `--work-dir=<dir>`, a directory it creates and removes. Every CLI
(`analyze_cli`, `cli`, `insert_ad`) shares this contract (`ad_inserter/runtime.py`):

- **`--work-dir`** must name an existing directory (exit 2 otherwise). The run
  creates one private directory inside it, points `tempfile` and `TMPDIR` at
  it, and removes it on the way out, so *every* temp file -- the pipeline's own
  (Whisper snippet WAVs, the decoded analysis PCM, the promo MP3 uploaded to
  `/api/merge`, the speaker samples for cloning) and the libraries' (pydub's
  MP3 export temp files) -- lands under `--work-dir` and nothing is written to
  the system temp dir. Without `--work-dir` the private directory is created in
  the system temp dir instead, and is removed the same way. Size the work dir's
  filesystem for the staged audio: about 115 MB of PCM per hour of upload for
  podcast analysis, 635 MB per hour for song analysis, and pydub's temp WAV of
  the whole mix when `insert_ad` exports.
- **SIGTERM and SIGINT** raise `SystemExit`: exit code **143** for SIGTERM, 130
  for SIGINT. `finally` blocks and `with` exits run, so temp files are removed
  and `insert_ad` deletes the voices it cloned. Only the first signal acts;
  later ones are ignored so they cannot cut that cleanup short. The blocking
  calls (ffmpeg subprocess waits, HTTP callbacks with their timeout) return to
  the interpreter on a signal. The analysis decoder kills its ffmpeg child when
  its wait is interrupted; pydub's ffmpeg children (full-quality decode, MP3
  export) are not killed by the pipeline, so signal the whole process group, as
  Node does. Two steps finish before the exit: a `/api/clone` request in
  flight (so its voice ID is known and the voice can be released) and the voice
  cleanup (release, then the direct delete if that fails). Once a signal has
  arrived each of those cleanup requests gets a 1 s timeout, so a release and
  its fallback for two voices fit in Node's 5 s grace period before SIGKILL. A SIGKILL skips all cleanup;
  the work dir is then Node's to remove. Long numeric steps inside a single
  C call (librosa's beat tracking on a long song) react only when they return.

### Environment each CLI reads
Nothing else is read; in particular no CLI reads `API_BASE_URL` (callback URLs
come from `--tts-url`/`--merge-url`/`--clone-url`) or needs `SLOTIFY_API_TOKEN`.

| Variable | `analyze_cli` | `cli` | `insert_ad` | Effect |
| --- | --- | --- | --- | --- |
| `PATH` | required | required | required | Must find `ffmpeg` and `ffprobe`; the run fails up front if `ffmpeg` is missing. |
| `TMPDIR` | optional | optional | optional | System temp dir, used only without `--work-dir`. The run overrides it for its own children. |
| `HOME` / `XDG_CACHE_HOME` | optional | optional | optional | Whisper's model cache (`$XDG_CACHE_HOME/whisper`, default `~/.cache/whisper`) when `openai-whisper` is installed; a missing model is downloaded on first use. |
| `OPENAI_API_KEY` | -- | required* | required* | *With `--llm-provider openai` (the default). The OpenAI SDK also honours `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`. |
| `SLOTIFY_OPENAI_MODELS` | -- | optional | optional | Model allow-list, default `gpt-4o-mini` (see [LLM configuration](#llm-configuration)). |
| `SLOTIFY_INTERNAL_TOKEN` | -- | required* | required* | *When the API authenticates callbacks: `/api/tts` and `/api/merge` (`cli`), `/api/tts` and `/api/clone` (`insert_ad`). See [Callbacks to the API](#callbacks-to-the-api). |
| `SLOTIFY_API_TOKEN` | -- | optional | optional | Fallback callback credential for hand runs; Node does not pass it. |
| `SLOTIFY_CALLBACK_TIMEOUT_S` | -- | optional | optional | Callback timeout, default 300 s. |
| `ELEVENLABS_API_KEY` | -- | -- | required with `--clone-voices` | Deletes a cloned voice directly when `/api/clone/{id}/release` fails or answers 404; the run exits 2 without it. |
| `ELEVENLABS_VOICE_ID_A` / `_B`, `ELEVENLABS_DEFAULT_VOICE_ID` | -- | -- | optional | Voices for speakers without `--voice-id-a`/`-b` (or a usable clone); a needed speaker with none fails with `Missing ElevenLabs voice ID for speaker A`. |
| `HUGGINGFACE_TOKEN` / `PYANNOTE_TOKEN` / `HF_TOKEN` | -- | -- | optional | Diarization with `pyannote.audio` (the token for its gated model; first set one wins). Without one `DUO` and `--clone-voices` fail with an error that names these variables. |
| `HF_HOME` / `HF_HUB_CACHE` | -- | -- | optional | Where `huggingface_hub` caches the pyannote model (default `~/.cache/huggingface`); read by the library, not by `ad_inserter`. |
| `TORCH_HOME` | -- | -- | optional | torch's cache dir for the weights pyannote loads (default `~/.cache/torch`); read by the library. |

## Memory
Analysis never holds the upload at full quality. `analysis.load_analysis_audio`
has ffmpeg decode it straight to mono 16-bit 16 kHz PCM (the format Whisper
transcribes), staged in the private temp dir and read back in one allocation:
about 115 MB per hour of audio, where the old 44.1 kHz stereo decode held about
635 MB plus transient copies. Silence thresholds are relative to the clip's own
dBFS and the downmix is the channel mean, so detection is unchanged: on
`audio_tests/` the podcast output is identical for `ryan-test1.mp3` and
`ryan-rap.mp3`, and for the 48 kHz `catsvdogs.mp3` and `convo.mp3` (resampled
differently than before) a silence midpoint moves by at most 1 ms and a length
by at most 2 ms. Measured on a 14 MB, 1-hour 44.1 kHz stereo MP3
(`analyze_cli --mode podcast`): peak RSS 1853 MB before, 143 MB after, wall time
206 s before, 59 s after, with identical candidates.

Song mode (`analysis.load_song_samples`) decodes to mono float32 but keeps
44.1 kHz: librosa's beat tracker works in frames of a fixed number of samples,
and at 22.05 kHz the `audio_tests/` tempos and beats change. Its output on
`audio_tests/` is byte-identical to before. Its peak is librosa's beat tracking,
about 180 MB per minute of audio (2.9 GB for a 15-minute file, 3.2 GB before),
so long uploads should not be analysed in song mode.

Where the full-quality audio is needed it is decoded only as far as needed:
`cli.py` takes candidates and snippets from the analysis copy and decodes just
the 8 s around the insertion point at 44.1 kHz stereo for the loudness context
and room tone (`analysis.load_audio_window`), since `/api/merge` splices the
original file. `insert_ad.py` splices locally, so it still decodes the whole
upload at full quality.

## Callbacks to the API
`cli.py` and `insert_ad.py` call back into the Node API (`/api/tts`,
`/api/merge`, `/api/clone`). Every such request goes through
`ad_inserter/http_client.py`, which reads:

| Variable | Effect |
| --- | --- |
| `SLOTIFY_INTERNAL_TOKEN` | Sent as `Authorization: Bearer <token>`. The Node API sets it in the environment of the pipelines that call back (`cli`, `insert_ad`). |
| `SLOTIFY_API_TOKEN` | Fallback credential when `SLOTIFY_INTERNAL_TOKEN` is unset (e.g. running a CLI by hand against a deployment that sets one). With neither set, no `Authorization` header is sent. |
| `SLOTIFY_CALLBACK_TIMEOUT_S` | Per-request timeout in seconds, default `300`. A slow or hung API now fails the run with `... request timed out` instead of hanging it. |

The internal credential is never sent to third parties: the ElevenLabs voice
cleanup uses only `xi-api-key`.

## Tests
```bash
python -m pytest backend/tests/python -q   # from the repository root
```
They need `backend/requirements.txt` and `pytest`; no network, API keys or
Whisper. Tests that decode through ffmpeg (all analysis does, WAV included) or
encode MP3 are skipped when `ffmpeg` is not on `PATH`.
`test_ad_inserter_runtime.py` runs each CLI as a subprocess, as Node does,
blocks it in a stub (Whisper, a hung ffmpeg, a hung `/api/merge` or `/api/tts`
on a loopback server), sends SIGTERM to it or its process group, and checks the
exit code 143, that the work dir and the system temp dir are left empty, and
that `insert_ad` released both voices it cloned (and deleted directly the one
the API answered 404 for).
