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
- `analysis.py` handles audio analysis: ffmpeg check, loading/standardizing audio, silence-based candidate detection for podcasts, beat/RMS analysis for songs, optional Whisper transcription, and building candidate payloads
- `analyze_cli.py` exposes a CLI helper that runs analysis and returns JSON for the Node API (`POST /api/insert-sections`)
- `cli.py` provides the single-speaker CLI workflow: parse args, pick candidates, call LLM to write promo/choose insertion, loudness match + room tone, then hand the splice to the API's `/api/merge` (plus debug artifacts)
- `insert_ad.py` handles two-speaker insertion (A/B/DUO), optional diarization, and optional voice cloning; it splices the ad in locally with `mix.insert_with_crossfade`
- `llm.py` builds the prompt and calls OpenAI to generate promo text and choose insertion index; parses JSON response into `LLMResult`. A malformed reply (not JSON, not an object, missing `promo_text`/`chosen_index`, a non-integer or out-of-range index) falls back to the product name and the heuristic insertion point instead of crashing
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
- `ELEVENLABS_API_KEY` for TTS, and for deleting cloned voices
- A voice for each speaker the ad mode uses: `--voice-id-a`/`ELEVENLABS_VOICE_ID_A` for A, `--voice-id-b`/`ELEVENLABS_VOICE_ID_B` for B, or `ELEVENLABS_DEFAULT_VOICE_ID` as a fallback. `A_ONLY` needs only A, `B_ONLY` only B, `DUO` both

Optional diarization (enables DUO mode and voice cloning):
- Install `pyannote.audio` separately
- Set `HUGGINGFACE_TOKEN` (or `PYANNOTE_TOKEN`) for model access

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
creates a *permanent* ElevenLabs voice. The run deletes every voice it cloned
once the ad is rendered, and also when rendering fails, with
`DELETE https://api.elevenlabs.io/v1/voices/{voice_id}` (header `xi-api-key:
$ELEVENLABS_API_KEY`, 30 s timeout). A failed delete is logged to stderr and does
not fail the run; check the ElevenLabs voice library for leftover
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
- `--llm-provider openai` (default `openai`)
- Set `OPENAI_API_KEY` in your environment

## Callbacks to the API
`cli.py` and `insert_ad.py` call back into the Node API (`/api/tts`,
`/api/merge`, `/api/clone`). Every such request goes through
`ad_inserter/http_client.py`, which reads:

| Variable | Effect |
| --- | --- |
| `SLOTIFY_INTERNAL_TOKEN` | Sent as `Authorization: Bearer <token>`. The Node API sets it in the environment of every Python child it spawns. |
| `SLOTIFY_API_TOKEN` | Fallback credential when `SLOTIFY_INTERNAL_TOKEN` is unset (e.g. running a CLI by hand against a deployment that sets one). With neither set, no `Authorization` header is sent. |
| `SLOTIFY_CALLBACK_TIMEOUT_S` | Per-request timeout in seconds, default `300`. A slow or hung API now fails the run with `... request timed out` instead of hanging it. |

The internal credential is never sent to third parties: the ElevenLabs voice
cleanup uses only `xi-api-key`.

## Tests
```bash
python -m pytest backend/tests/python -q   # from the repository root
```
They need `backend/requirements.txt` and `pytest`; no network, API keys or
Whisper. Tests that encode or decode MP3 are skipped when `ffmpeg` is not on
`PATH`.
