# The `ad_inserter` pipeline

The hackathon build's audio analysis, mixing and two-speaker insertion, moved
out of the README when that was rewritten around the ranker. Nothing here has
changed: these are the same modules and the same CLI flags.

Run every command from the `backend/` directory so `python -m ad_inserter.*`
resolves the package.

Where this sits relative to the rest of the system: `ad_inserter` does candidate
*detection* and audio *mixing*. Candidate *ranking* is `ml/src/slotify_rank`
(see [`model-inference.md`](model-inference.md)). The two are separate on
purpose — mixing has nothing to learn, and ranking has nothing to render.

## Ad Inserter backend module
- `__init__.py` exposes the package modules (analysis, llm, mix, tts, insert_ad) and version
- `analysis.py` handles audio analysis: ffmpeg check, loading/standardizing audio, silence-based candidate detection for podcasts, beat/RMS analysis for songs, optional Whisper transcription, and building candidate payloads
- `analyze_cli.py` exposes a CLI helper that runs analysis and returns JSON for the Node API
- `cli.py` provides the single-speaker CLI workflow: parse args, pick candidates, call LLM to write promo/choose insertion, loudness match + room tone + crossfade, and export output (plus debug artifacts)
- `insert_ad.py` handles two-speaker insertion (A/B/DUO), optional diarization, and optional voice cloning
- `llm.py` builds the prompt and calls OpenAI to generate promo text and choose insertion index; parses JSON response into `LLMResult`
- `mix.py` does audio mixing utilities: LUFS measurement, loudness matching, looping room tone, ducking, crossfade insertion, and context window extraction
- `tts.py` builds sponsor reads with ElevenLabs (single or multi-statement blocks)

### How is insertion point chosen?
Semantic context:
- Uses Whisper locally (if installed) to transcribe short context windows around candidate insertion points before evaluating topic transitions and sentence boundaries
- If Whisper is not available, fall back to silence-based insertion

Rhythmic/syntactic context:
- Uses librosa to estimate tempo and beat times
- Finds low-energy (RMS) valleys, snaps to the nearest beat, and inserts the promo there

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
- `ELEVENLABS_API_KEY` for TTS
- `ELEVENLABS_VOICE_ID_A` and `ELEVENLABS_VOICE_ID_B` for speaker mapping, or set `ELEVENLABS_DEFAULT_VOICE_ID` as a fallback

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

Optional voice cloning (requires diarization + ElevenLabs API key):
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
