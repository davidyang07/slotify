#TODO: combine with main readme

# Ad Inserter Backend

This module inserts a provided promo audio clip into a main audio track (podcast or song). It can optionally use an LLM to generate a 1-sentence promo text and choose the best insertion point based on semantic context.

prompt engineering for intelligent choice of insertion point

# Here’s what each file in backend/ad_inserter/ does:

`__init__.py` exposes the package modules (analysis, llm, mix) and version.
`analysis.py` handles audio analysis: ffmpeg check, loading/standardizing audio, silence-based candidate detection for podcasts, beat/RMS analysis for songs, optional Whisper transcription, and building candidate payloads.
`cli.py` provides the CLI workflow: parse args, pick candidates, call LLM to write promo/choose insertion, loudness match + room tone + crossfade, and export output (plus debug artifacts).
`llm.py` builds the prompt and calls OpenAI to generate promo text and choose insertion index; parses JSON response into LLMResult.
`mix.py` does audio mixing utilities: LUFS measurement, loudness matching, looping room tone, ducking, crossfade insertion, and context window extraction.

## Install

1) Install Python deps:

```bash
pip install -r backend/requirements.txt
```

2) Install ffmpeg (required by pydub):

- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`

3) Optional: install Whisper locally for transcripts:

```bash
pip install openai-whisper
```

## Usage

Run from the `backend/` directory so `python -m ad_inserter.cli` can find the package.

Podcast example:

```bash
python -m ad_inserter.cli \
  --main path/to/main.mp3 \
  --promo-audio path/to/promo.wav \
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
  --promo-audio path/to/promo.mp3 \
  --product-name "Pulse Water" \
  --product-desc "Electrolytes without the sugar crash" \
  --mode song \
  --out song_with_ad.mp3
```

## Two-speaker ad insertion

This feature inserts an AI-written ad into a two-person conversation. It can speak as Speaker A, Speaker B, or a short back-and-forth.

### Required env vars

- `OPENAI_API_KEY` for ad script generation (unless `--llm-provider none`).
- `ELEVENLABS_API_KEY` for TTS.
- `ELEVENLABS_VOICE_ID_A` and `ELEVENLABS_VOICE_ID_B` for speaker mapping, or set `ELEVENLABS_DEFAULT_VOICE_ID` as a fallback.

Optional diarization (enables DUO mode and voice cloning):
- Install `pyannote.audio` separately.
- Set `HUGGINGFACE_TOKEN` (or `PYANNOTE_TOKEN`) for model access.

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

## How it works

Semantic context (podcasts):
- Uses Whisper locally (if installed) to transcribe short context windows around candidate insertion points.
- The LLM selects the best insertion point based on topic transitions and sentence boundaries, and writes a 1-sentence promo matching the tone.
- If Whisper is not available, it falls back to silence-based insertion.

Rhythmic context (songs):
- Uses librosa to estimate tempo and beat times.
- Finds low-energy (RMS) valleys, snaps to the nearest beat, and inserts the promo there.

## LLM configuration

- `--llm-provider openai` (default `openai`).
- Set `OPENAI_API_KEY` in your environment.
