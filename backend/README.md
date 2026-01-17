# Ad Inserter Backend

This module inserts a provided promo audio clip into a main audio track (podcast or song). It can optionally use an LLM to generate a 1-sentence promo text and choose the best insertion point based on semantic context.

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
- Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your environment.
