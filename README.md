# Seamless Sponsor Insertion - UofTHacks 13

Pipeline: Upload an audio file with speech → AI recommends the best insertion point(s) → ElevenLabs generates a human-like ad read → the system stitches it in smoothly and exports the final audio.

Demo: https://www.youtube.com/watch?v=S4m1lpipni0

## Core features
- User input for podcast audio and product name - product details are optional
- AI-recommended insertion timestamps using semantic + syntactic analysis
- ElevenLabs TTS for realistic sponsor reads (single speaker or multi-way conversation)
- Preview insertions before rendering final output
- Export monetized episodes with loudness matching + crossfades

## Tech stack
- Frontend: React + TypeScript + Vite
- Backend API: Node.js + Express
- Audio pipeline: Python (pydub, librosa, pyloudnorm)
- AI services: OpenAI (placement + copy), ElevenLabs (TTS/voice cloning)
- Media tools: ffmpeg/ffprobe

## Local deployment
### Prereqs
- Node.js (for `frontend/` and `backend/`)
- Python (for `backend/ad_inserter`)
- `ffmpeg` + `ffprobe`

### 1) Backend API
```bash
cd backend
npm install
pip install -r requirements.txt
```

Set env vars (examples):
```bash
export ELEVENLABS_API_KEY="..."
export OPENAI_API_KEY="..."
```

Run the API:
```bash
npm run dev
```

The API listens on `http://localhost:3001`.

### 2) Frontend
```bash
cd frontend
npm install
npm run dev
```

The UI runs on `http://localhost:5173` and calls the backend.

---

## Ad Inserter backend module
This module generates and inserts a product promotion into a main audio track (podcast or rhythmic song). An LLM generates the 1-sentence promo text and choose the best insertion point based on semantic + syntactic context.

### What each file in `backend/ad_inserter/` does
- `__init__.py` exposes the package modules (analysis, llm, mix) and version
- `analysis.py` handles audio analysis: ffmpeg check, loading/standardizing audio, silence-based candidate detection for podcasts, beat/RMS analysis for songs, optional Whisper transcription, and building candidate payloads
- `analyze_cli.py` exposes a CLI helper that runs analysis and returns JSON for the Node API
- `cli.py` provides the single-speaker CLI workflow: parse args, pick candidates, call LLM to write promo/choose insertion, loudness match + room tone + crossfade, and export output (plus debug artifacts)
- `insert_ad.py` handles two-speaker insertion (A/B/DUO), optional diarization, and optional voice cloning
- `llm.py` builds the prompt and calls OpenAI to generate promo text and choose insertion index; parses JSON response into `LLMResult`
- `mix.py` does audio mixing utilities: LUFS measurement, loudness matching, looping room tone, ducking, crossfade insertion, and context window extraction
- `tts.py` builds sponsor reads with ElevenLabs (single or multi-statement blocks)

### Install
```bash
pip install -r backend/requirements.txt
```

Install ffmpeg (required by pydub):
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`

Optional: install Whisper locally for transcripts:
```bash
pip install openai-whisper
```

### Usage
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

## How it works
Semantic context:
- Uses Whisper locally (if installed) to transcribe short context windows around candidate insertion points
- The LLM selects the best insertion point based on topic transitions and sentence boundaries, and writes a 1-sentence promo matching the tone
- If Whisper is not available, it falls back to silence-based insertion

Rhythmic/syntactic context:
- Uses librosa to estimate tempo and beat times
- Finds low-energy (RMS) valleys, snaps to the nearest beat, and inserts the promo there

## LLM configuration
- `--llm-provider openai` (default `openai`)
- Set `OPENAI_API_KEY` in your environment
