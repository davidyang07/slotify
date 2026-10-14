/**
 * CLI helper: synthesize speech for a voice clone and play it locally.
 *
 * Usage:
 *   npx tsx scripts/speak.ts <voiceId>
 *
 * Set SLOTIFY_API_TOKEN when the API requires one.
 */

import { play } from "@elevenlabs/elevenlabs-js";
import { Readable } from "node:stream";
import "dotenv/config";

const apiBase = process.env.VITE_API_BASE_URL ?? "http://localhost:3001";
const voiceId = process.argv[2] ?? process.env.VOICE_ID;
const text =
  process.env.TTS_TEXT ??
  "Hello, this is a test of the voice cloned on ElevenLabs.";

if (!voiceId) {
  throw new Error("Provide a voiceId as argv[2] or VOICE_ID.");
}

const token = process.env.SLOTIFY_API_TOKEN?.trim();
const response = await fetch(`${apiBase}/api/tts`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  },
  body: JSON.stringify({
    voiceId,
    text,
    modelId: "eleven_multilingual_v2",
    outputFormat: "mp3_44100_128",
    voiceSettings: {
      stability: 0,
      similarityBoost: 0,
      useSpeakerBoost: true,
      speed: 0.7,
    },
  }),
});

if (!response.ok) {
  const errorText = await response.text();
  throw new Error(`TTS failed: ${response.status} ${errorText}`);
}

if (!response.body) {
  throw new Error("No audio stream returned.");
}

const stream = Readable.fromWeb(response.body as any);
await play(stream);
