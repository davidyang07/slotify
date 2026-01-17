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

const response = await fetch(`${apiBase}/api/clone`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
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

const stream = Readable.fromWeb(response.body);
await play(stream);

