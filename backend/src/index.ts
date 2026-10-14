import "dotenv/config";

import { createApp } from "./app";
import { port, allowedOrigins, configuredApiToken, rateLimitsFromEnv } from "./config";
import { readCapabilities } from "./routes/capabilities";

const app = createApp();

app.listen(port, () => {
  const capabilities = readCapabilities();
  const limits = rateLimitsFromEnv();
  console.log(`API listening on http://localhost:${port}`);
  console.log(`CORS allowed origins: ${allowedOrigins.join(", ")}`);
  // Placement is the core demo and needs no credentials, so the startup banner
  // states what works rather than only what is missing.
  console.log(`Placement: enabled (${capabilities.ranker.active} ranker)`);
  console.log(`  ${capabilities.ranker.reason}`);
  console.log(
    `Voice cloning / TTS: ${capabilities.tts ? "enabled" : "disabled (ELEVENLABS_API_KEY not set)"}`,
  );
  console.log(
    `OpenAI enrichment: ${
      capabilities.openaiEnhancement ? "enabled" : "disabled (OPENAI_API_KEY not set)"
    }`,
  );
  console.log(
    `Auth: ${configuredApiToken() ? "bearer token required (SLOTIFY_API_TOKEN)" : "open (SLOTIFY_API_TOKEN not set)"}`,
  );
  const describe = (perMin: number) => (perMin > 0 ? `${perMin}/min` : "off");
  console.log(
    `Rate limits per client IP: paid ${describe(limits.paidPerMin)}, compute ${describe(limits.computePerMin)}`,
  );
});
