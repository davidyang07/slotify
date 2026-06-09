import "dotenv/config";
import cors from "cors";
import express from "express";

import { port, allowedOrigins } from "./config";
import { healthRouter } from "./routes/health";
import { cloneRouter } from "./routes/clone";
import { mergeRouter } from "./routes/merge";
import { insertSectionsRouter } from "./routes/insert-sections";
import { relatedProductsRouter } from "./routes/related-products";
import { ttsRouter } from "./routes/tts";
import { generateRouter } from "./routes/generate";
import { adInsertRouter } from "./routes/ad-insert";

// Check for ElevenLabs API key
if (!process.env.ELEVENLABS_API_KEY) {
  console.warn(
    "Warning: ELEVENLABS_API_KEY not set. Voice cloning and TTS will fail.",
  );
}

const app = express();

app.use(
  cors({
    origin(origin, callback) {
      if (!origin || allowedOrigins.includes(origin)) {
        callback(null, true);
        return;
      }
      callback(new Error("CORS blocked"));
    },
  }),
);
app.use(express.json({ limit: "2mb" }));

app.use(healthRouter);
app.use(cloneRouter);
app.use(mergeRouter);
app.use(insertSectionsRouter);
app.use(relatedProductsRouter);
app.use(ttsRouter);
app.use(generateRouter);
app.use(adInsertRouter);

app.listen(port, () => {
  console.log(`API listening on http://localhost:${port}`);
  console.log(`CORS allowed origins: ${allowedOrigins.join(", ")}`);
  if (!process.env.ELEVENLABS_API_KEY) {
    console.warn(
      "⚠️  ELEVENLABS_API_KEY not set. Voice cloning and TTS endpoints will fail.",
    );
  } else {
    console.log("✓ ElevenLabs API key configured");
  }
});
