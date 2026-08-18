import "dotenv/config";
import cors from "cors";
import express from "express";

import { port, allowedOrigins } from "./config";
import { healthRouter } from "./routes/health";
import { capabilitiesRouter, readCapabilities } from "./routes/capabilities";
import { cloneRouter } from "./routes/clone";
import { mergeRouter } from "./routes/merge";
import { insertSectionsRouter } from "./routes/insert-sections";
import { relatedProductsRouter } from "./routes/related-products";
import { ttsRouter } from "./routes/tts";
import { generateRouter } from "./routes/generate";
import { adInsertRouter } from "./routes/ad-insert";


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
app.use(capabilitiesRouter);
app.use(cloneRouter);
app.use(mergeRouter);
app.use(insertSectionsRouter);
app.use(relatedProductsRouter);
app.use(ttsRouter);
app.use(generateRouter);
app.use(adInsertRouter);

app.listen(port, () => {
  const capabilities = readCapabilities();
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
});
