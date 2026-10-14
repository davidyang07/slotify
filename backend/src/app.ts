import cors from "cors";
import express from "express";

import { allowedOrigins, rateLimitsFromEnv } from "./config";
import { requireApiToken } from "./middleware/auth";
import { CorsRejectedError, errorHandler } from "./middleware/error-handler";
import { createRateLimiter } from "./middleware/rate-limit";
import { healthRouter } from "./routes/health";
import { capabilitiesRouter } from "./routes/capabilities";
import { cloneRouter } from "./routes/clone";
import { mergeRouter } from "./routes/merge";
import { insertSectionsRouter } from "./routes/insert-sections";
import { relatedProductsRouter } from "./routes/related-products";
import { ttsRouter } from "./routes/tts";
import { generateRouter } from "./routes/generate";
import { adInsertRouter } from "./routes/ad-insert";

/** The API without a listener, so tests can drive the real middleware stack. */
export const createApp = () => {
  const app = express();

  app.use(
    cors({
      origin(origin, callback) {
        if (!origin || allowedOrigins.includes(origin)) {
          callback(null, true);
          return;
        }
        callback(new CorsRejectedError());
      },
    }),
  );
  // Auth and rate limits run before any body is parsed or buffered.
  app.use(requireApiToken);
  app.use(createRateLimiter(rateLimitsFromEnv()));
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

  app.use(errorHandler);
  return app;
};
