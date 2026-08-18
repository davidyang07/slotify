import { Router } from "express";
import { readCapabilities } from "./capabilities";

export const healthRouter = Router();

healthRouter.get("/api/health", (_req, res) => {
  const capabilities = readCapabilities();
  res.json({
    status: "ok",
    timestamp: new Date().toISOString(),
    elevenlabsConfigured: capabilities.tts,
    capabilities,
  });
});
