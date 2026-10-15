import { Router } from "express";
import { asyncHandler } from "../lib/http";
import { readiness as defaultReadiness, type Readiness } from "../services/readiness";
import { readCapabilities } from "./capabilities";

/**
 * GET (and HEAD) /api/health: a readiness check. 200 when ffmpeg, ffprobe and
 * the analyser's Python imports work; 503 naming the check that failed.
 */
export const createHealthRouter = (readiness: Readiness = defaultReadiness) => {
  const router = Router();
  router.get(
    "/api/health",
    asyncHandler(async (_req, res) => {
      const report = await readiness.report();
      const capabilities = readCapabilities();
      res.status(report.ready ? 200 : 503).json({
        status: report.ready ? "ok" : "unavailable",
        ...(report.ready
          ? {}
          : { error: `Not ready: ${report.failed.join(", ")} unavailable.` }),
        timestamp: new Date().toISOString(),
        checkedAt: report.checkedAt,
        checks: report.checks,
        failed: report.failed,
        elevenlabsConfigured: capabilities.tts,
        capabilities,
      });
    }),
  );
  return router;
};

export const healthRouter = createHealthRouter();
