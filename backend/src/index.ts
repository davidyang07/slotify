import "dotenv/config";

import { createApp } from "./app";
import { port, allowedOrigins, configuredApiToken, rateLimitsFromEnv } from "./config";
import { isShuttingDown, runShutdownHooks } from "./lib/lifecycle";
import { removeAllTempDirsSync } from "./lib/temp";
import { readCapabilities } from "./routes/capabilities";
import { readiness } from "./services/readiness";

const app = createApp();

/** Shutdown hooks (children, cloned voices) share this deadline. */
const SHUTDOWN_DEADLINE_MS = 15_000;
// After the hooks, how long in-flight requests get to answer and run their
// `finally` blocks before their connections are cut.
const DRAIN_MS = 2_000;

const server = app.listen(port, () => {
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
  // Once at boot; /api/health serves the cached result (see services/readiness.ts).
  void readiness.refresh().then((report) => {
    console.log(
      report.ready
        ? "Readiness: ok (ffmpeg, ffprobe, analyser imports)"
        : `Readiness: NOT READY -- ${report.failed
            .map((name) => `${name}: ${report.checks[name]?.error ?? "failed"}`)
            .join("; ")}`,
    );
  });
});

/**
 * SIGTERM/SIGINT: stop accepting connections, refuse new heavy jobs (503),
 * stop every child process (SIGTERM, grace, SIGKILL) and delete the cloned
 * voices, let in-flight requests finish their cleanup, remove any temp dir
 * still left, and exit 0. A second signal exits at once.
 */
const shutdown = async (signal: NodeJS.Signals) => {
  if (isShuttingDown()) {
    console.warn(`${signal} received again; exiting now.`);
    process.exit(1);
  }
  console.log(`${signal} received; shutting down.`);
  const closed = new Promise<void>((resolve) => server.close(() => resolve()));
  server.closeIdleConnections();
  await runShutdownHooks(SHUTDOWN_DEADLINE_MS);
  // A keep-alive connection becomes idle once its response is sent; close
  // each as it does, rather than waiting out the drain for all of them.
  const drain = setInterval(() => server.closeIdleConnections(), 50);
  await Promise.race([closed, new Promise((resolve) => setTimeout(resolve, DRAIN_MS))]);
  clearInterval(drain);
  server.closeAllConnections();
  removeAllTempDirsSync();
  console.log("Shutdown complete.");
  process.exit(0);
};

for (const signal of ["SIGTERM", "SIGINT"] as const) {
  process.on(signal, () => {
    void shutdown(signal);
  });
}
