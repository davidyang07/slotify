/**
 * Readiness: can this process actually do the core job right now?
 *
 * /api/health used to answer "ok" whenever Node was up, including on a box
 * with no ffmpeg or with a PYTHON_BIN that cannot import the analyser's
 * dependencies -- where every analysis fails. It now reports the checks below.
 * They spawn processes (importing librosa takes seconds), so they run once at
 * boot and their result is cached; a request for a result older than
 * `maxAgeMs` gets the cached one at once and starts a fresh check in the
 * background, so a probe with a short timeout is never kept waiting.
 */

import { BACKEND_DIR, pythonBin } from "../config";
import { clientSafeDetail } from "../lib/errors";
import { pythonEnv, runProcess, SubprocessError } from "../lib/subprocess";

export interface ReadinessCheck {
  name: string;
  /** Resolves when the dependency is usable; rejects with why not. */
  run: () => Promise<void>;
}

export interface CheckResult {
  ok: boolean;
  /** Client-safe reason, when not ok. */
  error?: string;
}

export interface ReadinessReport {
  ready: boolean;
  checkedAt: string;
  checks: Record<string, CheckResult>;
  failed: string[];
}

/** The analyser's imports; a missing one fails every analysis. */
export const ANALYSER_MODULES = ["pydub", "numpy", "librosa", "pyloudnorm", "soundfile"];

const CHECK_TIMEOUT_MS = 60_000;

const describe = (error: unknown): string => {
  if (error instanceof SubprocessError) {
    return (error.reason === "exit" && clientSafeDetail(error.stderr)) || clientSafeDetail(error.message);
  }
  return clientSafeDetail(error instanceof Error ? error.message : String(error));
};

export const defaultChecks = (): ReadinessCheck[] => [
  {
    name: "ffmpeg",
    run: async () => {
      await runProcess("ffmpeg", ["-hide_banner", "-version"], { label: "ffmpeg", timeoutMs: CHECK_TIMEOUT_MS });
    },
  },
  {
    name: "ffprobe",
    run: async () => {
      await runProcess("ffprobe", ["-hide_banner", "-version"], { label: "ffprobe", timeoutMs: CHECK_TIMEOUT_MS });
    },
  },
  {
    name: "analyser",
    run: async () => {
      await runProcess(pythonBin, ["-c", `import ${ANALYSER_MODULES.join(", ")}`], {
        label: "the analyser's Python",
        cwd: BACKEND_DIR,
        env: pythonEnv(),
        timeoutMs: CHECK_TIMEOUT_MS,
      });
    },
  },
];

export const createReadiness = ({
  checks,
  maxAgeMs = 60_000,
  now = Date.now,
}: {
  checks: ReadinessCheck[];
  maxAgeMs?: number;
  now?: () => number;
}) => {
  let cached: { at: number; report: ReadinessReport } | null = null;
  let inFlight: Promise<ReadinessReport> | null = null;

  const runAll = (): Promise<ReadinessReport> => {
    inFlight ??= (async () => {
      const results = await Promise.all(
        checks.map(async (check): Promise<[string, CheckResult]> => {
          try {
            await check.run();
            return [check.name, { ok: true }];
          } catch (error) {
            return [check.name, { ok: false, error: describe(error) }];
          }
        }),
      );
      const report: ReadinessReport = {
        ready: results.every(([, result]) => result.ok),
        checkedAt: new Date(now()).toISOString(),
        checks: Object.fromEntries(results),
        failed: results.filter(([, result]) => !result.ok).map(([name]) => name),
      };
      cached = { at: now(), report };
      return report;
    })().finally(() => {
      inFlight = null;
    });
    return inFlight;
  };

  return {
    /** The cached report (re-checked in the background when stale), or the first one. */
    report: async (): Promise<ReadinessReport> => {
      if (!cached) return runAll();
      if (now() - cached.at >= maxAgeMs) void runAll();
      return cached.report;
    },
    /** Run the checks now; used at boot. */
    refresh: runAll,
  };
};

export type Readiness = ReturnType<typeof createReadiness>;

/** The process-wide readiness, over the real dependencies. */
export const readiness = createReadiness({ checks: defaultChecks() });
