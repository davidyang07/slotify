import { BACKEND_DIR, analyserTimeoutMs, pipelineTimeoutMs, pythonBin } from "../config";
import { HEURISTIC_V1 } from "../lib/heuristic-config";
import { pythonEnv, runProcess, SubprocessError } from "../lib/subprocess";

// Candidate and snippet budgets come from the canonical config, like every
// other baseline constant; the analyser keeps the longest silences.
const { max_candidates: MAX_CANDIDATES, snippet_count: SNIPPET_COUNT } =
  HEURISTIC_V1.silence_detection;

/**
 * Run `python -m ad_inserter.analyze_cli` for the given audio and parse its
 * JSON stdout. Resolves to the parsed analysis payload.
 */
export const runPythonAnalyze = async (
  audioPath: string,
  mode: string,
  { signal }: { signal?: AbortSignal } = {},
): Promise<any> => {
  const args = [
    "-m",
    "ad_inserter.analyze_cli",
    "--audio",
    audioPath,
    "--mode",
    mode,
    "--max-candidates",
    String(MAX_CANDIDATES),
    "--snippet-count",
    String(SNIPPET_COUNT),
  ];
  const { stdout } = await runProcess(pythonBin, args, {
    label: "analyze_cli",
    cwd: BACKEND_DIR,
    env: pythonEnv(),
    timeoutMs: analyserTimeoutMs,
    timeoutEnvVar: "SLOTIFY_ANALYSER_TIMEOUT_MS",
    signal,
  });
  try {
    return JSON.parse(stdout.trim() || "{}");
  } catch (error) {
    throw new SubprocessError(
      "analyze_cli printed output that is not JSON.",
      "analyze_cli",
      "output",
      `${error instanceof Error ? error.message : String(error)}\n${stdout.slice(0, 2000)}`,
    );
  }
};

/**
 * Run a Python `ad_inserter` module that writes its result to a file (the
 * single- and two-speaker insertion pipelines). Resolves when it exits 0.
 */
export const runPythonModule = async (
  args: string[],
  label = "ad_inserter",
  { signal }: { signal?: AbortSignal } = {},
): Promise<void> => {
  await runProcess(pythonBin, args, {
    label,
    cwd: BACKEND_DIR,
    env: pythonEnv(),
    timeoutMs: pipelineTimeoutMs,
    timeoutEnvVar: "SLOTIFY_PIPELINE_TIMEOUT_MS",
    signal,
  });
};
