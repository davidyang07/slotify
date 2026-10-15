import { BACKEND_DIR, analyserTimeoutMs, pipelineTimeoutMs, pythonBin } from "../config";
import { HEURISTIC_V1 } from "../lib/heuristic-config";
import { childEnv, runProcess, SubprocessError, type ChildKind } from "../lib/subprocess";

// Candidate and snippet budgets come from the canonical config, like every
// other baseline constant; the analyser keeps the longest silences.
const { max_candidates: MAX_CANDIDATES, snippet_count: SNIPPET_COUNT } =
  HEURISTIC_V1.silence_detection;

/**
 * The `ad_inserter` CLIs put every temp file under `--work-dir`, the request's
 * own temp dir, which the route removes in its `finally`. TMPDIR points there
 * too, so a library that reaches for the system temp dir (pydub's decode
 * buffers) also writes where it will be cleaned up.
 */
const workDirArgs = (workDir: string) => [`--work-dir=${workDir}`];
const workDirEnv = (kind: ChildKind, workDir: string) => childEnv(kind, { TMPDIR: workDir });

/**
 * Run `python -m ad_inserter.analyze_cli` for the given audio and parse its
 * JSON stdout. Resolves to the parsed analysis payload.
 */
export const runPythonAnalyze = async (
  audioPath: string,
  mode: string,
  { signal, workDir }: { signal?: AbortSignal; workDir: string },
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
    ...workDirArgs(workDir),
  ];
  const { stdout } = await runProcess(pythonBin, args, {
    label: "analyze_cli",
    cwd: BACKEND_DIR,
    // Local signal processing: no provider key, no callback token.
    env: workDirEnv("python", workDir),
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
 * Run a Python `ad_inserter` pipeline that writes its result to a file (the
 * single- and two-speaker insertion pipelines). Resolves when it exits 0.
 * `kind` picks the environment allow-list: what that pipeline calls.
 */
export const runPythonModule = async (
  args: string[],
  label: string,
  { signal, workDir, kind }: { signal?: AbortSignal; workDir: string; kind: "generate" | "ad-insert" },
): Promise<void> => {
  await runProcess(pythonBin, [...args, ...workDirArgs(workDir)], {
    label,
    cwd: BACKEND_DIR,
    env: workDirEnv(kind, workDir),
    timeoutMs: pipelineTimeoutMs,
    timeoutEnvVar: "SLOTIFY_PIPELINE_TIMEOUT_MS",
    signal,
  });
};
