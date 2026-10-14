/**
 * Bridge from the Express API to the PyTorch ranker in `ml/`.
 *
 * A subprocess, not an HTTP service. The repository already runs Python this
 * way for `ad_inserter`, the model loads in ~0.15 s against a feature pass that
 * takes tens of seconds, and a second long-lived service is a second thing that
 * can be down during a demo. If the feature pass is ever cached and the model
 * becomes the dominant cost, `slotify_rank.inference.predictor` is already
 * written to be constructed once and reused, so a persistent service is a new
 * entry point rather than a rewrite.
 *
 * Contract with the CLI: JSON on stdout, progress on stderr, non-zero exit on
 * failure. Nothing here repairs a malformed payload -- a ranking that cannot be
 * parsed is a failure, not something to substitute for.
 */

import path from "node:path";
import { clientSafeDetail } from "../lib/errors";
import { mlDir, mlPythonBin, REPO_ROOT } from "../lib/ranker-mode";
import { pythonEnv, runProcess, SubprocessError } from "../lib/subprocess";

export interface LearnedRankedCandidate {
  candidate_id: string;
  insertion_time_seconds: number;
  insertion_ms: number;
  raw_score: number;
  normalized_score: number;
  rank: number;
  acceptability_logit: number | null;
  audio_available: boolean;
  text_available: boolean;
  feature_status: string;
}

export interface LearnedRankingPayload {
  schema_version: string;
  episode_id: string;
  duration_seconds: number | null;
  score_scale: string;
  is_calibrated_probability: boolean;
  candidate_count: number;
  ranked: LearnedRankedCandidate[];
  excluded: Array<{ candidate_id: string; reason: string; detail: string }>;
  warnings: string[];
  timings_seconds: Record<string, number>;
  model: {
    model_variant: string;
    model_run_id: string;
    parameter_count: number;
    training_label_source: string;
    training_data_provenance: string;
    [key: string]: unknown;
  };
}

/** Schema versions this client knows how to read. */
const SUPPORTED_SCHEMA_VERSIONS = new Set(["inference-v1.0.0"]);

/** `message` is safe to show a client; `detail` (stderr) is for the server log. */
export class LearnedRankerError extends Error {
  constructor(
    message: string,
    readonly detail = "",
  ) {
    super(message);
    this.name = "LearnedRankerError";
  }
}

export interface RunLearnedRankerOptions {
  audioPath: string;
  checkpointPath: string;
  normalizerPath: string | null;
  episodeId: string;
  /** Skip Whisper transcription. Faster; every candidate is then text-masked. */
  skipTranscription?: boolean;
  timeoutMs?: number;
  /** Kills the ranker when aborted (the client went away). */
  signal?: AbortSignal;
}

const DEFAULT_TIMEOUT_MS = Number.parseInt(
  process.env.SLOTIFY_RANKER_TIMEOUT_MS ?? "600000",
  10,
);

export const runLearnedRanker = async ({
  audioPath,
  checkpointPath,
  normalizerPath,
  episodeId,
  skipTranscription = false,
  timeoutMs = DEFAULT_TIMEOUT_MS,
  signal,
}: RunLearnedRankerOptions): Promise<LearnedRankingPayload> => {
  const args = [
    "-m",
    "slotify_rank.cli",
    "infer",
    "rank",
    "--audio",
    audioPath,
    "--checkpoint",
    checkpointPath,
    "--episode-id",
    episodeId,
    "--output",
    "-",
  ];
  if (normalizerPath) args.push("--normalizer", normalizerPath);
  if (skipTranscription) args.push("--no-transcribe");

  let stdout: string;
  try {
    ({ stdout } = await runProcess(mlPythonBin(), args, {
      label: "the learned ranker",
      cwd: mlDir,
      env: pythonEnv({
        // The ml package is installed into ml/.venv in editable mode, but a
        // bare `python` on PATH would not see it; src/ on PYTHONPATH makes the
        // subprocess work either way.
        PYTHONPATH: [path.join(mlDir, "src"), process.env.PYTHONPATH]
          .filter(Boolean)
          .join(path.delimiter),
      }),
      timeoutMs,
      signal,
    }));
  } catch (error) {
    if (!(error instanceof SubprocessError)) throw error;
    if (error.reason === "timeout") {
      throw new LearnedRankerError(
        `The learned ranker did not finish within ${timeoutMs} ms. Transcription ` +
          "dominates the runtime on long audio; raise SLOTIFY_RANKER_TIMEOUT_MS " +
          "or use a shorter clip.",
        error.stderr,
      );
    }
    if (error.reason === "exit") {
      const last = clientSafeDetail(error.stderr);
      throw new LearnedRankerError(
        last ? `${error.message.replace(/\.$/, "")}: ${last}` : error.message,
        error.stderr,
      );
    }
    throw new LearnedRankerError(
      error.reason === "spawn"
        ? `Could not start the learned ranker (${path.basename(mlPythonBin())}).`
        : error.message,
      `${error.message}\n${error.stderr}`,
    );
  }
  return parseLearnedRanking(stdout);
};

/** Parse and validate the CLI's stdout. Exported so it can be tested directly. */
export const parseLearnedRanking = (stdout: string): LearnedRankingPayload => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout.trim() || "{}");
  } catch (error) {
    throw new LearnedRankerError(
      `The learned ranker produced output that is not JSON: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }

  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new LearnedRankerError("The learned ranker's output is not a JSON object.");
  }
  const payload = parsed as Partial<LearnedRankingPayload>;
  if (
    typeof payload.schema_version !== "string" ||
    !SUPPORTED_SCHEMA_VERSIONS.has(payload.schema_version)
  ) {
    throw new LearnedRankerError(
      `Unsupported inference schema_version ${JSON.stringify(payload.schema_version)}; ` +
        `this API reads ${[...SUPPORTED_SCHEMA_VERSIONS].join(", ")}.`,
    );
  }
  if (!Array.isArray(payload.ranked)) {
    throw new LearnedRankerError(
      "The learned ranker's output has no ranked list; refusing to report a ranking.",
    );
  }
  if (payload.is_calibrated_probability === true) {
    // Nothing in this repository is calibrated. A payload claiming otherwise is
    // either a bug or a different model, and either way must not reach the UI.
    throw new LearnedRankerError(
      "The learned ranker claims calibrated probabilities, which no model in this " +
        "repository has been calibrated to produce.",
    );
  }
  if (!Array.isArray(payload.warnings) || !Array.isArray(payload.excluded)) {
    throw new LearnedRankerError(
      "The learned ranker's output has no warnings or excluded list; refusing to report a ranking.",
    );
  }
  const model = payload.model;
  if (
    !model ||
    typeof model !== "object" ||
    typeof model.model_variant !== "string" ||
    typeof model.model_run_id !== "string" ||
    typeof model.training_label_source !== "string" ||
    !model.training_label_source
  ) {
    // The provenance a response reports -- above all what the model was trained
    // on -- comes from here. A ranking that cannot say is not reported.
    throw new LearnedRankerError(
      "The learned ranker's output does not identify the model and its training " +
        "label source; refusing to report a ranking.",
    );
  }
  return payload as LearnedRankingPayload;
};

/** Path relative to the repository root, for logging. */
export const relativeToRepo = (target: string): string =>
  path.relative(REPO_ROOT, target);
