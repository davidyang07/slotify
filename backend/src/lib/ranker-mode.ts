/**
 * Which ranker the product should use, and whether it can.
 *
 * Three modes, set with `RANKER_MODE`:
 *
 *   heuristic  always use the credential-free offline baseline
 *   learned    require the PyTorch ranker; fail loudly if it is unusable
 *   auto       use the PyTorch ranker when a checkpoint is configured and
 *              usable, otherwise the heuristic (default)
 *
 * `auto` is the default and it resolves to `heuristic` on a fresh clone,
 * because a fresh clone has no trained checkpoint. That is the honest
 * behaviour: the alternative -- quietly labelling heuristic output as model
 * output so the demo looks better -- is the exact failure this file exists to
 * prevent. Whatever it resolves to is reported in every response.
 */

import fs from "node:fs";
import path from "node:path";
import { BACKEND_DIR } from "../config";
import type { RankerMode } from "../types";

export const REPO_ROOT = path.resolve(BACKEND_DIR, "..");

const isWindows = process.platform === "win32";

/** Directory containing the `slotify_rank` Python package. */
export const mlDir = process.env.SLOTIFY_ML_DIR
  ? path.resolve(process.env.SLOTIFY_ML_DIR)
  : path.join(REPO_ROOT, "ml");

/**
 * Interpreter used for the ML package. It is deliberately separate from
 * `PYTHON_BIN` (which runs `ad_inserter`): the ML extras are a ~1 GB install
 * that most contributors keep in `ml/.venv`, and the audio pipeline does not
 * need them.
 */
export const mlPythonBin = (): string => {
  if (process.env.SLOTIFY_ML_PYTHON) return process.env.SLOTIFY_ML_PYTHON;
  const venv = isWindows
    ? path.join(mlDir, ".venv", "Scripts", "python.exe")
    : path.join(mlDir, ".venv", "bin", "python");
  if (fs.existsSync(venv)) return venv;
  return process.env.PYTHON_BIN ?? "python";
};

const requestedMode = (): RankerMode => {
  const raw = String(process.env.RANKER_MODE ?? "auto").trim().toLowerCase();
  if (raw === "learned" || raw === "heuristic" || raw === "auto") return raw;
  throw new Error(
    `RANKER_MODE must be one of learned | heuristic | auto, got ${JSON.stringify(raw)}`,
  );
};

export interface ResolvedRankerMode {
  /** What was asked for. */
  requested: RankerMode;
  /** What will actually run. Never "auto". */
  mode: "learned" | "heuristic";
  checkpointPath: string | null;
  normalizerPath: string | null;
  /** Why `mode` is what it is, in one sentence, for logs and /api/capabilities. */
  reason: string;
}

export const resolveRankerMode = (): ResolvedRankerMode => {
  const requested = requestedMode();
  const configured = process.env.SLOTIFY_RANKER_CHECKPOINT;
  const checkpointPath = configured ? path.resolve(REPO_ROOT, configured) : null;
  const normalizerPath = checkpointPath
    ? process.env.SLOTIFY_RANKER_NORMALIZER
      ? path.resolve(REPO_ROOT, process.env.SLOTIFY_RANKER_NORMALIZER)
      : path.join(path.dirname(checkpointPath), "normalizer.json")
    : null;

  if (requested === "heuristic") {
    return {
      requested,
      mode: "heuristic",
      checkpointPath: null,
      normalizerPath: null,
      reason: "RANKER_MODE=heuristic: the offline baseline was requested.",
    };
  }

  const present =
    checkpointPath !== null &&
    fs.existsSync(checkpointPath) &&
    normalizerPath !== null &&
    fs.existsSync(normalizerPath);

  if (present) {
    return {
      requested,
      mode: "learned",
      checkpointPath,
      normalizerPath,
      reason: `Learned ranker checkpoint configured at ${path.relative(REPO_ROOT, checkpointPath!)}.`,
    };
  }

  const missing = !checkpointPath
    ? "SLOTIFY_RANKER_CHECKPOINT is not set"
    : !fs.existsSync(checkpointPath)
      ? `checkpoint ${path.relative(REPO_ROOT, checkpointPath)} does not exist`
      : `normalizer ${path.relative(REPO_ROOT, normalizerPath!)} does not exist`;

  if (requested === "learned") {
    throw new Error(
      `RANKER_MODE=learned but the learned ranker is unusable: ${missing}. ` +
        "Set SLOTIFY_RANKER_CHECKPOINT, or use RANKER_MODE=auto to fall back " +
        "to the heuristic baseline.",
    );
  }

  return {
    requested,
    mode: "heuristic",
    checkpointPath: null,
    normalizerPath: null,
    reason: `RANKER_MODE=auto fell back to the heuristic baseline: ${missing}.`,
  };
};
