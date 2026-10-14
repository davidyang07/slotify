import fs from "node:fs";
import path from "node:path";
import { BACKEND_DIR } from "../config";

/**
 * Thin TypeScript loader for the canonical, language-neutral heuristic
 * configuration at `config/heuristic_offline_v1.json`.
 *
 * The same file is loaded by `backend/ad_inserter/heuristic_config.py` and by
 * `ml/src/slotify_rank/config/settings.py`, so the baseline constants exist in
 * exactly one place. Values are read once at module load; a missing or
 * malformed file is a hard failure rather than a silent default, because a
 * wrong baseline would silently invalidate every experiment that compares
 * against it.
 */

export const HEURISTIC_CONFIG_PATH = path.resolve(
  BACKEND_DIR,
  "..",
  "config",
  "heuristic_offline_v1.json",
);

export interface HeuristicScoringConfig {
  base_score: number;
  pause_reward_max: number;
  pause_reward_saturation_ms: number;
  song_mode_bonus: number;
  sentence_end_reward: number;
  no_sentence_end_penalty: number;
  unavailable_snippet_sentinel: string;
  mid_episode_reward: number;
  mid_episode_min_ratio: number;
  mid_episode_max_ratio: number;
  edge_penalty: number;
  edge_guard_seconds: number;
  score_min: number;
  score_max: number;
}

export interface HeuristicSelectionConfig {
  min_separation_seconds: number;
  requested_count: number;
  max_returned: number;
  /** Fixture-only; see lib/baseline-parity.ts. The product never pads. */
  ratio_fallback_positions: number[];
  ratio_fallback_score: number;
  spacing_fallback_score: number;
}

export interface HeuristicProfile {
  description: string;
  is_canonical_baseline: boolean;
  silence_detection: {
    min_silence_len_ms: number;
    silence_threshold_offset_db: number;
    fallback_silence_threshold_dbfs: number;
    max_candidates: number;
    snippet_count: number;
    /** How the analyser picks `max_candidates` silences; documentation only. */
    candidate_selection: string;
  };
  merge: { min_gap_ms: number };
  scoring: HeuristicScoringConfig;
  selection: HeuristicSelectionConfig;
  route_fallback_candidates: {
    ratio_positions: number[];
    fixed_positions_ms: number[];
  };
  /**
   * The frozen `heuristic_offline_v1` slot finalisation. `end_guard_ms` is
   * still used by the product; the `confidence_*` mapping is NOT -- the product
   * reports a `placement_score` instead (see lib/placement.ts) and these keys
   * survive only so backend/scripts/dump-heuristic-golden.ts can keep
   * reproducing the frozen baseline record that ml/ evaluates against.
   */
  slot_finalisation: {
    end_guard_ms: number;
    confidence_base: number;
    confidence_scale: number;
    confidence_min: number;
    confidence_max: number;
    time_seconds_decimals: number;
  };
  sentence_boundary_pattern: { regex: string };
}

interface HeuristicConfigFile {
  config_version: string;
  canonical_profile: string;
  profiles: Record<string, HeuristicProfile>;
}

type Shape = "number" | "integer" | "string" | "boolean" | "number[]" | "regex";

/**
 * Every key the TypeScript and Python loaders read from the canonical profile,
 * with its shape. A key missing here used to surface as NaN scores rather than
 * as an error.
 */
const CANONICAL_SCHEMA: Record<string, Record<string, Shape>> = {
  silence_detection: {
    min_silence_len_ms: "number",
    silence_threshold_offset_db: "number",
    fallback_silence_threshold_dbfs: "number",
    max_candidates: "integer",
    snippet_count: "integer",
    candidate_selection: "string",
  },
  merge: { min_gap_ms: "number" },
  scoring: {
    base_score: "number",
    pause_reward_max: "number",
    pause_reward_saturation_ms: "number",
    song_mode_bonus: "number",
    sentence_end_reward: "number",
    no_sentence_end_penalty: "number",
    unavailable_snippet_sentinel: "string",
    mid_episode_reward: "number",
    mid_episode_min_ratio: "number",
    mid_episode_max_ratio: "number",
    edge_penalty: "number",
    edge_guard_seconds: "number",
    score_min: "number",
    score_max: "number",
  },
  selection: {
    min_separation_seconds: "number",
    requested_count: "integer",
    max_returned: "integer",
    ratio_fallback_positions: "number[]",
    ratio_fallback_score: "number",
    spacing_fallback_score: "number",
  },
  route_fallback_candidates: {
    ratio_positions: "number[]",
    fixed_positions_ms: "number[]",
  },
  slot_finalisation: {
    end_guard_ms: "number",
    confidence_base: "number",
    confidence_scale: "number",
    confidence_min: "number",
    confidence_max: "number",
    time_seconds_decimals: "integer",
  },
  sentence_boundary_pattern: { regex: "regex" },
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

const matchesShape = (value: unknown, shape: Shape): boolean => {
  switch (shape) {
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "integer":
      return Number.isInteger(value) && (value as number) >= 0;
    case "string":
      return typeof value === "string" && value.length > 0;
    case "boolean":
      return typeof value === "boolean";
    case "number[]":
      return (
        Array.isArray(value) &&
        value.every((entry) => typeof entry === "number" && Number.isFinite(entry))
      );
    case "regex":
      if (typeof value !== "string") return false;
      try {
        new RegExp(value);
        return true;
      } catch {
        return false;
      }
  }
};

const SHAPE_NAMES: Record<Shape, string> = {
  number: "a finite number",
  integer: "a non-negative integer",
  string: "a non-empty string",
  boolean: "a boolean",
  "number[]": "an array of finite numbers",
  regex: "a valid regular expression string",
};

/**
 * Validate a parsed config file. Throws one error naming every bad key, so a
 * broken edit is fixed in one pass. Exported for tests.
 */
export const validateHeuristicConfig = (
  parsed: unknown,
  where = HEURISTIC_CONFIG_PATH,
): HeuristicConfigFile => {
  if (!isRecord(parsed)) {
    throw new Error(`Heuristic config ${where} is not a JSON object.`);
  }
  const problems: string[] = [];
  for (const key of ["config_version", "canonical_profile"]) {
    if (!matchesShape(parsed[key], "string")) problems.push(`${key} must be a non-empty string`);
  }
  const profiles = parsed.profiles;
  const canonicalName = parsed.canonical_profile;
  const canonical =
    isRecord(profiles) && typeof canonicalName === "string" ? profiles[canonicalName] : undefined;
  if (!isRecord(canonical)) {
    throw new Error(
      `Heuristic config ${where} declares canonical_profile ` +
        `"${String(canonicalName)}" but no such profile exists.`,
    );
  }
  if (!matchesShape(canonical.is_canonical_baseline, "boolean")) {
    problems.push(`${canonicalName}.is_canonical_baseline must be a boolean`);
  }
  for (const [section, keys] of Object.entries(CANONICAL_SCHEMA)) {
    const block = canonical[section];
    if (!isRecord(block)) {
      problems.push(`${canonicalName}.${section} must be an object`);
      continue;
    }
    for (const [key, shape] of Object.entries(keys)) {
      if (!matchesShape(block[key], shape)) {
        problems.push(`${canonicalName}.${section}.${key} must be ${SHAPE_NAMES[shape]}`);
      }
    }
  }
  if (problems.length) {
    throw new Error(`Heuristic config ${where} is invalid: ${problems.join("; ")}.`);
  }
  return parsed as unknown as HeuristicConfigFile;
};

const loadConfigFile = (): HeuristicConfigFile => {
  let raw: string;
  try {
    raw = fs.readFileSync(HEURISTIC_CONFIG_PATH, "utf8");
  } catch (error) {
    throw new Error(
      `Unable to read the canonical heuristic config at ${HEURISTIC_CONFIG_PATH}: ` +
        `${error instanceof Error ? error.message : String(error)}`,
    );
  }
  return validateHeuristicConfig(JSON.parse(raw));
};

const configFile = loadConfigFile();

export const HEURISTIC_CONFIG_VERSION = configFile.config_version;

/** Name of the canonical baseline, e.g. "heuristic_offline_v1". */
export const BASELINE_VERSION = configFile.canonical_profile;

/** The canonical offline baseline profile used by the product path. */
export const HEURISTIC_V1: HeuristicProfile =
  configFile.profiles[configFile.canonical_profile];
