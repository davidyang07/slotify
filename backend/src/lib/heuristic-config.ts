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
}

interface HeuristicConfigFile {
  config_version: string;
  canonical_profile: string;
  profiles: Record<string, HeuristicProfile>;
}

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

  const parsed = JSON.parse(raw) as HeuristicConfigFile;
  const canonical = parsed.profiles?.[parsed.canonical_profile];
  if (!canonical) {
    throw new Error(
      `Heuristic config ${HEURISTIC_CONFIG_PATH} declares canonical_profile ` +
        `"${parsed.canonical_profile}" but no such profile exists.`,
    );
  }
  return parsed;
};

const configFile = loadConfigFile();

export const HEURISTIC_CONFIG_VERSION = configFile.config_version;

/** Name of the canonical baseline, e.g. "heuristic_offline_v1". */
export const BASELINE_VERSION = configFile.canonical_profile;

/** The canonical offline baseline profile used by the product path. */
export const HEURISTIC_V1: HeuristicProfile =
  configFile.profiles[configFile.canonical_profile];
