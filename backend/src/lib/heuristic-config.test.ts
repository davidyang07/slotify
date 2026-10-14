import assert from "node:assert/strict";
import fs from "node:fs";
import { describe, test } from "node:test";

import {
  HEURISTIC_CONFIG_PATH,
  HEURISTIC_CONFIG_VERSION,
  validateHeuristicConfig,
} from "./heuristic-config";

const canonical = () => JSON.parse(fs.readFileSync(HEURISTIC_CONFIG_PATH, "utf8"));
const profileOf = (config: any) => config.profiles[config.canonical_profile];

describe("heuristic config validation", () => {
  test("the committed config is valid and versioned", () => {
    assert.doesNotThrow(() => validateHeuristicConfig(canonical()));
    assert.equal(HEURISTIC_CONFIG_VERSION, "heuristic-config-v1.1.0");
  });

  test("the analyser limits the API passes are in the config", () => {
    const { silence_detection } = profileOf(canonical());
    assert.equal(silence_detection.snippet_count, 12);
    assert.equal(silence_detection.max_candidates, 30);
    assert.equal(silence_detection.candidate_selection, "longest_silence");
  });

  test("a missing scoring key is a load error naming it, not NaN scores", () => {
    const config = canonical();
    delete profileOf(config).scoring.base_score;
    assert.throws(() => validateHeuristicConfig(config, "test.json"), /scoring\.base_score must be a finite number/);
  });

  test("every bad key is reported at once", () => {
    const config = canonical();
    const profile = profileOf(config);
    profile.merge.min_gap_ms = "400";
    profile.selection.ratio_fallback_positions = [0.2, "x"];
    delete profile.slot_finalisation;
    assert.throws(
      () => validateHeuristicConfig(config, "test.json"),
      (error: Error) =>
        /merge\.min_gap_ms/.test(error.message) &&
        /selection\.ratio_fallback_positions must be an array of finite numbers/.test(error.message) &&
        /slot_finalisation must be an object/.test(error.message),
    );
  });

  test("an uncompilable sentence pattern is rejected", () => {
    const config = canonical();
    profileOf(config).sentence_boundary_pattern.regex = "[unclosed";
    assert.throws(() => validateHeuristicConfig(config, "test.json"), /sentence_boundary_pattern\.regex/);
  });

  test("candidate limits must be non-negative integers", () => {
    const config = canonical();
    profileOf(config).silence_detection.snippet_count = 2.5;
    assert.throws(() => validateHeuristicConfig(config, "test.json"), /snippet_count must be a non-negative integer/);
  });

  test("a missing canonical profile is still a clear error", () => {
    const config = canonical();
    config.canonical_profile = "nope";
    assert.throws(() => validateHeuristicConfig(config, "test.json"), /no such profile/);
  });
});
