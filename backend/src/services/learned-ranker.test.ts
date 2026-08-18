import assert from "node:assert/strict";
import { test, describe } from "node:test";

import { LearnedRankerError, parseLearnedRanking } from "./learned-ranker";

const validPayload = {
  schema_version: "inference-v1.0.0",
  episode_id: "upload",
  duration_seconds: 600,
  score_scale: "episode_relative_min_max",
  is_calibrated_probability: false,
  candidate_count: 2,
  ranked: [
    {
      candidate_id: "upload:000060000",
      insertion_time_seconds: 60,
      insertion_ms: 60_000,
      raw_score: 1.2,
      normalized_score: 100,
      rank: 1,
      acceptability_logit: 0.3,
      audio_available: true,
      text_available: true,
      feature_status: "complete",
    },
  ],
  excluded: [],
  warnings: [],
  timings_seconds: { score: 0.01 },
  model: {
    model_variant: "gated",
    model_run_id: "gated-abc",
    parameter_count: 489_477,
    training_label_source: "weak_heuristic",
    training_data_provenance: "weak_supervision",
  },
};

describe("parsing the learned ranker's output", () => {
  test("a well-formed payload is accepted", () => {
    const parsed = parseLearnedRanking(JSON.stringify(validPayload));
    assert.equal(parsed.ranked.length, 1);
    assert.equal(parsed.model.model_variant, "gated");
  });

  test("non-JSON output is a failure, not an empty ranking", () => {
    assert.throws(
      () => parseLearnedRanking("Traceback (most recent call last): ..."),
      LearnedRankerError,
    );
  });

  test("an unknown schema version is rejected", () => {
    const payload = { ...validPayload, schema_version: "inference-v9.0.0" };
    assert.throws(
      () => parseLearnedRanking(JSON.stringify(payload)),
      /Unsupported inference schema_version/,
    );
  });

  test("a payload with no ranked list is rejected rather than treated as empty", () => {
    const { ranked: _ranked, ...withoutRanked } = validPayload;
    assert.throws(
      () => parseLearnedRanking(JSON.stringify(withoutRanked)),
      /no ranked list/,
    );
  });

  test("a payload claiming calibrated probabilities is refused", () => {
    // Nothing here has been calibrated against human labels. A payload saying
    // otherwise must never reach a UI that would render it as a percentage.
    const payload = { ...validPayload, is_calibrated_probability: true };
    assert.throws(
      () => parseLearnedRanking(JSON.stringify(payload)),
      /claims calibrated probabilities/,
    );
  });

  test("an empty ranked list is a legitimate answer", () => {
    const payload = { ...validPayload, ranked: [], candidate_count: 0 };
    const parsed = parseLearnedRanking(JSON.stringify(payload));
    assert.deepEqual(parsed.ranked, []);
  });
});
