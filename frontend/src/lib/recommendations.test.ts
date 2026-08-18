import assert from "node:assert/strict";
import { test, describe } from "node:test";

import {
  clampSlotsToDuration,
  describeScoreScale,
  describeSource,
  emptyStateMessage,
  makeManualSlot,
  parsePlacementResponse,
} from "./recommendations";

const apiSlot = (index: number, time: number, score: number) => ({
  candidate_id: `upload:${String(Math.round(time * 1000)).padStart(9, "0")}`,
  insertion_ms: Math.round(time * 1000),
  insertion_time_seconds: time,
  placement_score: score,
  raw_score: score / 100,
  rank: index + 1,
  source: "heuristic_offline_v1" as const,
  signals: [{ kind: "supporting" as const, label: "Pause of 1.8s detected" }],
  rationale: "Ranked at 60.0s: pause of 1.8s detected.",
  rationale_source: "measured_signals" as const,
  silence_ms: 1800,
  snippet: "So that was it.",
});

describe("the UI shows exactly what the server returned", () => {
  test("zero recommendations stay zero", () => {
    const result = parsePlacementResponse({
      placementStatus: "no_candidates",
      slots: [],
      duration: 600,
      candidateCount: 0,
      warning: "Analysis completed but found no eligible insertion point.",
    });
    assert.equal(result.status, "no_candidates");
    assert.deepEqual(result.slots, []);
  });

  test("one recommendation stays one", () => {
    const result = parsePlacementResponse({
      placementStatus: "ok",
      slots: [apiSlot(0, 60, 88)],
      duration: 600,
      candidateCount: 4,
    });
    assert.equal(result.slots.length, 1);
    assert.equal(result.slots[0].time, 60);
    assert.equal(result.slots[0].placementScore, 88);
  });

  test("two recommendations stay two", () => {
    const result = parsePlacementResponse({
      placementStatus: "ok",
      slots: [apiSlot(0, 60, 88), apiSlot(1, 180, 62)],
      duration: 600,
      candidateCount: 9,
    });
    assert.equal(result.slots.length, 2);
  });

  test("no ratio-positioned fallback slot is ever produced", () => {
    const duration = 600;
    const result = parsePlacementResponse({
      placementStatus: "ok",
      slots: [apiSlot(0, 60, 88)],
      duration,
    });
    // The removed behaviour placed slots at 22 %, 48 % and 72 % of duration.
    const forbidden = [0.22, 0.48, 0.72].map((ratio) => ratio * duration);
    for (const slot of result.slots) {
      assert.ok(
        !forbidden.includes(slot.time),
        `slot at ${slot.time}s matches a removed fallback position`,
      );
    }
    assert.equal(result.slots.length, 1);
  });

  test("no hard-coded 92/85/78 score can appear from an empty response", () => {
    const result = parsePlacementResponse({ placementStatus: "ok", slots: [] });
    assert.deepEqual(
      result.slots.map((slot) => slot.placementScore),
      [],
    );
  });

  test("a slot with an unusable timestamp is dropped, not defaulted to zero", () => {
    const result = parsePlacementResponse({
      placementStatus: "ok",
      slots: [{ ...apiSlot(0, 60, 88), insertion_time_seconds: "not a number" }],
    });
    assert.deepEqual(result.slots, []);
  });

  test("a malformed body degrades to unavailable rather than throwing", () => {
    const result = parsePlacementResponse(null);
    assert.equal(result.status, "unavailable");
    assert.deepEqual(result.slots, []);
  });
});

describe("empty states say which of the three things happened", () => {
  test("analyser failure is distinguished from an empty result", () => {
    const failed = parsePlacementResponse({
      placementStatus: "unavailable",
      slots: [],
      error: "Audio analysis failed; no insertion points can be reported.",
    });
    assert.match(emptyStateMessage(failed), /Audio analysis failed/);

    const empty = parsePlacementResponse({
      placementStatus: "no_candidates",
      slots: [],
    });
    assert.match(emptyStateMessage(empty), /No reliable insertion point/);
  });

  test("before any analysis the prompt is to run one", () => {
    assert.match(emptyStateMessage(null), /Upload audio and run analysis/);
  });
});

describe("provenance is rendered accurately", () => {
  test("the heuristic is never described as a model", () => {
    const label = describeSource({
      candidateGeneration: "signal_offline_v1",
      source: "heuristic_offline_v1",
      mode: "heuristic",
      modelVariant: null,
      modelRunId: null,
      scoreScale: "absolute_unit_interval",
      isCalibratedProbability: false,
    });
    assert.equal(label, "Heuristic baseline");
  });

  test("the learned ranker names its variant", () => {
    const label = describeSource({
      candidateGeneration: "signal_offline_v1",
      source: "learned_ranker",
      mode: "learned",
      modelVariant: "gated",
      modelRunId: "gated-abc123",
      scoreScale: "episode_relative_min_max",
      isCalibratedProbability: false,
    });
    assert.equal(label, "Learned ranker (gated)");
  });

  test("the score scale is always spelled out as not a probability", () => {
    const note = describeScoreScale({
      candidateGeneration: "signal_offline_v1",
      source: "learned_ranker",
      mode: "learned",
      modelVariant: "gated",
      modelRunId: "gated-abc123",
      scoreScale: "episode_relative_min_max",
      isCalibratedProbability: false,
    });
    assert.match(note, /Not a probability/);
  });
});

describe("manual slots and clamping", () => {
  test("a manual slot carries no score and is tagged manual", () => {
    const slot = makeManualSlot(42, []);
    assert.equal(slot.source, "manual");
    assert.equal(slot.placementScore, null);
    assert.equal(slot.rank, null);
  });

  test("manual slot ids do not collide with existing ones", () => {
    const existing = [makeManualSlot(10, [])];
    const next = makeManualSlot(20, existing);
    assert.notEqual(next.id, existing[0].id);
  });

  test("clamping moves times into range without adding or dropping slots", () => {
    const result = parsePlacementResponse({
      placementStatus: "ok",
      slots: [apiSlot(0, 700, 88)],
    });
    const clamped = clampSlotsToDuration(result.slots, 600);
    assert.equal(clamped.length, 1);
    assert.equal(clamped[0].time, 600);
  });
});
