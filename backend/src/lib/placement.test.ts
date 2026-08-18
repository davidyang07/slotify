import assert from "node:assert/strict";
import { test, describe } from "node:test";

import {
  buildPlacementSignals,
  buildRationale,
  clampSlotMs,
  dedupeByMs,
  toPlacementScores,
} from "./placement";
import { makeCandidateId, rankWithHeuristic } from "./ranking";

describe("placement signals are measured, never decorative", () => {
  test("a silent, mid-episode sentence boundary reports exactly what was found", () => {
    const signals = buildPlacementSignals({
      mode: "podcast",
      silenceMs: 1800,
      snippet: "So that was the whole story.",
      timeSeconds: 300,
      durationSeconds: 600,
    });
    assert.deepEqual(signals, [
      { kind: "supporting", label: "Pause of 1.8s detected" },
      { kind: "supporting", label: "Falls on a sentence boundary" },
      { kind: "supporting", label: "Mid-episode (50% through)" },
    ]);
  });

  test("no transcript is reported as a caution, not silently omitted", () => {
    const signals = buildPlacementSignals({
      mode: "podcast",
      silenceMs: 0,
      snippet: "",
      timeSeconds: 60,
      durationSeconds: 600,
    });
    assert.ok(
      signals.some(
        (signal) =>
          signal.kind === "caution" &&
          signal.label === "No transcript context available here",
      ),
    );
  });

  test("nothing is padded to a fixed count", () => {
    const sparse = buildPlacementSignals({
      mode: "podcast",
      silenceMs: 0,
      snippet: "",
      timeSeconds: 1,
      durationSeconds: null,
    });
    // One caution (no transcript) and nothing else: no duration means no
    // position signal, and there is no filler to make up a third bullet.
    assert.equal(sparse.length, 1);
  });

  test("a mid-sentence cut is a caution", () => {
    const signals = buildPlacementSignals({
      mode: "podcast",
      silenceMs: 900,
      snippet: "and then we were going to",
      timeSeconds: 300,
      durationSeconds: 600,
    });
    assert.ok(
      signals.some(
        (signal) => signal.kind === "caution" && signal.label === "Cuts mid-sentence",
      ),
    );
  });

  test("edges are called out on both sides", () => {
    const early = buildPlacementSignals({
      mode: "podcast",
      silenceMs: 800,
      snippet: "Welcome back.",
      timeSeconds: 2,
      durationSeconds: 60,
    });
    const late = buildPlacementSignals({
      mode: "podcast",
      silenceMs: 800,
      snippet: "Thanks for listening.",
      timeSeconds: 58,
      durationSeconds: 60,
    });
    assert.ok(early.some((signal) => signal.label === "Very close to the start"));
    assert.ok(late.some((signal) => signal.label === "Very close to the end"));
  });
});

describe("rationale never claims more than the signals", () => {
  test("with no supporting signal it says so", () => {
    const rationale = buildRationale(
      [{ kind: "caution", label: "Cuts mid-sentence" }],
      42,
    );
    assert.match(rationale, /no supporting signal was measured/);
  });

  test("with supporting signals it lists them", () => {
    const rationale = buildRationale(
      [{ kind: "supporting", label: "Pause of 1.8s detected" }],
      42,
    );
    assert.equal(rationale, "Ranked at 42.0s: pause of 1.8s detected.");
  });
});

describe("placement scores are ranking scores, not confidences", () => {
  test("a unit-interval score maps linearly onto 0-100, not onto 70-95", () => {
    assert.deepEqual(
      toPlacementScores([0, 0.5, 1], "absolute_unit_interval"),
      [0, 50, 100],
    );
  });

  test("unbounded scores are normalized within the episode only", () => {
    assert.deepEqual(
      toPlacementScores([-2, 0, 2], "episode_relative_min_max"),
      [0, 50, 100],
    );
  });

  test("identical scores do not get an invented spread", () => {
    assert.deepEqual(
      toPlacementScores([1.5, 1.5, 1.5], "episode_relative_min_max"),
      [50, 50, 50],
    );
  });

  test("an empty ranking produces no scores", () => {
    assert.deepEqual(toPlacementScores([], "episode_relative_min_max"), []);
  });
});

describe("slot finalisation", () => {
  test("a slot is clamped inside the audio with the end guard applied", () => {
    assert.equal(clampSlotMs(120_000, 60), 59_800);
    assert.equal(clampSlotMs(-5, 60), 0);
    assert.equal(clampSlotMs(12_345, null), 12_345);
  });

  test("duplicate timestamps collapse to the highest-ranked one", () => {
    const deduped = dedupeByMs([
      { ms: 100, tag: "first" },
      { ms: 100, tag: "second" },
      { ms: 200, tag: "third" },
    ]);
    assert.deepEqual(deduped, [
      { ms: 100, tag: "first" },
      { ms: 200, tag: "third" },
    ]);
  });
});

describe("heuristic ranking carries its own identity", () => {
  test("candidate ids match the ml package's deterministic format", () => {
    assert.equal(makeCandidateId("upload", 60_000), "upload:000060000");
  });

  test("the result names the heuristic, not a model", () => {
    const result = rankWithHeuristic({
      candidates: [{ ms: 60_000, silenceMs: 1800, snippet: "Right." }],
      episodeId: "upload",
      durationSeconds: 600,
      mode: "podcast",
      minSeparationSeconds: 6,
      count: 3,
    });
    assert.equal(result.source, "heuristic_offline_v1");
    assert.equal(result.mode, "heuristic");
    assert.equal(result.modelVariant, null);
    assert.equal(result.scoreScale, "absolute_unit_interval");
    assert.equal(result.ranked.length, 1);
  });

  test("an empty candidate set ranks nothing", () => {
    const result = rankWithHeuristic({
      candidates: [],
      episodeId: "upload",
      durationSeconds: 600,
      mode: "podcast",
      minSeparationSeconds: 6,
      count: 3,
    });
    assert.deepEqual(result.ranked, []);
  });
});
