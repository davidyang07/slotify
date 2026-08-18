import assert from "node:assert/strict";
import { test, describe } from "node:test";

import { mergeCandidates, scoreCandidate, selectTopSlots } from "./candidates";
import { padSelectionForBaselineParity } from "./baseline-parity";
import type { ScoredCandidate } from "../types";

const scored = (ms: number, score: number): ScoredCandidate => ({
  ms,
  silenceMs: 1000,
  snippet: "A sentence.",
  score,
});

describe("selectTopSlots never invents a slot", () => {
  test("no candidates yields no results", () => {
    assert.deepEqual(selectTopSlots([], 6, 3), []);
  });

  test("one candidate yields exactly one result", () => {
    const result = selectTopSlots([scored(60_000, 0.9)], 6, 3);
    assert.equal(result.length, 1);
    assert.equal(result[0].ms, 60_000);
  });

  test("two candidates yield exactly two results", () => {
    const result = selectTopSlots([scored(60_000, 0.9), scored(120_000, 0.8)], 6, 3);
    assert.equal(result.length, 2);
    assert.deepEqual(
      result.map((entry) => entry.ms),
      [60_000, 120_000],
    );
  });

  test("three candidates yield three results in score order", () => {
    const result = selectTopSlots(
      [scored(60_000, 0.5), scored(120_000, 0.9), scored(180_000, 0.7)],
      6,
      3,
    );
    assert.deepEqual(
      result.map((entry) => entry.ms),
      [120_000, 180_000, 60_000],
    );
  });

  test("requesting more than exist does not pad", () => {
    const result = selectTopSlots([scored(60_000, 0.9)], 6, 10);
    assert.equal(result.length, 1);
  });

  test("separation filtering may legitimately reduce the count", () => {
    // Four strong candidates inside a single 6 s window: only one survives.
    const result = selectTopSlots(
      [
        scored(100_000, 0.9),
        scored(101_000, 0.9),
        scored(103_000, 0.9),
        scored(105_500, 0.9),
      ],
      6,
      3,
    );
    assert.equal(result.length, 1);
    assert.equal(result[0].ms, 100_000);
  });

  test("every returned slot is one of the inputs", () => {
    const inputs = [scored(30_000, 0.4), scored(90_000, 0.8), scored(150_000, 0.6)];
    const result = selectTopSlots(inputs, 6, 3);
    for (const entry of result) {
      assert.ok(
        inputs.some(
          (input) =>
            input.ms === entry.ms &&
            input.score === entry.score &&
            input.snippet === entry.snippet,
        ),
        `slot at ${entry.ms}ms is not one of the supplied candidates`,
      );
    }
  });

  test("a known duration cannot conjure ratio-positioned slots", () => {
    // The removed behaviour keyed off duration; the signature no longer accepts
    // one, and this asserts the outcome rather than the signature.
    const result = selectTopSlots([], 6, 3);
    assert.equal(result.length, 0);
  });
});

describe("padSelectionForBaselineParity is fixture-only and marks its output", () => {
  test("real entries keep the candidate origin", () => {
    const padded = padSelectionForBaselineParity([scored(88_000, 0.9)], 200, 6, 3);
    assert.equal(padded[0].origin, "candidate");
    assert.equal(padded[0].ms, 88_000);
  });

  test("padding is tagged synthetic and only appears here", () => {
    const padded = padSelectionForBaselineParity([scored(88_000, 0.9)], 200, 6, 3);
    assert.equal(padded.length, 3);
    const origins = padded.map((entry) => entry.origin);
    assert.deepEqual(origins, ["candidate", "ratio_fallback", "ratio_fallback"]);
  });

  test("without a duration it falls through to spacing padding", () => {
    const padded = padSelectionForBaselineParity([], null, 6, 3);
    assert.deepEqual(
      padded.map((entry) => entry.origin),
      ["spacing_fallback", "spacing_fallback", "spacing_fallback"],
    );
  });
});

describe("scoring and merging are unchanged", () => {
  test("a long pause at a sentence boundary scores higher than a mid-sentence cut", () => {
    const boundary = scoreCandidate(
      { ms: 300_000, silenceMs: 2400, snippet: "We'll come back to that." },
      600,
      "podcast",
    );
    const midSentence = scoreCandidate(
      { ms: 300_000, silenceMs: 300, snippet: "and then we were going to" },
      600,
      "podcast",
    );
    assert.ok(boundary > midSentence);
  });

  test("near-duplicates merge and the longer pause wins", () => {
    const merged = mergeCandidates(
      [{ ms: 60_000, silenceMs: 900, snippet: "So that was it." }],
      [{ ms: 60_200, silenceMs: 2500, snippet: "So that was it." }],
    );
    assert.equal(merged.length, 1);
    assert.equal(merged[0].silenceMs, 2500);
  });
});
