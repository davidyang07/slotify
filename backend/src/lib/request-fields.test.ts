import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { HttpError } from "./http";
import {
  optionalBoolean,
  optionalEnum,
  optionalNumber,
  optionalObject,
  optionalStatements,
  optionalString,
} from "./request-fields";
import { asText, endsWithSentenceBoundary, normalizeStatements } from "./text";
import { HEURISTIC_V1 } from "./heuristic-config";

const isBadRequest = (error: unknown) => error instanceof HttpError && error.status === 400;
const HOSTILE = { toString: 1, valueOf: 1 };

describe("typed request fields", () => {
  test("strings are trimmed; blank and absent are undefined", () => {
    assert.equal(optionalString({ a: "  x " }, "a"), "x");
    assert.equal(optionalString({ a: "   " }, "a"), undefined);
    assert.equal(optionalString({}, "a"), undefined);
    assert.equal(optionalString(null, "a"), undefined);
  });

  test("any other shape is a 400, including multer's bracket objects and repeats", () => {
    for (const value of [HOSTILE, ["a", "b"], 5, true]) {
      assert.throws(() => optionalString({ a: value }, "a"), isBadRequest);
    }
    assert.throws(() => optionalString({ a: "x".repeat(11) }, "a", 10), isBadRequest);
  });

  test("inherited properties are not fields", () => {
    assert.equal(optionalString({}, "toString"), undefined);
    assert.equal(optionalString({}, "constructor"), undefined);
  });

  test("numbers come from JSON numbers or numeric strings only", () => {
    assert.equal(optionalNumber({ n: 1.5 }, "n"), 1.5);
    assert.equal(optionalNumber({ n: " 2.25 " }, "n"), 2.25);
    assert.equal(optionalNumber({ n: "" }, "n"), undefined);
    for (const value of ["abc", "5abc", HOSTILE, Number.NaN, Number.POSITIVE_INFINITY, [1]]) {
      assert.throws(() => optionalNumber({ n: value }, "n"), isBadRequest);
    }
  });

  test("enums are case-insensitive and closed", () => {
    assert.equal(optionalEnum({ m: "Song" }, "m", ["podcast", "song"] as const), "song");
    assert.throws(() => optionalEnum({ m: "radio" }, "m", ["podcast", "song"] as const), isBadRequest);
  });

  test("booleans accept JSON booleans and the usual form strings", () => {
    assert.equal(optionalBoolean({ b: "TRUE" }, "b"), true);
    assert.equal(optionalBoolean({ b: "0" }, "b"), false);
    assert.equal(optionalBoolean({ b: false }, "b"), false);
    assert.throws(() => optionalBoolean({ b: "maybe" }, "b"), isBadRequest);
  });

  test("statements: string, JSON-array string or string array", () => {
    assert.deepEqual(optionalStatements({ s: "One." }, "s"), ["One."]);
    assert.deepEqual(optionalStatements({ s: '["One.", " ", "Two."]' }, "s"), ["One.", "Two."]);
    assert.deepEqual(optionalStatements({ s: ["One.", ""] }, "s"), ["One."]);
    assert.throws(() => optionalStatements({ s: [HOSTILE] }, "s"), isBadRequest);
    assert.throws(() => optionalStatements({ s: 5 }, "s"), isBadRequest);
  });

  test("objects must be objects", () => {
    assert.deepEqual(optionalObject({ o: { a: 1 } }, "o"), { a: 1 });
    assert.throws(() => optionalObject({ o: "x" }, "o"), isBadRequest);
    assert.throws(() => optionalObject({ o: [] }, "o"), isBadRequest);
  });
});

describe("text helpers never call a hostile toString", () => {
  test("asText keeps strings and finite numbers only", () => {
    assert.equal(asText("a"), "a");
    assert.equal(asText(3), "3");
    assert.equal(asText(Number.NaN), "");
    assert.equal(asText(HOSTILE), "");
    assert.equal(asText(Symbol("s")), "");
  });

  test("normalizeStatements ignores other shapes", () => {
    assert.deepEqual(normalizeStatements(["a", HOSTILE, null, 2]), ["a", "2"]);
    assert.deepEqual(normalizeStatements('[{"toString": 1}, "b"]'), ["b"]);
    assert.deepEqual(normalizeStatements(HOSTILE), []);
  });

  test("endsWithSentenceBoundary uses the canonical config pattern", () => {
    assert.equal(
      HEURISTIC_V1.sentence_boundary_pattern.regex,
      "[.!?][\"')\\]]?\\s*$",
    );
    assert.equal(endsWithSentenceBoundary("Done."), true);
    assert.equal(endsWithSentenceBoundary('He said "go!"'), true);
    assert.equal(endsWithSentenceBoundary("and then"), false);
    assert.equal(endsWithSentenceBoundary(HOSTILE), false);
  });
});
