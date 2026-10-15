import assert from "node:assert/strict";
import { afterEach, describe, test } from "node:test";

import { ClientDisconnectedError, HttpError } from "../lib/http";
import {
  buildSponsorBlock,
  checkTtsBudget,
  estimateSpeechSeconds,
  setElevenLabsClientForTests,
} from "./elevenlabs";

const isBadRequest = (pattern: RegExp) => (error: unknown) =>
  error instanceof HttpError && error.status === 400 && pattern.test(error.message);

/** A fake SDK that records each convert call and its abort signal. */
const recordingClient = (onConvert: (signal: AbortSignal | undefined) => Promise<void> = async () => undefined) => {
  const calls: Array<{ text: string; signal: AbortSignal | undefined }> = [];
  return {
    calls,
    client: {
      textToSpeech: {
        convert: async (_voiceId: string, request: { text: string }, options?: { abortSignal?: AbortSignal }) => {
          calls.push({ text: request.text, signal: options?.abortSignal });
          await onConvert(options?.abortSignal);
          throw new Error("fake provider stops here");
        },
      },
    },
  };
};

afterEach(() => setElevenLabsClientForTests(null));

describe("TTS spend is bounded before the first paid call", () => {
  test("too many statements is a 400", () => {
    const statements = Array.from({ length: 400 }, (_, index) => `Line ${index}.`);
    assert.throws(() => checkTtsBudget(statements), isBadRequest(/At most 8 statements/));
    assert.doesNotThrow(() => checkTtsBudget(["One.", "Two."]));
  });

  test("too much text in total is a 400, even in one statement", () => {
    assert.throws(() => checkTtsBudget(["x".repeat(1_501)]), isBadRequest(/At most 1500 characters/));
    assert.doesNotThrow(() => checkTtsBudget(["x".repeat(1_500)]));
  });

  test("a block that cannot fit in 20 s is refused on its estimate", () => {
    const long = Array.from({ length: 4 }, () => "y".repeat(120));
    assert.throws(() => checkTtsBudget(long), isBadRequest(/at least \d+ s to read/));
    // A normal two-line sponsor read is well within it.
    assert.doesNotThrow(() =>
      checkTtsBudget(["This episode is brought to you by Acme.", "Visit acme.example today."]),
    );
  });

  test("the estimate is a lower bound that weighs dense scripts and pauses", () => {
    assert.equal(estimateSpeechSeconds(["a".repeat(200)], 0), 10);
    assert.ok(estimateSpeechSeconds(["本".repeat(100)], 0) > estimateSpeechSeconds(["a".repeat(100)], 0));
    assert.equal(estimateSpeechSeconds(["a".repeat(20), "b".repeat(20)], 1_000), 3);
  });

  test("buildSponsorBlock enforces the same budget", async () => {
    const { calls, client } = recordingClient();
    setElevenLabsClientForTests(client);
    await assert.rejects(
      buildSponsorBlock({ voiceId: "v", statements: Array.from({ length: 9 }, () => "Hi.") }),
      isBadRequest(/At most 8/),
    );
    assert.equal(calls.length, 0);
  });
});

describe("TTS stops when the client leaves", () => {
  test("no call is made for a client that has already gone", async () => {
    const { calls, client } = recordingClient();
    setElevenLabsClientForTests(client);
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      buildSponsorBlock({ voiceId: "v", statements: ["One.", "Two."], signal: controller.signal }),
      (error: unknown) => error instanceof ClientDisconnectedError,
    );
    assert.equal(calls.length, 0);
  });

  test("the abort signal reaches the SDK, and no later statement is synthesized", async () => {
    const controller = new AbortController();
    const { calls, client } = recordingClient(async () => {
      controller.abort();
    });
    setElevenLabsClientForTests(client);
    await assert.rejects(
      buildSponsorBlock({ voiceId: "v", statements: ["One.", "Two.", "Three."], signal: controller.signal }),
    );
    assert.equal(calls.length, 1);
    assert.equal(calls[0].signal, controller.signal);
  });
});
