import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { ElevenLabsError } from "@elevenlabs/elevenlabs-js";
import {
  clientSafeDetail,
  describeFailure,
  failureBody,
  logFailure,
  PROVIDER_MESSAGES,
  ProviderError,
} from "./errors";
import { HttpError } from "./http";
import { SubprocessError } from "./subprocess";

const TRACEBACK = [
  "Traceback (most recent call last):",
  '  File "/home/user/slotify/backend/ad_inserter/analyze_cli.py", line 81, in run',
  "    audio = analysis.standardize_audio(analysis.load_audio(args.audio))",
  "            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
  "pydub.exceptions.CouldntDecodeError: Decoding failed. ffmpeg returned error code: 1",
  "",
  "/tmp/insert-sections-AbC123/1700-broken.mp3: Invalid data found when processing input",
  "",
].join("\n");

// The failure helpers log by design; keep test output readable.
const quietly = <T>(run: () => T): T => {
  const original = console.error;
  console.error = () => undefined;
  try {
    return run();
  } finally {
    console.error = original;
  }
};

describe("client-safe error detail", () => {
  test("keeps the last meaningful line and removes paths", () => {
    const detail = clientSafeDetail(TRACEBACK);
    assert.equal(detail, "<path>: Invalid data found when processing input");
  });

  test("never returns a traceback frame", () => {
    const detail = clientSafeDetail(
      'Traceback (most recent call last):\n  File "/srv/x.py", line 3, in <module>\n',
    );
    assert.equal(detail, "");
  });

  test("this API's own routes are left alone", () => {
    assert.equal(
      clientSafeDetail("TTS failed: see GET /api/capabilities or http://localhost:3001/ad/insert"),
      "TTS failed: see GET /api/capabilities or http://localhost:3001/ad/insert",
    );
  });

  test("Windows paths are removed too", () => {
    assert.equal(
      clientSafeDetail("Error opening C:\\Users\\me\\AppData\\Temp\\x.mp3"),
      "Error opening <path>",
    );
  });

  test("a subprocess exit is described by its stderr, a timeout by our message", () => {
    const exited = new SubprocessError("analyze_cli exited with code 1.", "analyze_cli", "exit", TRACEBACK);
    assert.equal(quietly(() => logFailure("x", exited)), "<path>: Invalid data found when processing input");

    const timedOut = new SubprocessError(
      "ffmpeg did not finish within 120 s and was stopped.",
      "ffmpeg",
      "timeout",
      "frame= 1 size= /tmp/a/b.mp3",
    );
    assert.equal(quietly(() => logFailure("x", timedOut)), timedOut.message);
  });

  test("the response sentence carries the reason", () => {
    const body = quietly(() => failureBody("Merge failed.", new Error("ENOENT: open '/tmp/x/y.mp3'")));
    assert.equal(body.error, "Merge failed. ENOENT: open '<path>'");
    assert.equal(body.detail, "ENOENT: open '<path>'");
  });

  test("a thrown non-Error with a hostile toString does not throw again", () => {
    const body = quietly(() => failureBody("TTS failed.", { toString: 1 }));
    assert.equal(body.error, "TTS failed. Unknown error.");
  });
});

describe("provider failures", () => {
  const PROVIDER_BODY = JSON.stringify({
    detail: { status: "invalid_api_key", message: "Invalid API key", request_id: "req_123" },
  });

  test("an ElevenLabs error is a 502 with a generic sentence, never the provider's body", () => {
    const error = new ElevenLabsError({ message: "Unauthorized", statusCode: 401, body: JSON.parse(PROVIDER_BODY) });
    const failure = quietly(() => describeFailure("TTS failed.", error));
    assert.equal(failure.status, 502);
    assert.equal(failure.body.error, `TTS failed. ${PROVIDER_MESSAGES.elevenlabs}`);
    const sent = JSON.stringify(failure.body);
    for (const leaked of ["Invalid API key", "req_123", "invalid_api_key", "401"]) {
      assert.ok(!sent.includes(leaked), `${leaked} reached the client: ${sent}`);
    }
  });

  test("an OpenAI or SerpAPI error is a 502 too, with the body only in the log", () => {
    const logged: unknown[][] = [];
    const original = console.error;
    console.error = (...args: unknown[]) => {
      logged.push(args);
    };
    try {
      const failure = describeFailure(
        "Related products failed.",
        new ProviderError("openai", 429, '{"error":{"message":"You exceeded your current quota"}}'),
      );
      assert.equal(failure.status, 502);
      assert.ok(!JSON.stringify(failure.body).includes("quota"));
      assert.ok(JSON.stringify(logged).includes("quota"), "the detail belongs in the log");
    } finally {
      console.error = original;
    }
  });

  test("a pipeline's provider exception is not forwarded from its stderr", () => {
    const stderr = [
      "Traceback (most recent call last):",
      '  File "/srv/backend/ad_inserter/llm.py", line 170, in generate',
      "openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-proj-abc'}}",
    ].join("\n");
    const error = new SubprocessError("ad_inserter.insert_ad exited with code 1.", "x", "exit", stderr);
    const detail = quietly(() => logFailure("Insertion failed.", error));
    assert.equal(detail, "An upstream provider rejected the request.");
  });

  test("an ordinary pipeline error still says what went wrong", () => {
    const error = new SubprocessError(
      "x exited with code 1.",
      "x",
      "exit",
      "RuntimeError: Speaker diarization unavailable; DUO mode requires diarization.",
    );
    assert.match(quietly(() => logFailure("x", error)), /DUO mode requires diarization/);
  });
});

describe("failure statuses", () => {
  test("an HttpError keeps its status and headers", () => {
    const failure = describeFailure("x", new HttpError(503, "Busy.", { "Retry-After": "30" }));
    assert.equal(failure.status, 503);
    assert.deepEqual(failure.headers, { "Retry-After": "30" });
    assert.equal(failure.body.error, "Busy.");
  });

  test("a child stopped by shutdown is a 503 to retry, anything else a 500", () => {
    const stopped = new SubprocessError("analyze_cli was stopped because the server is shutting down.", "a", "shutdown", "");
    assert.equal(quietly(() => describeFailure("x", stopped)).status, 503);
    assert.equal(quietly(() => describeFailure("x", new Error("boom"))).status, 500);
  });
});
