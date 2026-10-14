import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { clientSafeDetail, failureBody, logFailure } from "./errors";
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
