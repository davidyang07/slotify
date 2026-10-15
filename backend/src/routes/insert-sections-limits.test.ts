/**
 * /api/insert-sections refusals that come before the analyser runs, so they
 * need ffprobe but not the Python environment.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, describe, test } from "node:test";
import type { Server } from "node:http";

// Read at config load: a 1.2-second song limit.
process.env.SLOTIFY_MAX_SONG_MINUTES = "0.02";

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "insert-limits-test-"));
let server: Server;
let baseUrl: string;

const ffmpeg = (args: string[]) => {
  const result = spawnSync("ffmpeg", ["-v", "error", "-y", ...args], { timeout: 30_000 });
  assert.equal(result.status, 0, String(result.stderr));
};

before(async () => {
  ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=3", path.join(scratch, "song.mp3")]);
  // Written to a pipe, a webm carries no duration.
  const piped = spawnSync(
    "ffmpeg",
    ["-v", "error", "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "libopus", "-f", "webm", "pipe:1"],
    { timeout: 30_000, maxBuffer: 10 * 1024 * 1024 },
  );
  assert.equal(piped.status, 0, String(piped.stderr));
  fs.writeFileSync(path.join(scratch, "no-duration.webm"), piped.stdout);

  const express = (await import("express")).default;
  const { insertSectionsRouter } = await import("./insert-sections");
  const { errorHandler } = await import("../middleware/error-handler");
  const app = express();
  app.use(insertSectionsRouter);
  app.use(errorHandler);
  await new Promise<void>((resolve) => {
    server = app.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  baseUrl = typeof address === "object" && address ? `http://127.0.0.1:${address.port}` : "";
});

after(() => {
  server?.close();
  fs.rmSync(scratch, { recursive: true, force: true });
});

const analyse = async (file: string, mode: string) => {
  const form = new FormData();
  form.append("audio", new Blob([fs.readFileSync(path.join(scratch, file))], { type: "audio/mpeg" }), file);
  form.append("mode", mode);
  const response = await fetch(`${baseUrl}/api/insert-sections`, { method: "POST", body: form });
  return { status: response.status, body: (await response.json()) as any };
};

describe("song mode has its own length cap (SLOTIFY_MAX_SONG_MINUTES)", () => {
  test("a song over the cap is a 413 that names the setting", async () => {
    const { status, body } = await analyse("song.mp3", "song");
    assert.equal(status, 413);
    assert.equal(body.placementStatus, "unavailable");
    assert.match(body.error, /SLOTIFY_MAX_SONG_MINUTES/);
  });

  test("a song whose length cannot be read is refused, not analysed blind", async () => {
    const { status, body } = await analyse("no-duration.webm", "song");
    assert.equal(status, 422);
    assert.match(body.error, /duration/);
  });
});
