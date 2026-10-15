/**
 * Uploads go to a per-request temp dir under fixed names, are identified by
 * content before anything reads them, and the dir is gone once the response
 * is.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, describe, test } from "node:test";
import type { Server } from "node:http";
import express from "express";

import { asyncHandler } from "../lib/http";
import { errorHandler } from "./error-handler";
import { claimWorkDir, inspectUploads, upload } from "./upload";

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "upload-test-"));
const ffmpeg = (args: string[]) => {
  const result = spawnSync("ffmpeg", ["-v", "error", "-y", ...args], { timeout: 30_000 });
  assert.equal(result.status, 0, String(result.stderr));
};
const fixture = (name: string) => path.join(scratch, name);

let server: Server;
let baseUrl: string;
/** What the last request's route saw. */
let seen: { paths: string[]; buffers: unknown[]; dir: string | null } = { paths: [], buffers: [], dir: null };

before(async () => {
  ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=2", fixture("tone.mp3")]);
  ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", fixture("tone.m4a")]);
  ffmpeg(["-f", "lavfi", "-i", "color=c=black:s=16x16:d=0.5", "-c:v", "mpeg4", fixture("video-only.mp4")]);
  // An HLS playlist pointing at a local file: what content probing would
  // otherwise open and read.
  fs.writeFileSync(
    fixture("playlist.m3u8"),
    `#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2,\n${fixture("tone.mp3")}\n#EXT-X-ENDLIST\n`,
  );

  const app = express();
  app.post(
    "/inspect",
    upload.single("audio", { dirPrefix: "upload-test-dir-" }),
    asyncHandler(async (req, res) => {
      const dir = await claimWorkDir(req, "upload-test-dir-");
      seen = { paths: [], buffers: [req.file?.buffer], dir };
      try {
        const files = await inspectUploads(req, { maxSeconds: 60 });
        seen.paths = files.map((file) => file.path);
        res.json(files.map((file) => ({ name: path.basename(file.path), probe: file.probe })));
      } finally {
        fs.rmSync(dir, { recursive: true, force: true });
      }
    }),
  );
  // Never claims its dir: the upload middleware removes it.
  app.post("/unclaimed", upload.single("audio", { dirPrefix: "upload-test-dir-" }), (req, res) => {
    seen = { paths: [req.file?.path ?? ""], buffers: [], dir: req.file ? path.dirname(req.file.path) : null };
    res.status(400).json({ error: "refused before any route work" });
  });
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

const send = (route: string, contents: Buffer, filename: string, type = "audio/mpeg") => {
  const form = new FormData();
  form.append("audio", new Blob([contents], { type }), filename);
  return fetch(`${baseUrl}${route}`, { method: "POST", body: form });
};

const statusOf = async (contents: Buffer, filename: string, type?: string) => {
  const response = await send("/inspect", contents, filename, type);
  const body = (await response.json()) as { error?: string };
  return { status: response.status, error: body.error ?? "" };
};

describe("upload storage", () => {
  test("uploads stream to disk under a fixed name with the detected extension", async () => {
    const response = await send("/inspect", fs.readFileSync(fixture("tone.m4a")), "../../etc/My Song.mp3");
    assert.equal(response.status, 200);
    const [file] = (await response.json()) as Array<{ name: string; probe: { formatName: string; durationSeconds: number } }>;
    assert.equal(file.name, "audio-0.m4a");
    assert.match(file.probe.formatName, /mp4/);
    assert.ok(file.probe.durationSeconds > 0.5);
    assert.deepEqual(seen.buffers, [undefined], "the upload was buffered in memory");
    assert.ok(seen.paths[0].startsWith(path.join(os.tmpdir(), "upload-test-dir-")));
  });

  test("an overlong client filename is not a 500", async () => {
    const { status } = await statusOf(fs.readFileSync(fixture("tone.mp3")), `${"x".repeat(4_000)}.mp3`);
    assert.equal(status, 200);
  });

  test("a request that ends before a route claims its dir still has it removed", async () => {
    const response = await send("/unclaimed", fs.readFileSync(fixture("tone.mp3")), "a.mp3");
    assert.equal(response.status, 400);
    assert.ok(seen.dir);
    const deadline = Date.now() + 2_000;
    while (fs.existsSync(seen.dir) && Date.now() < deadline) await new Promise((r) => setTimeout(r, 20));
    assert.equal(fs.existsSync(seen.dir), false);
  });
});

describe("content checks", () => {
  test("an HLS playlist is a 415, whatever it is called", async () => {
    const playlist = fs.readFileSync(fixture("playlist.m3u8"));
    for (const [name, type] of [
      ["episode.mp3", "audio/mpeg"],
      ["episode", "application/octet-stream"],
      ["episode.m3u8", "audio/x-mpegurl"],
    ]) {
      const { status, error } = await statusOf(playlist, name, type);
      assert.equal(status, 415, `${name}: ${error}`);
    }
  });

  test("a file that is not media is a 415, not a 500", async () => {
    const { status, error } = await statusOf(Buffer.from("this is not audio"), "broken.mp3");
    assert.equal(status, 415);
    assert.match(error, /not a supported audio file/);
  });

  test("a container with no audio stream is a 422", async () => {
    const { status } = await statusOf(fs.readFileSync(fixture("video-only.mp4")), "clip.mp4", "video/mp4");
    assert.equal(status, 422);
  });

  test("audio longer than the limit is a 413 that names the setting", async () => {
    const long = fixture("long.mp3");
    ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "61", "-b:a", "8k", long]);
    const { status, error } = await statusOf(fs.readFileSync(long), "long.mp3");
    assert.equal(status, 413);
    assert.match(error, /SLOTIFY_MAX_AUDIO_MINUTES/);
  });
});
