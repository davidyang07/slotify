/**
 * buildMergeFilter as strings, then the same filters through a real ffmpeg.
 * This replaces scripts/test-merge.ts and scripts/test-preview-export.ts, which
 * tested a hand-copied filter string rather than the one the route builds.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, describe, test } from "node:test";

import { buildMergeFilter } from "./merge-filter";
import { getAudioDuration, runFfmpeg } from "../services/ffmpeg";

describe("buildMergeFilter", () => {
  test("splices the ad between two trims of the main audio", () => {
    const { filter, previewStart, previewEnd } = buildMergeFilter({
      insertAt: 5,
      previewSeconds: 0,
      mainDuration: 10,
    });
    assert.ok(filter.includes("atrim=0.000:5.000,asetpts=PTS-STARTPTS[a0]"), filter);
    assert.ok(filter.includes("atrim=5.000,asetpts=PTS-STARTPTS[a1]"), filter);
    assert.ok(filter.endsWith("[a0][ad][a1]concat=n=3:v=0:a=1[out]"), filter);
    assert.equal(previewStart, null);
    assert.equal(previewEnd, null);
  });

  test("pause and crossfade are applied, not just validated", () => {
    const { filter } = buildMergeFilter({
      insertAt: 5,
      previewSeconds: 0,
      mainDuration: 10,
      crossfade: 0.08,
      pause: 0.2,
    });
    assert.ok(filter.includes("afade=t=out:st=4.920:d=0.080[a0]"), filter);
    assert.ok(filter.includes("afade=t=in:st=0:d=0.080[a1]"), filter);
    assert.equal(filter.match(/anullsrc=r=44100:cl=stereo,atrim=0:0\.200/g)?.length, 2);
    assert.ok(filter.endsWith("[a0][p0][ad][p1][a1]concat=n=5:v=0:a=1[out]"), filter);
  });

  test("tiny values never reach ffmpeg in exponent notation", () => {
    const { filter } = buildMergeFilter({
      insertAt: 1e-7,
      previewSeconds: 1e-7,
      mainDuration: 10,
      crossfade: 1e-7,
      pause: 1e-7,
    });
    assert.ok(!/\de-\d/.test(filter), filter);
    // Values that round to zero are dropped rather than sent as zero-length.
    assert.ok(!filter.includes("afade"), filter);
    assert.ok(!filter.includes("anullsrc"), filter);
  });

  test("fades are clamped to the audio on each side of the cut", () => {
    const { filter } = buildMergeFilter({
      insertAt: 0.05,
      previewSeconds: 0,
      mainDuration: 0.1,
      crossfade: 1,
    });
    assert.ok(filter.includes("afade=t=out:st=0.000:d=0.050"), filter);
    assert.ok(filter.includes("afade=t=in:st=0:d=0.050"), filter);
  });

  test("an insert at 0 s has no fade-out at all", () => {
    const { filter } = buildMergeFilter({
      insertAt: 0,
      previewSeconds: 0,
      mainDuration: 10,
      crossfade: 0.08,
    });
    assert.ok(!filter.includes("afade=t=out"), filter);
  });

  test("a preview renders only the window around the cut", () => {
    const { filter, previewStart, previewEnd } = buildMergeFilter({
      insertAt: 5,
      previewSeconds: 3,
      mainDuration: 7,
    });
    assert.equal(previewStart, 2);
    assert.equal(previewEnd, 7);
    assert.ok(filter.includes("atrim=2.000:5.000"), filter);
    assert.ok(filter.includes("atrim=5.000:7.000"), filter);
  });
});

const ffmpegAvailable =
  spawnSync("ffmpeg", ["-version"]).status === 0 && spawnSync("ffprobe", ["-version"]).status === 0;

describe("merged durations through ffmpeg", { skip: !ffmpegAvailable && "ffmpeg unavailable" }, () => {
  let dir: string;
  let mainPath: string;
  let adPath: string;

  const tone = (outPath: string, seconds: number, frequency: number) =>
    runFfmpeg([
      "-y",
      "-f",
      "lavfi",
      "-i",
      `sine=frequency=${frequency}:duration=${seconds}`,
      "-c:a",
      "libmp3lame",
      "-q:a",
      "2",
      outPath,
    ]);

  const merge = async (options: Parameters<typeof buildMergeFilter>[0]): Promise<number> => {
    const outPath = path.join(dir, "out.mp3");
    const { filter } = buildMergeFilter(options);
    await runFfmpeg(
      ["-y", "-i", mainPath, "-i", adPath, "-filter_complex", filter, "-map", "[out]",
        "-c:a", "libmp3lame", "-q:a", "2", outPath],
      { timeoutMs: 30_000 },
    );
    const duration = await getAudioDuration(outPath);
    assert.ok(duration !== null);
    return duration;
  };

  before(async () => {
    dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "merge-filter-test-"));
    mainPath = path.join(dir, "main.mp3");
    adPath = path.join(dir, "ad.mp3");
    await tone(mainPath, 10, 440);
    await tone(adPath, 2, 880);
  });

  after(async () => {
    await fs.promises.rm(dir, { recursive: true, force: true });
  });

  // mp3 encoder padding adds a few tens of ms per encode.
  const near = (actual: number, expected: number) =>
    assert.ok(Math.abs(actual - expected) < 0.15, `${actual.toFixed(3)}s, expected ~${expected}s`);

  test("export: main + ad + two pauses, with no restart of the main audio", async () => {
    near(await merge({ insertAt: 5, previewSeconds: 0, mainDuration: 10, crossfade: 0.08, pause: 0.2 }), 12.4);
  });

  test("preview: window + ad + two pauses", async () => {
    near(await merge({ insertAt: 5, previewSeconds: 3, mainDuration: 10, crossfade: 0.08, pause: 0.12 }), 8.24);
  });

  test("the edges of the audio still merge", async () => {
    near(await merge({ insertAt: 0, previewSeconds: 0, mainDuration: 10, crossfade: 0.08, pause: 0.2 }), 12.4);
    near(await merge({ insertAt: 10, previewSeconds: 0, mainDuration: 10, crossfade: 0.08, pause: 0.2 }), 12.4);
    near(await merge({ insertAt: 1e-7, previewSeconds: 0, mainDuration: 10, crossfade: 1e-7, pause: 1e-7 }), 12);
  });
});
