/**
 * Test script to verify preview/export merge structure.
 *
 * This script creates a 10s main tone and two short statement tones,
 * builds a sponsor block with a 150ms pause, and verifies:
 * - Preview duration ~= 3s + sponsor_block + 3s
 * - Final duration ~= main + sponsor_block
 * - No main audio restart (duration sanity checks)
 *
 * Usage:
 *   node test_preview_export.js
 */

import fs from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const MAIN_DURATION = 10;
const STATEMENT_ONE = 1.2;
const STATEMENT_TWO = 1.0;
const PAUSE_SECONDS = 0.15;
const INSERT_TIME = 5;
const PREVIEW_WINDOW = 3;

async function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const process = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    process.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    process.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    process.on("error", reject);
    process.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(stderr || `Command exited with code ${code}`));
      }
    });
  });
}

async function getDuration(filePath) {
  const { stdout } = await runCommand("ffprobe", [
    "-v",
    "error",
    "-show_entries",
    "format=duration",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
    filePath,
  ]);
  const duration = Number.parseFloat(stdout.trim());
  return Number.isFinite(duration) ? duration : null;
}

async function createTone(outputPath, duration, frequency) {
  await runCommand("ffmpeg", [
    "-y",
    "-f",
    "lavfi",
    "-i",
    `sine=frequency=${frequency}:duration=${duration}`,
    "-c:a",
    "libmp3lame",
    "-q:a",
    "2",
    outputPath,
  ]);
}

async function runTest() {
  const tempDir = await fs.promises.mkdtemp(
    path.join(os.tmpdir(), "preview-export-test-"),
  );
  const mainPath = path.join(tempDir, "main.mp3");
  const s1Path = path.join(tempDir, "statement1.mp3");
  const s2Path = path.join(tempDir, "statement2.mp3");
  const pausePath = path.join(tempDir, "pause.mp3");
  const sponsorPath = path.join(tempDir, "sponsor_block.mp3");
  const previewPath = path.join(tempDir, "preview.mp3");
  const exportPath = path.join(tempDir, "export.mp3");

  console.log("Creating test audio...");
  await createTone(mainPath, MAIN_DURATION, 440);
  await createTone(s1Path, STATEMENT_ONE, 880);
  await createTone(s2Path, STATEMENT_TWO, 660);
  await runCommand("ffmpeg", [
    "-y",
    "-f",
    "lavfi",
    "-i",
    "anullsrc=channel_layout=stereo:sample_rate=44100",
    "-t",
    PAUSE_SECONDS.toString(),
    pausePath,
  ]);

  const sponsorFilter = [
    "[0:a]aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS[a0]",
    "[1:a]aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS[a1]",
    "[2:a]aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS[a2]",
    "[a0][a1][a2]concat=n=3:v=0:a=1[out]",
  ].join(";");

  await runCommand("ffmpeg", [
    "-y",
    "-i",
    s1Path,
    "-i",
    pausePath,
    "-i",
    s2Path,
    "-filter_complex",
    sponsorFilter,
    "-map",
    "[out]",
    "-c:a",
    "libmp3lame",
    "-q:a",
    "2",
    sponsorPath,
  ]);

  const sponsorDuration = await getDuration(sponsorPath);
  const expectedPreview = PREVIEW_WINDOW * 2 + (sponsorDuration ?? 0);
  const expectedExport = MAIN_DURATION + (sponsorDuration ?? 0);

  const previewFilter = [
    `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
      `atrim=${INSERT_TIME - PREVIEW_WINDOW}:${INSERT_TIME},asetpts=PTS-STARTPTS[a0]`,
    `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
      `atrim=${INSERT_TIME}:${INSERT_TIME + PREVIEW_WINDOW},asetpts=PTS-STARTPTS[a1]`,
    `[1:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
      `asetpts=PTS-STARTPTS[ad]`,
    `[a0][ad][a1]concat=n=3:v=0:a=1[out]`,
  ].join(";");

  await runCommand("ffmpeg", [
    "-y",
    "-i",
    mainPath,
    "-i",
    sponsorPath,
    "-filter_complex",
    previewFilter,
    "-map",
    "[out]",
    "-c:a",
    "libmp3lame",
    "-q:a",
    "2",
    previewPath,
  ]);

  const exportFilter = [
    `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
      `atrim=0:${INSERT_TIME},asetpts=PTS-STARTPTS[a0]`,
    `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
      `atrim=${INSERT_TIME},asetpts=PTS-STARTPTS[a1]`,
    `[1:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
      `asetpts=PTS-STARTPTS[ad]`,
    `[a0][ad][a1]concat=n=3:v=0:a=1[out]`,
  ].join(";");

  await runCommand("ffmpeg", [
    "-y",
    "-i",
    mainPath,
    "-i",
    sponsorPath,
    "-filter_complex",
    exportFilter,
    "-map",
    "[out]",
    "-c:a",
    "libmp3lame",
    "-q:a",
    "2",
    exportPath,
  ]);

  const previewDuration = await getDuration(previewPath);
  const exportDuration = await getDuration(exportPath);

  console.log("Sponsor block duration:", sponsorDuration?.toFixed(3));
  console.log("Preview duration:", previewDuration?.toFixed(3));
  console.log("Export duration:", exportDuration?.toFixed(3));
  console.log("Expected preview:", expectedPreview.toFixed(3));
  console.log("Expected export:", expectedExport.toFixed(3));

  const previewDiff =
    previewDuration !== null ? Math.abs(previewDuration - expectedPreview) : null;
  const exportDiff =
    exportDuration !== null ? Math.abs(exportDuration - expectedExport) : null;

  console.log(
    previewDiff !== null && previewDiff < 0.5
      ? "Preview duration check PASSED"
      : "Preview duration check FAILED",
  );
  console.log(
    exportDiff !== null && exportDiff < 0.5
      ? "Export duration check PASSED"
      : "Export duration check FAILED",
  );

  console.log(`Artifacts saved in: ${tempDir}`);
}

runTest().catch((error) => {
  console.error("Test failed:", error.message);
  process.exit(1);
});
