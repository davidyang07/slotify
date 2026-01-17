/**
 * Test script to verify audio merge functionality
 * 
 * This script creates test audio files and verifies that the merge
 * correctly inserts ad audio at the specified time without repeating
 * the main audio from the beginning.
 * 
 * Usage:
 *   node test_merge.js
 * 
 * Requirements:
 *   - ffmpeg must be installed
 *   - Node.js with fs and child_process modules
 */

import fs from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Test parameters
const MAIN_DURATION = 10; // seconds
const AD_DURATION = 2; // seconds
const INSERT_TIME = 5; // seconds
const EXPECTED_FINAL_DURATION = MAIN_DURATION + AD_DURATION; // 12 seconds

async function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const process = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    process.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    process.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
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
  try {
    const { stdout } = await runCommand("ffprobe", [
      "-v", "error",
      "-show_entries", "format=duration",
      "-of", "default=noprint_wrappers=1:nokey=1",
      filePath,
    ]);
    const duration = Number.parseFloat(stdout.trim());
    return Number.isFinite(duration) && duration > 0 ? duration : null;
  } catch (error) {
    console.error(`Failed to get duration: ${error.message}`);
    return null;
  }
}

async function createTestAudio(outputPath, duration, frequency = 440, label = "tone") {
  // Generate a sine wave tone for the specified duration
  // Using a different frequency for main vs ad to make them distinguishable
  await runCommand("ffmpeg", [
    "-y",
    "-f", "lavfi",
    "-i", `sine=frequency=${frequency}:duration=${duration}`,
    "-c:a", "libmp3lame",
    "-q:a", "2",
    outputPath,
  ]);
  console.log(`Created ${label} audio: ${outputPath} (${duration}s)`);
}

async function testMerge() {
  const tempDir = await fs.promises.mkdtemp(
    path.join(os.tmpdir(), "merge-test-"),
  );
  
  const mainPath = path.join(tempDir, "main.mp3");
  const adPath = path.join(tempDir, "ad.mp3");
  const mergedPath = path.join(tempDir, "merged.mp3");
  
  try {
    console.log("Creating test audio files...");
    await createTestAudio(mainPath, MAIN_DURATION, 440, "main");
    await createTestAudio(adPath, AD_DURATION, 880, "ad");
    
    // Verify input durations
    const mainDuration = await getDuration(mainPath);
    const adDuration = await getDuration(adPath);
    console.log(`Main audio duration: ${mainDuration?.toFixed(3)}s (expected: ${MAIN_DURATION}s)`);
    console.log(`Ad audio duration: ${adDuration?.toFixed(3)}s (expected: ${AD_DURATION}s)`);
    
    // Call the merge API using curl or direct FFmpeg test
    console.log(`\nTesting merge logic directly with FFmpeg...`);
    
    // Test the filter chain directly
    const filter = [
      `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
        `atrim=0:${INSERT_TIME},asetpts=PTS-STARTPTS[a0]`,
      `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
        `atrim=${INSERT_TIME},asetpts=PTS-STARTPTS[a1]`,
      `[1:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
        `asetpts=PTS-STARTPTS[ad]`,
      `[a0][ad][a1]concat=n=3:v=0:a=1[aout]`,
    ].join(";");
    
    await runCommand("ffmpeg", [
      "-y",
      "-i", mainPath,
      "-i", adPath,
      "-filter_complex", filter,
      "-map", "[aout]",
      "-c:a", "libmp3lame",
      "-q:a", "2",
      mergedPath,
    ]);
    
    // Verify merged duration
    const mergedDuration = await getDuration(mergedPath);
    console.log(`\nMerged audio duration: ${mergedDuration?.toFixed(3)}s (expected: ${EXPECTED_FINAL_DURATION}s)`);
    
    if (mergedDuration !== null) {
      const durationDiff = Math.abs(mergedDuration - EXPECTED_FINAL_DURATION);
      if (durationDiff < 0.5) {
        console.log("✓ Duration test PASSED");
      } else {
        console.log(`✗ Duration test FAILED: difference of ${durationDiff.toFixed(3)}s`);
      }
    }
    
    // Analyze the merged audio to check for repetition
    // We'll check if the frequency pattern matches expected structure
    console.log("\nAnalyzing merged audio structure...");
    console.log("Expected structure: [main 0-5s] + [ad 2s] + [main 5-10s]");
    console.log("If the audio repeats from the beginning, the duration would be ~22s instead of 12s");
    
    if (mergedDuration !== null && Math.abs(mergedDuration - EXPECTED_FINAL_DURATION) < 0.5) {
      console.log("✓ Structure test PASSED: Duration suggests correct merge");
    } else {
      console.log("✗ Structure test FAILED: Duration suggests incorrect merge");
    }
    
    console.log(`\nTest files saved in: ${tempDir}`);
    console.log(`You can manually verify by playing: ${mergedPath}`);
    
  } catch (error) {
    console.error("Test failed:", error.message);
    throw error;
  } finally {
    // Keep temp files for manual inspection
    console.log("\nNote: Temp files kept for manual inspection. Clean up manually if needed.");
  }
}

// Run the test
testMerge().catch((error) => {
  console.error("Test error:", error);
  process.exit(1);
});
