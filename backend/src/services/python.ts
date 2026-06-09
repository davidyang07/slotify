import { spawn } from "node:child_process";
import { BACKEND_DIR, pythonBin } from "../config";

/**
 * Run `python -m ad_inserter.analyze_cli` for the given audio and parse its
 * JSON stdout. Resolves to the parsed analysis payload.
 */
export const runPythonAnalyze = (
  audioPath: string,
  mode: string,
): Promise<any> =>
  new Promise((resolve, reject) => {
    const args = [
      "-m",
      "ad_inserter.analyze_cli",
      "--audio",
      audioPath,
      "--mode",
      mode,
      "--snippet-count",
      "12",
    ];
    const child = spawn(pythonBin, args, {
      cwd: BACKEND_DIR,
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(stdout.trim() || "{}"));
        } catch (parseError) {
          reject(parseError);
        }
      } else {
        reject(new Error(stderr || `analyze_cli exited with code ${code}`));
      }
    });
  });

/**
 * Run a Python `ad_inserter` module that writes its result to a file (the
 * single- and two-speaker insertion pipelines). Resolves when it exits 0.
 */
export const runPythonModule = (
  args: string[],
  label = "ad_inserter",
): Promise<void> =>
  new Promise((resolve, reject) => {
    const child = spawn(pythonBin, args, {
      cwd: BACKEND_DIR,
      env: process.env,
    });
    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(stderr || `${label} exited with code ${code}`));
      }
    });
  });
