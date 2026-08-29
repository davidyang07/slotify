#!/usr/bin/env node
/**
 * Regenerate artifacts/reports/resume_evidence.md and .json.
 *
 * The report says PASS or FAIL for every claim a resume might make about this
 * repository, reading each verdict out of a generated artifact. Nothing in the
 * chain below can produce a number: `dataset stats` and `features stats`
 * recount what is on disk, and `report resume-evidence` reads those counts
 * against thresholds that live in the committed experiment definition. If a
 * claim is unsupported, the way to change that is to run the experiment, not to
 * run this again.
 *
 *   npm run resume-evidence                 # regenerate and write
 *   npm run resume-evidence -- --check      # CI: fail if the report has drifted
 *   npm run resume-evidence -- --split-version v3
 */

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ML_DIR = path.join(REPO_ROOT, "ml");

const argv = process.argv.slice(2);
const check = argv.includes("--check");
const splitIndex = argv.indexOf("--split-version");
const splitVersion = splitIndex === -1 ? "v3" : argv[splitIndex + 1] ?? "v3";

const pythonBin = (() => {
  if (process.env.SLOTIFY_ML_PYTHON) return process.env.SLOTIFY_ML_PYTHON;
  const venv =
    process.platform === "win32"
      ? path.join(ML_DIR, ".venv", "Scripts", "python.exe")
      : path.join(ML_DIR, ".venv", "bin", "python");
  return fs.existsSync(venv) ? venv : process.env.PYTHON_BIN ?? "python";
})();

/*
 * In --check mode nothing is regenerated first. That is deliberate: the point of
 * the check is to prove the COMMITTED report still agrees with the COMMITTED
 * artifacts, and recomputing the artifacts first would quietly repair exactly
 * the disagreement the check exists to find.
 */
const steps = check
  ? [
      {
        name: "resume evidence (check)",
        args: ["report", "resume-evidence", "--check"],
      },
    ]
  : [
      {
        name: "dataset statistics",
        args: ["dataset", "stats", "--split-version", splitVersion],
      },
      {
        name: "feature statistics",
        args: ["features", "stats", "--split-version", splitVersion],
      },
      {
        name: "resume evidence",
        args: ["report", "resume-evidence"],
      },
    ];

let failed = 0;
for (const step of steps) {
  console.log(`\n=== ${step.name} ===`);
  const result = spawnSync(pythonBin, ["-m", "slotify_rank.cli", ...step.args], {
    cwd: ML_DIR,
    stdio: "inherit",
    env: {
      ...process.env,
      PYTHONPATH: [path.join(ML_DIR, "src"), process.env.PYTHONPATH]
        .filter(Boolean)
        .join(path.delimiter),
    },
  });
  if (result.status !== 0) {
    failed += 1;
    console.error(`  ${step.name} failed (exit ${result.status}).`);
  }
}

if (failed) {
  console.error(`\n${failed} step(s) failed.`);
} else if (check) {
  console.log("\nThe committed resume evidence agrees with its raw artifacts.");
} else {
  console.log("\nSee artifacts/reports/resume_evidence.md.");
}
process.exit(failed ? 1 : 0);
