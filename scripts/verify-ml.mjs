#!/usr/bin/env node
/**
 * Verify the ranking half of the repository: the Python package, the committed
 * evidence, and -- when a local corpus is present -- the dataset itself.
 *
 * `npm run verify` covers the two Node workspaces. This covers everything else,
 * and `npm run verify:all` runs both. Between them they check every claim this
 * repository makes that can be checked without collecting new human labels.
 *
 * TWO TIERS, AND WHY THE SECOND ONE IS NOT SILENT.
 *
 * The first tier needs nothing but a checkout: the test suite, and the two
 * generated reports re-derived from the committed artifacts and compared. That
 * tier runs everywhere, CI included, and a failure in it is a failure.
 *
 * The second tier reads `data/`, which is never committed -- the corpus is
 * reconstructed from `ml/configs/sources_v2.yaml` rather than stored. On a fresh
 * clone those steps cannot run, and this script says so by name rather than
 * passing quietly: a check that skips without saying which one is worse than no
 * check, because it reads as coverage.
 *
 * Nothing here writes a label, mutates the label database, or regenerates an
 * artifact. `--check` modes compare; they never repair.
 *
 *   npm run verify:ml
 *   npm run verify:ml -- --skip-tests     # reports only, for a fast loop
 */

import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { hygieneFindings } from "./lib/hygiene.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ML_DIR = path.join(REPO_ROOT, "ml");
const DATA_DIR = path.join(REPO_ROOT, "data");

const argv = process.argv.slice(2);
const skipTests = argv.includes("--skip-tests");
const splitIndex = argv.indexOf("--split-version");
const splitVersion = splitIndex === -1 ? "v4" : argv[splitIndex + 1] ?? "v4";

const pythonBin = (() => {
  if (process.env.SLOTIFY_ML_PYTHON) return process.env.SLOTIFY_ML_PYTHON;
  const venv =
    process.platform === "win32"
      ? path.join(ML_DIR, ".venv", "Scripts", "python.exe")
      : path.join(ML_DIR, ".venv", "bin", "python");
  return fs.existsSync(venv) ? venv : process.env.PYTHON_BIN ?? "python";
})();

const candidatesManifest = path.join(DATA_DIR, "manifests", "candidates.jsonl");
const featuresManifest = path.join(DATA_DIR, "manifests", "features.jsonl");
const queueArtifact = path.join(DATA_DIR, "labels", "queue_full-v2.json");

const hasCorpus = fs.existsSync(candidatesManifest);
const hasFeatures = fs.existsSync(featuresManifest);
const hasQueue = fs.existsSync(queueArtifact);

// Reports written while verifying go here, never over a committed artifact.
const scratchDir = fs.mkdtempSync(path.join(os.tmpdir(), "slotify-verify-"));

/**
 * @type {{name: string, args: string[]|null, need?: string, module?: boolean}[]}
 * `need` names the missing local input when a step cannot run.
 */
const steps = [];

if (!skipTests) {
  steps.push({
    name: "ML test suite",
    // The whole suite: dataset schema and validation, split leakage, queue
    // integrity, label integrity, feature assembly, the ranking losses, the
    // trainer, the evaluator, the baselines and the evidence reports.
    module: "pytest",
    args: ["-q"],
  });
}

steps.push(
  {
    name: "committed model evidence agrees with its artifacts",
    args: ["report", "model-evidence", "--check"],
  },
  {
    // Two assertions in one pass, and --check is what keeps it read-only: the
    // committed report still agrees with its artifacts, AND every capability
    // this repository claims to implement is established by committed code, a
    // test and an artifact. It deliberately says nothing about the empirical
    // claims, which no amount of software can settle.
    name: "claim evidence agrees, and every implemented capability is established",
    args: ["report", "claim-evidence", "--check", "--require-implemented"],
  },
  {
    // 25 checks, including split_leakage_by_episode and split_leakage_by_series.
    // The report goes to a scratch path so verifying never rewrites a committed
    // artifact: a verification that edits its own evidence is not one.
    name: "dataset validation, including split leakage by episode and by series",
    args: hasCorpus
      ? [
          "dataset",
          "validate",
          "--split-version",
          splitVersion,
          "--output",
          path.join(scratchDir, "validation_report.json"),
        ]
      : null,
    need: "data/manifests/candidates.jsonl (run `dataset import-local` first)",
  },
  {
    name: "multimodal feature validation",
    args: hasFeatures
      ? [
          "features",
          "validate",
          "--split-version",
          splitVersion,
          "--output",
          path.join(scratchDir, "feature_validation_report.json"),
        ]
      : null,
    need: "data/manifests/features.jsonl (run `pipeline features` first)",
  },
  {
    name: "labelling queue summary agrees with the queue",
    args: hasQueue ? ["label", "queue-summary", "--check"] : null,
    need: "data/labels/queue_full-v2.json (run `label queue` first)",
  },
  {
    name: "label quality controls",
    // Read-only by construction: `label check` reports and never mutates, so it
    // cannot touch the canonical label database.
    args: hasQueue
      ? [
          "label",
          "check",
          "--queue",
          queueArtifact,
          "--split-version",
          splitVersion,
          "--output",
          path.join(scratchDir, "label_quality_report.json"),
        ]
      : null,
    need: "data/labels/queue_full-v2.json",
  },
  {
    // Reports how far the labelling round has to go. It is NOT run with
    // --require-ready: the gate is not met until human labels exist, and a
    // verification that fails for want of data collection would say nothing
    // about the software.
    name: "experiment readiness gate",
    args: hasCorpus
      ? [
          "experiment",
          "readiness",
          "--split-version",
          splitVersion,
          "--experiment-config",
          "configs/experiment_v2.yaml",
          "--output",
          path.join(scratchDir, "readiness_report.json"),
        ]
      : null,
    need: "data/manifests/candidates.jsonl",
  },
);

let failed = 0;
let skipped = 0;
const ran = [];

// Repository hygiene, over what git actually tracks rather than what is lying
// around: a committed cache or credential is permanent, an uncommitted one is
// not a repository problem.
console.log("\n=== repository hygiene ===");
try {
  const tracked = execFileSync("git", ["ls-files"], {
    cwd: REPO_ROOT,
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
  }).split(/\r?\n/);
  const findings = hygieneFindings(tracked);
  if (findings.length) {
    failed += 1;
    console.error(`  ${findings.length} committed file(s) should not be tracked:`);
    for (const finding of findings.slice(0, 40)) {
      console.error(`    ${finding.path} - ${finding.rule}: ${finding.reason}`);
    }
    if (findings.length > 40) {
      console.error(`    ... and ${findings.length - 40} more.`);
    }
  } else {
    ran.push("repository hygiene");
    console.log(`  ${tracked.filter(Boolean).length} tracked file(s), no findings.`);
  }
} catch (error) {
  skipped += 1;
  console.log(`  SKIPPED - git ls-files unavailable: ${error.message}`);
}

for (const step of steps) {
  if (step.args === null) {
    skipped += 1;
    console.log(`\n=== ${step.name} ===`);
    console.log(`  SKIPPED - no local corpus: needs ${step.need}`);
    continue;
  }
  console.log(`\n=== ${step.name} ===`);
  const args = step.module
    ? ["-m", step.module, ...step.args]
    : ["-m", "slotify_rank.cli", ...step.args];
  const result = spawnSync(pythonBin, args, {
    cwd: ML_DIR,
    stdio: "inherit",
    env: {
      ...process.env,
      PYTHONPATH: [path.join(ML_DIR, "src"), process.env.PYTHONPATH]
        .filter(Boolean)
        .join(path.delimiter),
    },
  });
  if (result.error) {
    failed += 1;
    console.error(`  ${step.name} could not start: ${result.error.message}`);
    continue;
  }
  if (result.status !== 0) {
    failed += 1;
    console.error(`  ${step.name} failed (exit ${result.status}).`);
    continue;
  }
  ran.push(step.name);
}

console.log("\n=== summary ===");
console.log(`  ${ran.length} check(s) passed.`);
if (skipped) {
  console.log(
    `  ${skipped} check(s) skipped: they read data/, which is not committed. ` +
      "Rebuild the corpus with `slotify-rank dataset import-local` and " +
      "`pipeline features` to run them."
  );
}
if (failed) {
  console.error(`  ${failed} check(s) FAILED.`);
  fs.rmSync(scratchDir, { recursive: true, force: true });
  process.exit(1);
}
fs.rmSync(scratchDir, { recursive: true, force: true });
console.log("  No failures.");
