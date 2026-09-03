#!/usr/bin/env node
/**
 * Fail if the repository has committed something it should not have.
 *
 * The rules live in `lib/hygiene.mjs` as a pure function so they can be tested;
 * this is the entry point that feeds them `git ls-files`. It judges what is
 * COMMITTED rather than what happens to be in a working tree: an editor's
 * scratch file nobody committed is not a repository problem, and a committed
 * cache, credential or corpus file is one forever.
 *
 * `npm run verify:ml` runs the same check inline. This file exists so CI can run
 * it on the Node-only job, without a Python environment.
 *
 *   node scripts/check-hygiene.mjs
 */

import { execFileSync } from "node:child_process";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { hygieneFindings } from "./lib/hygiene.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const tracked = execFileSync("git", ["ls-files"], {
  cwd: REPO_ROOT,
  encoding: "utf8",
  maxBuffer: 32 * 1024 * 1024,
}).split(/\r?\n/);

const findings = hygieneFindings(tracked);

for (const finding of findings) {
  console.error(`${finding.path} [${finding.rule}] ${finding.reason}`);
}

if (findings.length) {
  console.error(`${findings.length} committed file(s) should not be tracked.`);
  process.exit(1);
}

console.log(`${tracked.filter(Boolean).length} tracked file(s), no findings.`);
