import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  COMMITTED_CHECKPOINT,
  hygieneFindings,
  HYGIENE_RULES,
  MAX_TRACKED_BYTES,
} from "./hygiene.mjs";
import { BOOTSTRAP_CHECKPOINT } from "./select-checkpoint.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const MB = 1024 * 1024;

test("a clean tree yields no findings", () => {
  assert.deepEqual(
    hygieneFindings([
      "README.md",
      "ml/src/slotify_rank/models/variants.py",
      "frontend/src/App.tsx",
      "artifacts/dataset/dataset_statistics.json",
      "config/heuristic_offline_v1.json",
    ]),
    []
  );
});

test("operating-system and editor droppings are findings", () => {
  const findings = hygieneFindings([
    ".DS_Store",
    "docs/Thumbs.db",
    "frontend/src/App.tsx.orig",
    "ml/notes.py~",
  ]);
  assert.equal(findings.length, 4);
  assert.deepEqual(
    [...new Set(findings.map((f) => f.rule))].sort(),
    ["editor-scratch", "os-metadata"]
  );
});

test("caches and compiled bytecode are findings even when nested", () => {
  const findings = hygieneFindings([
    "ml/src/slotify_rank/__pycache__/cli.cpython-312.pyc",
    "frontend/node_modules/react/index.js",
    "ml/.pytest_cache/CACHEDIR.TAG",
    "ml/.coverage",
  ]);
  assert.equal(findings.length, 4);
});

test("credentials are findings, and the reason says so", () => {
  const [finding] = hygieneFindings(["backend/.env"]);
  assert.equal(finding.rule, "credential");
  assert.match(finding.reason, /credential/);
});

test("committed env templates are not credentials; every other .env variant is", () => {
  assert.deepEqual(
    hygieneFindings([
      ".env.example",
      "backend/.env.example",
      "frontend/.env.sample",
      "ml/.env.template",
    ]),
    []
  );
  const findings = hygieneFindings([
    ".env",
    "backend/.env.local",
    "frontend/.env.production",
    // A local copy of a template is a real env file, whatever it started as.
    "backend/.env.example.local",
  ]);
  assert.deepEqual(
    findings.map((f) => f.rule),
    ["credential", "credential", "credential", "credential"]
  );
});

test("product media is exempt; corpus audio is not", () => {
  assert.deepEqual(
    hygieneFindings([
      "backend/audio_tests/ryan-test1.mp3",
      "frontend/src/assets/slotify-soundwave.mp4",
      "frontend/src/assets/react.svg",
      "frontend/src/assets/logo.png",
    ]),
    []
  );
  const [finding] = hygieneFindings(["data/normalized/ep_0001.wav"]);
  assert.equal(finding.rule, "audio-corpus");
});

test("the media exemption covers media, not any binary in a media directory", () => {
  const findings = hygieneFindings([
    "frontend/src/assets/model.bin",
    "backend/audio_tests/corpus.zip",
    "backend/audio_tests/weights.pt",
  ]);
  assert.deepEqual(
    findings.map((f) => [f.path, f.rule]),
    [
      ["frontend/src/assets/model.bin", "heavy-binary"],
      ["backend/audio_tests/corpus.zip", "heavy-binary"],
      ["backend/audio_tests/weights.pt", "heavy-binary"],
    ]
  );
});

test("the documented bootstrap checkpoint is exempt; another weight file is not", () => {
  assert.deepEqual(
    hygieneFindings(["artifacts/training/gated-d8ed976101aa4c3b/best_checkpoint.pt"]),
    []
  );
  const [finding] = hygieneFindings([
    "artifacts/training/gated-d8ed976101aa4c3b/last_checkpoint.pt",
  ]);
  assert.equal(finding.rule, "heavy-binary");
});

test("another run's best_checkpoint.pt is not exempt just for its file name", () => {
  const findings = hygieneFindings([
    "artifacts/training/gated-0123456789abcdef/best_checkpoint.pt",
    "artifacts/training/sweep/gated-d8ed976101aa4c3b/best_checkpoint.pt",
  ]);
  assert.deepEqual(
    findings.map((f) => f.rule),
    ["heavy-binary", "heavy-binary"]
  );
});

test("the exempt checkpoint is the one .gitignore re-includes and the demo falls back to", () => {
  assert.equal(COMMITTED_CHECKPOINT, BOOTSTRAP_CHECKPOINT.split(path.sep).join("/"));
  const gitignore = fs.readFileSync(path.join(REPO_ROOT, ".gitignore"), "utf8");
  assert.ok(
    gitignore.split(/\r?\n/).includes(`!${COMMITTED_CHECKPOINT}`),
    ".gitignore no longer re-includes the checkpoint hygiene exempts"
  );
});

test("any file over the size limit is a finding, exemptions included", () => {
  const findings = hygieneFindings([
    { path: "frontend/src/assets/slotify-soundwave.mp4", size: 71 * MB },
    { path: COMMITTED_CHECKPOINT, size: 50 * MB },
    { path: "ml/tests/fixtures/huge.json", size: MAX_TRACKED_BYTES + 1 },
  ]);
  assert.deepEqual(
    findings.map((f) => f.rule),
    ["oversized", "oversized", "oversized"]
  );
  assert.match(findings[0].reason, /71\.0 MB/);
});

test("files at or under the size limit, or of unknown size, pass the size rule", () => {
  assert.deepEqual(
    hygieneFindings([
      { path: "frontend/src/assets/slotify-soundwave.mp4", size: 2.4 * MB },
      { path: COMMITTED_CHECKPOINT, size: 5.7 * MB },
      { path: "README.md", size: MAX_TRACKED_BYTES },
      { path: "docs/unknown.md", size: null },
      "docs/bare-path.md",
    ]),
    []
  );
});

test("a more specific rule wins over the size rule", () => {
  const [finding] = hygieneFindings([{ path: "data/raw/ep_0001.wav", size: 300 * MB }]);
  assert.equal(finding.rule, "audio-corpus");
});

test("a path is reported once, under the first rule it breaks", () => {
  const findings = hygieneFindings([
    "ml/src/slotify_rank/__pycache__/cli.cpython-312.pyc",
  ]);
  assert.equal(findings.length, 1);
  assert.ok(HYGIENE_RULES.includes(findings[0].rule));
});

test("backslash-separated paths from a Windows caller are normalised", () => {
  const [finding] = hygieneFindings(["docs\\Thumbs.db"]);
  assert.equal(finding.path, "docs/Thumbs.db");
  assert.equal(finding.rule, "os-metadata");
});
