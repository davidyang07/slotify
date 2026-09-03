import assert from "node:assert/strict";
import test from "node:test";

import { hygieneFindings, HYGIENE_RULES } from "./hygiene.mjs";

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

test("product media is exempt; corpus audio is not", () => {
  assert.deepEqual(
    hygieneFindings([
      "backend/audio_tests/rogan-test1.mp3",
      "frontend/src/assets/slotify-soundwave.mp4",
    ]),
    []
  );
  const [finding] = hygieneFindings(["data/normalized/ep_0001.wav"]);
  assert.equal(finding.rule, "audio-corpus");
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
