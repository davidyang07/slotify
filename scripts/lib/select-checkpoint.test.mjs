/**
 * Which checkpoint the product serves.
 *
 * The rule under test is "a human-trained run if any exists, otherwise the
 * committed bootstrap" -- deliberately not "the newest" and deliberately not
 * "the best metric". Preferring the best metric would silently pick a weakly
 * supervised run that scored well against its own teacher, which is precisely
 * the number that means nothing.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, describe, it } from "node:test";

import {
  BOOTSTRAP_CHECKPOINT,
  selectCheckpoint,
} from "./select-checkpoint.mjs";

const temporaryRoots = [];

const makeRepo = () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "slotify-checkpoint-"));
  temporaryRoots.push(root);
  return root;
};

after(() => {
  for (const root of temporaryRoots) {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

/** Write a run directory complete enough to be selectable. */
const addRun = (root, name, summary, { complete = true } = {}) => {
  const dir = path.join(root, "artifacts", "training", name);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(
    path.join(dir, "training_summary.json"),
    JSON.stringify(summary),
  );
  if (complete) {
    fs.writeFileSync(path.join(dir, "best_checkpoint.pt"), "weights");
    fs.writeFileSync(path.join(dir, "normalizer.json"), "{}");
  }
  return dir;
};

describe("selectCheckpoint", () => {
  it("returns nothing usable when there is no checkpoint at all", () => {
    const selected = selectCheckpoint(makeRepo());
    assert.equal(selected.checkpoint, null);
    assert.equal(selected.isHumanTrained, false);
    assert.match(selected.reason, /heuristic baseline/);
  });

  it("falls back to the committed bootstrap and says what it is", () => {
    const root = makeRepo();
    addRun(root, path.basename(path.dirname(BOOTSTRAP_CHECKPOINT)), {
      run_id: "gated-d8ed976101aa4c3b",
      label_source: "weak_heuristic",
      model_variant: "gated",
    });
    const selected = selectCheckpoint(root);
    assert.equal(selected.checkpoint, BOOTSTRAP_CHECKPOINT);
    assert.equal(selected.labelSource, "weak_heuristic");
    assert.equal(selected.isHumanTrained, false);
    assert.match(selected.reason, /not ranking quality/);
  });

  it("prefers a human-trained run over the bootstrap", () => {
    const root = makeRepo();
    addRun(root, path.basename(path.dirname(BOOTSTRAP_CHECKPOINT)), {
      run_id: "bootstrap",
      label_source: "weak_heuristic",
      model_variant: "gated",
    });
    addRun(root, "gated-human", {
      run_id: "gated-human",
      label_source: "human",
      model_variant: "gated",
      generated_at: "2026-09-01T00:00:00+00:00",
    });
    const selected = selectCheckpoint(root);
    assert.equal(selected.isHumanTrained, true);
    assert.equal(selected.runId, "gated-human");
    assert.equal(selected.labelSource, "human");
  });

  it("takes the most recent human-trained run, not the best-scoring one", () => {
    const root = makeRepo();
    addRun(root, "gated-old", {
      run_id: "gated-old",
      label_source: "human",
      generated_at: "2026-09-01T00:00:00+00:00",
      best_validation_ndcg_at_3: 0.99,
    });
    addRun(root, "gated-new", {
      run_id: "gated-new",
      label_source: "human",
      generated_at: "2026-09-02T00:00:00+00:00",
      best_validation_ndcg_at_3: 0.42,
    });
    assert.equal(selectCheckpoint(root).runId, "gated-new");
  });

  it("ignores a run whose weights or normalizer are missing", () => {
    const root = makeRepo();
    addRun(
      root,
      "gated-human",
      { run_id: "gated-human", label_source: "human" },
      { complete: false },
    );
    assert.equal(selectCheckpoint(root).checkpoint, null);
  });

  it("ignores a run with an unreadable summary rather than crashing", () => {
    const root = makeRepo();
    const dir = path.join(root, "artifacts", "training", "broken");
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, "training_summary.json"), "{ not json");
    fs.writeFileSync(path.join(dir, "best_checkpoint.pt"), "weights");
    fs.writeFileSync(path.join(dir, "normalizer.json"), "{}");
    assert.equal(selectCheckpoint(root).checkpoint, null);
  });
});
