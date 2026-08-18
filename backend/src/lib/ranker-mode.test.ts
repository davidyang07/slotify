import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test, describe, afterEach } from "node:test";

import { resolveRankerMode } from "./ranker-mode";

const saved = {
  RANKER_MODE: process.env.RANKER_MODE,
  SLOTIFY_RANKER_CHECKPOINT: process.env.SLOTIFY_RANKER_CHECKPOINT,
  SLOTIFY_RANKER_NORMALIZER: process.env.SLOTIFY_RANKER_NORMALIZER,
};

const restore = () => {
  for (const [key, value] of Object.entries(saved)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
};

/** A directory holding a checkpoint and normalizer that merely exist. */
const fakeCheckpointDir = (): string => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ranker-mode-"));
  fs.writeFileSync(path.join(dir, "best_checkpoint.pt"), "not a real checkpoint");
  fs.writeFileSync(path.join(dir, "normalizer.json"), "{}");
  return dir;
};

describe("ranker mode resolution", () => {
  afterEach(restore);

  test("defaults to auto and falls back to the heuristic with nothing configured", () => {
    delete process.env.RANKER_MODE;
    delete process.env.SLOTIFY_RANKER_CHECKPOINT;
    const resolved = resolveRankerMode();
    assert.equal(resolved.requested, "auto");
    assert.equal(resolved.mode, "heuristic");
    assert.match(resolved.reason, /SLOTIFY_RANKER_CHECKPOINT is not set/);
  });

  test("heuristic mode ignores a configured checkpoint", () => {
    const dir = fakeCheckpointDir();
    process.env.RANKER_MODE = "heuristic";
    process.env.SLOTIFY_RANKER_CHECKPOINT = path.join(dir, "best_checkpoint.pt");
    const resolved = resolveRankerMode();
    assert.equal(resolved.mode, "heuristic");
    assert.equal(resolved.checkpointPath, null);
  });

  test("auto selects the learned ranker when a checkpoint is present", () => {
    const dir = fakeCheckpointDir();
    process.env.RANKER_MODE = "auto";
    process.env.SLOTIFY_RANKER_CHECKPOINT = path.join(dir, "best_checkpoint.pt");
    const resolved = resolveRankerMode();
    assert.equal(resolved.mode, "learned");
    assert.ok(resolved.checkpointPath?.endsWith("best_checkpoint.pt"));
    assert.ok(resolved.normalizerPath?.endsWith("normalizer.json"));
  });

  test("learned mode fails loudly rather than degrading silently", () => {
    process.env.RANKER_MODE = "learned";
    delete process.env.SLOTIFY_RANKER_CHECKPOINT;
    assert.throws(() => resolveRankerMode(), /RANKER_MODE=learned but the learned ranker is unusable/);
  });

  test("a checkpoint without its normalizer is not usable", () => {
    const dir = fakeCheckpointDir();
    fs.rmSync(path.join(dir, "normalizer.json"));
    process.env.RANKER_MODE = "auto";
    process.env.SLOTIFY_RANKER_CHECKPOINT = path.join(dir, "best_checkpoint.pt");
    const resolved = resolveRankerMode();
    assert.equal(resolved.mode, "heuristic");
    assert.match(resolved.reason, /normalizer .* does not exist/);
  });

  test("an unknown mode is rejected", () => {
    process.env.RANKER_MODE = "magic";
    assert.throws(() => resolveRankerMode(), /RANKER_MODE must be one of/);
  });
});
