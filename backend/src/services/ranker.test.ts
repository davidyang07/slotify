/**
 * Which ranker answers, and what the response then says about it. The learned
 * ranker is replaced by a stand-in "python" (SLOTIFY_ML_PYTHON) that prints a
 * fixed payload or fails like a crashed model run.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, afterEach, before, describe, test } from "node:test";

import { rankCandidates, type RankRequest } from "./ranker";
import { LearnedRankerError } from "./learned-ranker";

const ENV_KEYS = [
  "RANKER_MODE",
  "SLOTIFY_RANKER_CHECKPOINT",
  "SLOTIFY_ML_PYTHON",
  "SLOTIFY_RANKER_NORMALIZER",
  "SLOTIFY_RANKER_WORKER",
];
const saved = Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]]));

let dir: string;

const payload = (trainingLabelSource: string) => ({
  schema_version: "inference-v1.0.0",
  episode_id: "upload",
  duration_seconds: 120,
  score_scale: "episode_relative_min_max",
  is_calibrated_probability: false,
  candidate_count: 1,
  ranked: [
    {
      candidate_id: "upload:000060000",
      insertion_time_seconds: 60,
      insertion_ms: 60_000,
      raw_score: 1.2,
      normalized_score: 100,
      rank: 1,
      acceptability_logit: 0.3,
      audio_available: true,
      text_available: true,
      feature_status: "complete",
    },
  ],
  excluded: [],
  warnings: [],
  timings_seconds: {},
  model: {
    model_variant: "gated",
    model_run_id: "gated-d8ed976101aa4c3b",
    parameter_count: 1,
    training_label_source: trainingLabelSource,
    training_data_provenance: "weak_supervision",
  },
});

/** A stand-in interpreter: prints `stdout`, or fails with a traceback. */
const fakePython = (name: string, script: string): string => {
  const target = path.join(dir, name);
  fs.writeFileSync(target, `#!/bin/sh\n${script}\n`, { mode: 0o755 });
  return target;
};

const request = (): RankRequest => ({
  candidates: [
    { ms: 30_000, silenceMs: 900, snippet: "First point.", generatedBy: "analyser" },
    { ms: 60_000, silenceMs: 1_500, snippet: "Second point.", generatedBy: "analyser" },
  ],
  episodeId: "upload",
  audioPath: "/dev/null",
  durationSeconds: 120,
  mode: "podcast",
  minSeparationSeconds: 6,
  count: 3,
});

describe("rankCandidates", { skip: process.platform === "win32" && "uses a /bin/sh stand-in" }, () => {
  before(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "ranker-test-"));
    fs.writeFileSync(path.join(dir, "best_checkpoint.pt"), "");
    fs.writeFileSync(path.join(dir, "normalizer.json"), "{}");
    process.env.SLOTIFY_RANKER_CHECKPOINT = path.join(dir, "best_checkpoint.pt");
    delete process.env.SLOTIFY_RANKER_NORMALIZER;
    // The stand-in answers one request and exits, like `infer rank`; this
    // suite is about what a ranking reports, not how the worker is kept.
    process.env.SLOTIFY_RANKER_WORKER = "0";
  });
  afterEach(() => {
    process.env.RANKER_MODE = saved.RANKER_MODE;
    if (saved.RANKER_MODE === undefined) delete process.env.RANKER_MODE;
  });
  after(() => {
    for (const key of ENV_KEYS) {
      if (saved[key] === undefined) delete process.env[key];
      else process.env[key] = saved[key];
    }
    fs.rmSync(dir, { recursive: true, force: true });
  });

  const crashing = () =>
    fakePython(
      "crash.sh",
      "echo 'Traceback (most recent call last):' >&2\n" +
        "echo '  File \"/srv/ml/slotify_rank/cli.py\", line 9' >&2\n" +
        "echo 'RuntimeError: CUDA out of memory' >&2\nexit 1",
    );

  test("auto: a model run that crashes falls back to the heuristic, and says why", async () => {
    process.env.RANKER_MODE = "auto";
    process.env.SLOTIFY_ML_PYTHON = crashing();
    const original = console.warn;
    console.warn = () => undefined;
    try {
      const result = await rankCandidates(request());
      assert.equal(result.source, "heuristic_offline_v1");
      assert.equal(result.mode, "heuristic");
      assert.equal(result.trainingLabelSource, null);
      assert.equal(result.warnings.length, 1);
      assert.match(result.warnings[0], /fell back to the heuristic baseline/);
      assert.match(result.warnings[0], /CUDA out of memory/);
      assert.ok(!result.warnings[0].includes("/srv/ml"), result.warnings[0]);
      assert.ok(result.ranked.length >= 1);
    } finally {
      console.warn = original;
    }
  });

  test("learned: the same crash fails loudly", async () => {
    process.env.RANKER_MODE = "learned";
    process.env.SLOTIFY_ML_PYTHON = crashing();
    await assert.rejects(rankCandidates(request()), LearnedRankerError);
  });

  test("a weak-label model is disclosed as a distillation of the heuristic", async () => {
    process.env.RANKER_MODE = "auto";
    const out = path.join(dir, "weak.json");
    fs.writeFileSync(out, JSON.stringify(payload("weak_heuristic")));
    process.env.SLOTIFY_ML_PYTHON = fakePython("weak.sh", `cat '${out}'`);

    const result = await rankCandidates(request());
    assert.equal(result.source, "learned_ranker");
    assert.equal(result.trainingLabelSource, "weak_heuristic");
    assert.ok(
      result.warnings.some((warning) =>
        /distillation of the heuristic baseline, trained without human labels/.test(warning) &&
        /not independent evidence/.test(warning),
      ),
      result.warnings.join(" | "),
    );
    // The exact-match analyser candidate keeps its detector.
    assert.equal(result.ranked[0].generatedBy, "analyser");
  });

  test("a disclosure the ranker already made is not repeated", async () => {
    process.env.RANKER_MODE = "auto";
    const out = path.join(dir, "weak-disclosed.json");
    const disclosure =
      "This ranking comes from a model that is a distillation of the heuristic " +
      "baseline (heuristic_offline_v1): it was trained on labels derived from that " +
      "baseline's own scores, without any human labels. Its ordering is therefore " +
      "not independent evidence of placement quality.";
    fs.writeFileSync(
      out,
      JSON.stringify({ ...payload("weak_heuristic"), warnings: [disclosure] }),
    );
    process.env.SLOTIFY_ML_PYTHON = fakePython("weak-disclosed.sh", `cat '${out}'`);

    const result = await rankCandidates(request());
    const mentions = result.warnings.filter((warning) => /distillation/.test(warning));
    assert.deepEqual(mentions, [disclosure]);
  });

  test("a human-label model carries no weak-label warning", async () => {
    process.env.RANKER_MODE = "auto";
    const out = path.join(dir, "human.json");
    fs.writeFileSync(out, JSON.stringify(payload("human")));
    process.env.SLOTIFY_ML_PYTHON = fakePython("human.sh", `cat '${out}'`);

    const result = await rankCandidates(request());
    assert.equal(result.trainingLabelSource, "human");
    assert.ok(!result.warnings.some((warning) => /distillation|label_source=/.test(warning)));
  });

  test("the heuristic path reports no training label source", async () => {
    process.env.RANKER_MODE = "heuristic";
    const result = await rankCandidates(request());
    assert.equal(result.trainingLabelSource, null);
    assert.equal(result.ranked[0].generatedBy, "analyser");
  });
});
