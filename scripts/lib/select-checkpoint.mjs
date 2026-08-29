/**
 * Which trained checkpoint the product should serve, and how confident it is.
 *
 * There are two kinds of checkpoint in `artifacts/training/`, and confusing them
 * is the most expensive mistake this repository can make:
 *
 *   label_source: "human"           trained on human judgements. This is the
 *                                   model an evaluation result describes, and
 *                                   the one production should serve.
 *   label_source: "weak_heuristic"  trained on targets derived from
 *                                   `heuristic_offline_v1`'s own score. A
 *                                   distillation of the baseline. It proves the
 *                                   architecture and the serving path work; it
 *                                   proves nothing about ranking quality.
 *
 * So the selection is not "the newest checkpoint" or "the one with the best
 * metric" -- it is "a human-trained one if any exists, and otherwise the
 * bootstrap, said out loud". Preferring the best metric would silently pick a
 * weak run that happened to score well against its own teacher.
 *
 * Every run summary is read from disk; nothing here is hard-coded except the
 * fallback path, which is the one checkpoint committed to the repository.
 */

import fs from "node:fs";
import path from "node:path";

/** The committed bootstrap. Weakly supervised, and labelled as such. */
export const BOOTSTRAP_CHECKPOINT = path.join(
  "artifacts",
  "training",
  "gated-d8ed976101aa4c3b",
  "best_checkpoint.pt",
);

const readJson = (file) => {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return null;
  }
};

/**
 * @param {string} repoRoot
 * @returns {{
 *   checkpoint: string | null,   // repo-relative, or null when nothing is usable
 *   labelSource: string | null,
 *   runId: string | null,
 *   variant: string | null,
 *   isHumanTrained: boolean,
 *   reason: string,
 * }}
 */
export const selectCheckpoint = (repoRoot) => {
  const trainingDir = path.join(repoRoot, "artifacts", "training");
  const runs = [];

  if (fs.existsSync(trainingDir)) {
    for (const entry of fs.readdirSync(trainingDir, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const dir = path.join(trainingDir, entry.name);
      const summary = readJson(path.join(dir, "training_summary.json"));
      if (!summary) continue;
      const checkpoint = path.join(dir, "best_checkpoint.pt");
      const normalizer = path.join(dir, "normalizer.json");
      if (!fs.existsSync(checkpoint) || !fs.existsSync(normalizer)) continue;
      runs.push({
        dir,
        checkpoint,
        labelSource: String(summary.label_source ?? ""),
        runId: summary.run_id ?? entry.name,
        variant: summary.model_variant ?? null,
        generatedAt: String(summary.generated_at ?? ""),
      });
    }
  }

  const human = runs
    .filter((run) => run.labelSource === "human")
    .sort((a, b) => a.generatedAt.localeCompare(b.generatedAt));

  if (human.length > 0) {
    const chosen = human[human.length - 1];
    return {
      checkpoint: path.relative(repoRoot, chosen.checkpoint),
      labelSource: chosen.labelSource,
      runId: chosen.runId,
      variant: chosen.variant,
      isHumanTrained: true,
      reason: `human-trained run ${chosen.runId} (${human.length} available; the most recent is served)`,
    };
  }

  const fallback = path.join(repoRoot, BOOTSTRAP_CHECKPOINT);
  if (fs.existsSync(fallback)) {
    const summary = readJson(
      path.join(path.dirname(fallback), "training_summary.json"),
    );
    return {
      checkpoint: BOOTSTRAP_CHECKPOINT,
      labelSource: String(summary?.label_source ?? "weak_heuristic"),
      runId: summary?.run_id ?? null,
      variant: summary?.model_variant ?? null,
      isHumanTrained: false,
      reason:
        "no human-trained checkpoint exists, so the committed weakly supervised " +
        "bootstrap is served -- it demonstrates the architecture and the serving " +
        "path, not ranking quality",
    };
  }

  return {
    checkpoint: null,
    labelSource: null,
    runId: null,
    variant: null,
    isHumanTrained: false,
    reason: "no usable checkpoint found; RANKER_MODE=auto will use the heuristic baseline",
  };
};
