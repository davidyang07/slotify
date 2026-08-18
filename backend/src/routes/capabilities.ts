/**
 * GET /api/capabilities -- what this deployment can actually do right now.
 *
 * The UI reads this on load so it can disable what is unavailable instead of
 * offering a button that fails on click. It also means the core placement demo
 * is visibly independent of every paid API: with no keys at all, `placement`
 * is true and everything else is false.
 */

import { Router } from "express";
import { resolveRankerMode } from "../lib/ranker-mode";
import { BASELINE_VERSION, HEURISTIC_CONFIG_VERSION } from "../lib/heuristic-config";

export const capabilitiesRouter = Router();

export interface Capabilities {
  /** Candidate detection + ranking. Requires only Python and ffmpeg. */
  placement: boolean;
  voiceCloning: boolean;
  tts: boolean;
  /** Optional OpenAI enrichment: sponsor copy and slot narration. */
  openaiEnhancement: boolean;
  learnedRanker: boolean;
  ranker: {
    requested: string;
    active: "learned" | "heuristic";
    reason: string;
    baselineVersion: string;
    heuristicConfigVersion: string;
  };
}

export const readCapabilities = (): Capabilities => {
  let ranker: Capabilities["ranker"];
  let learnedRanker = false;
  try {
    const resolved = resolveRankerMode();
    learnedRanker = resolved.mode === "learned";
    ranker = {
      requested: resolved.requested,
      active: resolved.mode,
      reason: resolved.reason,
      baselineVersion: BASELINE_VERSION,
      heuristicConfigVersion: HEURISTIC_CONFIG_VERSION,
    };
  } catch (error) {
    // RANKER_MODE=learned with nothing usable. Report it rather than silently
    // degrading: the operator asked for the model and did not get it.
    ranker = {
      requested: String(process.env.RANKER_MODE ?? "auto"),
      active: "heuristic",
      reason: error instanceof Error ? error.message : String(error),
      baselineVersion: BASELINE_VERSION,
      heuristicConfigVersion: HEURISTIC_CONFIG_VERSION,
    };
  }

  const elevenlabs = Boolean(process.env.ELEVENLABS_API_KEY);
  return {
    placement: true,
    voiceCloning: elevenlabs,
    tts: elevenlabs,
    openaiEnhancement: Boolean(process.env.OPENAI_API_KEY),
    learnedRanker,
    ranker,
  };
};

capabilitiesRouter.get("/api/capabilities", (_req, res) => {
  res.json(readCapabilities());
});
