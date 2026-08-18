/**
 * What the backend can do, fetched once so the UI can disable rather than fail.
 *
 * The important consequence: with no ElevenLabs key the app still uploads,
 * analyses and ranks. Only ad generation is switched off, and it says why.
 */

export interface Capabilities {
  placement: boolean;
  voiceCloning: boolean;
  tts: boolean;
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

/**
 * The assumption to hold before /api/capabilities answers: placement may work,
 * generation may not. Optimism about a paid API is what produces a button that
 * fails on click.
 */
export const UNKNOWN_CAPABILITIES: Capabilities = {
  placement: true,
  voiceCloning: false,
  tts: false,
  openaiEnhancement: false,
  learnedRanker: false,
  ranker: {
    requested: "auto",
    active: "heuristic",
    reason: "Capabilities have not been fetched yet.",
    baselineVersion: "heuristic_offline_v1",
    heuristicConfigVersion: "",
  },
};

export const fetchCapabilities = async (
  apiBase: string,
): Promise<Capabilities> => {
  const response = await fetch(`${apiBase}/api/capabilities`);
  if (!response.ok) {
    throw new Error(`Capabilities request failed: ${response.status}`);
  }
  return (await response.json()) as Capabilities;
};

/** Why ad generation is unavailable, or null when it is available. */
export const generationBlockedReason = (
  capabilities: Capabilities,
): string | null => {
  if (capabilities.tts && capabilities.voiceCloning) return null;
  return "Ad generation needs an ElevenLabs API key on the server. Placement analysis works without one.";
};
