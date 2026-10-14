/**
 * Pure rules behind the one-screen workspace: what a break row says about
 * itself, what "Add a break at mm:ss" accepts, and what the Render button says
 * and why it is off. No React, so `workspace.test.ts` runs it under node.
 */

import { formatTimestamp } from "./format";
import type { PlacementSignal, Slot } from "./recommendations";

export interface Sponsor {
  id: string;
  name: string;
  script: string;
}

/** The tone and language values `/api/tts` and `/api/insert-sections` accept. */
export const TONES = [
  { value: "professional", label: "Professional" },
  { value: "friendly-casual", label: "Friendly & Casual" },
  { value: "energetic", label: "Energetic" },
  { value: "serious", label: "Serious" },
  { value: "warm", label: "Warm" },
  { value: "conversational", label: "Conversational" },
  { value: "enthusiastic", label: "Enthusiastic" },
];

export const LANGUAGES = [
  { value: "en", label: "English" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "it", label: "Italian" },
  { value: "pt", label: "Portuguese" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "zh", label: "Chinese" },
];

/** Sponsors per episode; ids come from a counter, so a removed one is never reused. */
export const MAX_SPONSORS = 3;

export type GenerationMode = "preview" | "render";

/**
 * What a preview or render said, and where to say it: a preview's outcome
 * belongs to the row that asked for it, a render's to the ad panel.
 */
export interface GenerationNote {
  mode: GenerationMode;
  /** The break row a preview note belongs to; null for a render. */
  slotId: string | null;
  tone: "ok" | "error" | "info";
  text: string;
}

export interface ReasonTag {
  text: string;
  kind: PlacementSignal["kind"] | "manual";
}

/** Shortest honest wording for each signal label the backend emits. */
const SHORT_LABELS: [RegExp, (match: RegExpMatchArray) => string][] = [
  [/^pause of ([\d.]+)\s*s\b/i, (m) => `pause ${m[1]}s`],
  [/^mid-episode/i, () => "mid-episode"],
  [/^falls on a sentence boundary/i, () => "sentence end"],
  [/^cuts mid-sentence/i, () => "mid-sentence"],
  [/^no transcript context/i, () => "no transcript"],
  [/^transcript context available/i, () => "transcript context"],
  [/^beat-aligned low-energy valley/i, () => "beat-aligned valley"],
  [/^low-energy valley/i, () => "low-energy valley"],
  [/^transcript sentence end, not beat-aligned/i, () => "not beat-aligned"],
  [/^very close to the start/i, () => "near the start"],
  [/^very close to the end/i, () => "near the end"],
  [/^short pause may feel abrupt/i, () => "abrupt pause"],
];

/**
 * One signal label as a reason tag: a known label gets its short form, an
 * unknown one is kept in its own words (lower-cased, without a trailing
 * parenthetical) rather than dropped or guessed at.
 */
export const shortSignalLabel = (label: string): string => {
  const text = label.trim();
  for (const [pattern, shorten] of SHORT_LABELS) {
    const match = text.match(pattern);
    if (match) return shorten(match);
  }
  const plain = text.replace(/\s*\([^)]*\)\s*$/, "").replace(/\.$/, "");
  return plain.charAt(0).toLowerCase() + plain.slice(1);
};

/** Says nothing specific about this break, so it only fills an empty row. */
const isLowInformationCaution = (label: string) => /^no transcript context/i.test(label.trim());

/**
 * The 2-3 reason tags a break row shows (DESIGN.md `reason-tag`), e.g.
 * `pause 1.2s · mid-episode`.
 *
 * Supporting signals come first, in the server's order. Cautions fill the row
 * up to three, so "near the end" is visible without expanding it; the
 * transcript caution every heuristic slot carries is used only when the row
 * would otherwise have fewer than two tags. The full list, every caution
 * included, is in the row's expandable details.
 */
export const reasonTags = (slot: Pick<Slot, "source" | "signals">, max = 3): ReasonTag[] => {
  if (slot.source === "manual") return [{ text: "added by you", kind: "manual" }];
  const supporting = slot.signals.filter((signal) => signal.kind === "supporting");
  const cautions = slot.signals.filter((signal) => signal.kind === "caution");
  const tags: ReasonTag[] = supporting
    .slice(0, max)
    .map((signal) => ({ text: shortSignalLabel(signal.label), kind: "supporting" }));
  const ordered = [
    ...cautions.filter((signal) => !isLowInformationCaution(signal.label)),
    ...cautions.filter((signal) => isLowInformationCaution(signal.label)),
  ];
  for (const signal of ordered) {
    if (tags.length >= max) break;
    if (isLowInformationCaution(signal.label) && tags.length >= 2) break;
    tags.push({ text: shortSignalLabel(signal.label), kind: "caution" });
  }
  return tags;
};

export type BreakTimeResult = { seconds: number } | { error: string };

/**
 * Read "Add a break at mm:ss". Seconds run 0-59; the time cannot be past the
 * end of the episode when its duration is known.
 */
export const parseBreakTime = (
  minutes: string,
  seconds: string,
  durationSeconds: number | null,
): BreakTimeResult => {
  const m = minutes.trim();
  const s = seconds.trim();
  if (!m && !s) return { error: "Enter a time in minutes and seconds." };
  if ((m && !/^\d+$/.test(m)) || (s && !/^\d+$/.test(s))) {
    return { error: "Use whole minutes and seconds, like 1:05." };
  }
  const mins = m ? Number(m) : 0;
  const secs = s ? Number(s) : 0;
  if (secs > 59) return { error: "Seconds run from 0 to 59." };
  const total = mins * 60 + secs;
  if (
    typeof durationSeconds === "number" &&
    Number.isFinite(durationSeconds) &&
    durationSeconds > 0 &&
    total > durationSeconds
  ) {
    return {
      error: `That is past the end of the episode (${formatTimestamp(durationSeconds)}).`,
    };
  }
  return { seconds: total };
};

/** "Render 1 ad break", "Render 2 ad breaks", "Rendering 2 ad breaks…". */
export const renderButtonLabel = (selectedCount: number, rendering: boolean): string => {
  const count = Math.max(0, Math.floor(selectedCount));
  const noun = count === 1 ? "ad break" : "ad breaks";
  if (rendering) return count ? `Rendering ${count} ${noun}…` : "Rendering…";
  return count ? `Render ${count} ${noun}` : "Render";
};

/**
 * Why Render is off, in the words shown under it, or null when it is on.
 * The capability reason wins: no amount of selecting fixes a missing key.
 */
export const renderBlockedReason = ({
  generationBlocked,
  selectedCount,
  hasSponsorName,
  analyzing,
  previewing,
}: {
  generationBlocked: string | null;
  selectedCount: number;
  hasSponsorName: boolean;
  analyzing: boolean;
  previewing: boolean;
}): string | null => {
  if (generationBlocked) return generationBlocked;
  if (previewing) return "Wait for the preview to finish.";
  if (selectedCount === 0) {
    return analyzing
      ? "Break points are still being found."
      : "Select at least one break point to render.";
  }
  if (!hasSponsorName) return "Add a sponsor name to render.";
  return null;
};
