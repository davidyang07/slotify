/** Tones and languages the UI offers for sponsor reads the server writes. */

export const SPONSOR_TONES = [
  "professional",
  "friendly-casual",
  "energetic",
  "serious",
  "warm",
  "conversational",
  "enthusiastic",
] as const;
export type SponsorTone = (typeof SPONSOR_TONES)[number];

export const SPONSOR_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"] as const;
export type SponsorLanguage = (typeof SPONSOR_LANGUAGES)[number];

/** How each tone is described to the copywriting model. */
export const TONE_DIRECTIONS: Record<SponsorTone, string> = {
  professional: "polished and professional",
  "friendly-casual": "friendly and casual",
  energetic: "energetic and upbeat",
  serious: "serious and measured",
  warm: "warm and personal",
  conversational: "natural and conversational",
  enthusiastic: "enthusiastic and excited",
};

export const LANGUAGE_NAMES: Record<SponsorLanguage, string> = {
  en: "English",
  es: "Spanish",
  fr: "French",
  de: "German",
  it: "Italian",
  pt: "Portuguese",
  ja: "Japanese",
  ko: "Korean",
  zh: "Chinese (Simplified)",
};

export interface SponsorCopyOptions {
  tone?: SponsorTone;
  language?: SponsorLanguage;
}
