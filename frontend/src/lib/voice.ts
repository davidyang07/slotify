/**
 * Releasing the voice the server cloned for the current file.
 *
 * Each clone occupies a slot in the operator's ElevenLabs account until it is
 * deleted. The server deletes idle clones on its own (SLOTIFY_VOICE_TTL_MIN),
 * but the UI knows sooner when one is finished with: when the file is replaced
 * and when the page goes away. No React here, so it runs under `node --test`.
 */

import type { ApiFetch } from "./api";

export const voiceReleasePath = (voiceId: string): string =>
  `/api/clone/${encodeURIComponent(voiceId)}/release`;

/**
 * Tell the server it may delete `voiceId` now. Fire and forget: it never
 * rejects, since nothing waits on it. `keepalive` lets the request outlive the
 * page on `pagehide`; it is a fetch rather than `sendBeacon` because a beacon
 * cannot carry the Authorization header a token-protected API requires.
 */
export const releaseVoice = (apiFetch: ApiFetch, voiceId: string): Promise<void> =>
  apiFetch(voiceReleasePath(voiceId), { method: "POST", keepalive: true }).then(
    () => undefined,
    () => undefined,
  );
