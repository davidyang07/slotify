/**
 * The ElevenLabs voices this process cloned, and their deletion.
 *
 * Every new file and every page reload used to clone a permanent voice into the
 * operator's ElevenLabs account, and nothing ever deleted one; a clone whose
 * client had gone was simply lost. The registry makes each clone a lease:
 *
 *   - the same audio (by sha256) reuses the live voice instead of cloning again;
 *   - a voice idle for SLOTIFY_VOICE_TTL_MIN is deleted;
 *   - at most SLOTIFY_MAX_LIVE_VOICES are kept, least recently used deleted first;
 *   - a voice whose client disconnected before receiving it is deleted at once;
 *   - POST /api/clone/:voiceId/release deletes one early, and shutdown deletes all.
 *
 * Only voices this registry created are ever deleted: a voice id a client
 * passes to /api/tts that the registry did not create is left alone.
 */

import { maxLiveVoices, voiceTtlMs } from "../config";

export interface VoiceRegistryOptions {
  /** Delete the voice upstream. A voice that is already gone is a success. */
  deleteVoice: (voiceId: string) => Promise<void>;
  ttlMs: number;
  maxLive: number;
  now?: () => number;
}

interface VoiceEntry {
  voiceId: string;
  audioHash: string | null;
  lastUsed: number;
  /** Responses that delivered this id to a client. */
  delivered: number;
  /** Requests holding the id that have not yet delivered or given up on it. */
  pending: number;
}

/** How many released ids are remembered, so /api/tts can say "expired". */
const RELEASED_MEMORY = 500;

export class VoiceRegistry {
  private readonly entries = new Map<string, VoiceEntry>();
  private readonly byHash = new Map<string, string>();
  private readonly creating = new Map<string, Promise<string>>();
  private readonly released = new Set<string>();
  private readonly now: () => number;
  private sweepTimer: NodeJS.Timeout | null = null;

  constructor(private readonly options: VoiceRegistryOptions) {
    this.now = options.now ?? Date.now;
  }

  get size(): number {
    return this.entries.size;
  }

  has(voiceId: string): boolean {
    return this.entries.has(voiceId);
  }

  /** True for a voice this registry created and has since deleted. */
  wasReleased(voiceId: string): boolean {
    return this.released.has(voiceId);
  }

  /** Mark a voice as used now. False when the registry does not own it. */
  touch(voiceId: string): boolean {
    const entry = this.entries.get(voiceId);
    if (!entry) return false;
    entry.lastUsed = this.now();
    return true;
  }

  /**
   * The voice for `audioHash`: the live one if this audio was cloned already,
   * otherwise the result of `create`. Concurrent requests for the same audio
   * share one creation. `audioHash` null (a pipeline's own clone, which it
   * deletes itself) always creates and is never shared. The caller must
   * follow with `delivered` or `abandoned` for the returned id.
   */
  async acquire(
    audioHash: string | null,
    create: () => Promise<string>,
  ): Promise<{ voiceId: string; reused: boolean }> {
    const liveId = audioHash ? this.byHash.get(audioHash) : undefined;
    if (liveId) {
      const entry = this.entries.get(liveId)!;
      entry.lastUsed = this.now();
      entry.pending += 1;
      return { voiceId: liveId, reused: true };
    }
    const inFlight = audioHash ? this.creating.get(audioHash) : undefined;
    if (inFlight) {
      const voiceId = await inFlight;
      const entry = this.entries.get(voiceId);
      if (entry) {
        entry.pending += 1;
        entry.lastUsed = this.now();
        return { voiceId, reused: true };
      }
      // Created and already discarded by its creator; clone afresh.
      return this.acquire(audioHash, create);
    }

    const creation = (async () => {
      // Make room first: the account's voice slots are the scarce thing.
      await this.evictToFit(this.options.maxLive - 1);
      const voiceId = await create();
      this.entries.set(voiceId, {
        voiceId,
        audioHash,
        lastUsed: this.now(),
        delivered: 0,
        pending: 1,
      });
      if (audioHash) this.byHash.set(audioHash, voiceId);
      this.released.delete(voiceId);
      return voiceId;
    })();
    if (audioHash) this.creating.set(audioHash, creation);
    try {
      return { voiceId: await creation, reused: false };
    } finally {
      if (audioHash) this.creating.delete(audioHash);
    }
  }

  /** The client received the id. */
  delivered(voiceId: string): void {
    const entry = this.entries.get(voiceId);
    if (!entry) return;
    entry.pending = Math.max(0, entry.pending - 1);
    entry.delivered += 1;
  }

  /**
   * The client went away before receiving the id. A voice no client ever
   * received, and no other request is about to hand out, is deleted now.
   */
  async abandoned(voiceId: string): Promise<void> {
    const entry = this.entries.get(voiceId);
    if (!entry) return;
    entry.pending = Math.max(0, entry.pending - 1);
    if (entry.delivered === 0 && entry.pending === 0) await this.release(voiceId);
  }

  /**
   * Delete a voice this registry created. Resolves to "released", or
   * "already-released" / "unknown" when there was nothing to delete, so
   * callers can be idempotent.
   */
  async release(voiceId: string): Promise<"released" | "already-released" | "unknown"> {
    const entry = this.entries.get(voiceId);
    if (!entry) return this.released.has(voiceId) ? "already-released" : "unknown";
    this.forget(entry);
    try {
      await this.options.deleteVoice(voiceId);
    } catch (error) {
      // Logged, not retried: the voice is out of the registry either way, and
      // a failed delete is visible in the ElevenLabs dashboard.
      console.error(`Could not delete cloned voice ${voiceId}.`, error);
    }
    return "released";
  }

  /** Delete every voice idle for longer than the TTL. */
  async sweep(): Promise<number> {
    const cutoff = this.now() - this.options.ttlMs;
    const idle = [...this.entries.values()].filter(
      (entry) => entry.lastUsed <= cutoff && entry.pending === 0,
    );
    await Promise.all(idle.map((entry) => this.release(entry.voiceId)));
    return idle.length;
  }

  /** Delete every voice; the shutdown hook. */
  async releaseAll(): Promise<void> {
    this.stop();
    await Promise.all([...this.entries.keys()].map((voiceId) => this.release(voiceId)));
  }

  /** Sweep on a timer that does not keep the process alive. */
  start(intervalMs = Math.max(1_000, Math.min(60_000, this.options.ttlMs / 2))): void {
    if (this.sweepTimer) return;
    this.sweepTimer = setInterval(() => {
      void this.sweep();
    }, intervalMs);
    this.sweepTimer.unref();
  }

  stop(): void {
    if (this.sweepTimer) clearInterval(this.sweepTimer);
    this.sweepTimer = null;
  }

  private forget(entry: VoiceEntry): void {
    this.entries.delete(entry.voiceId);
    if (entry.audioHash && this.byHash.get(entry.audioHash) === entry.voiceId) {
      this.byHash.delete(entry.audioHash);
    }
    this.released.add(entry.voiceId);
    if (this.released.size > RELEASED_MEMORY) {
      this.released.delete(this.released.values().next().value as string);
    }
  }

  /** Delete least-recently-used voices until at most `limit` remain. */
  private async evictToFit(limit: number): Promise<void> {
    const byAge = [...this.entries.values()].sort((a, b) => a.lastUsed - b.lastUsed);
    const excess = byAge.slice(0, Math.max(0, byAge.length - Math.max(0, limit)));
    await Promise.all(excess.map((entry) => this.release(entry.voiceId)));
  }
}

/** Configured from the environment by services/elevenlabs.ts. */
export const createVoiceRegistry = (deleteVoice: (voiceId: string) => Promise<void>) =>
  new VoiceRegistry({ deleteVoice, ttlMs: voiceTtlMs, maxLive: maxLiveVoices });
