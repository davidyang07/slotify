import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { VoiceRegistry } from "./voice-registry";

const setup = ({ ttlMs = 60_000, maxLive = 3 } = {}) => {
  let clock = 1_000;
  let created = 0;
  const deleted: string[] = [];
  const registry = new VoiceRegistry({
    deleteVoice: async (voiceId) => {
      deleted.push(voiceId);
    },
    ttlMs,
    maxLive,
    now: () => clock,
  });
  const create = async () => `voice-${++created}`;
  return {
    registry,
    deleted,
    create,
    createdCount: () => created,
    advance: (ms: number) => {
      clock += ms;
    },
  };
};

/** Clone and deliver, as a successful /api/clone does. */
const cloned = async (registry: VoiceRegistry, hash: string | null, create: () => Promise<string>) => {
  const { voiceId, reused } = await registry.acquire(hash, create);
  registry.delivered(voiceId);
  return { voiceId, reused };
};

describe("voice registry", () => {
  test("the same audio reuses the live voice instead of cloning again", async () => {
    const { registry, create, createdCount } = setup();
    const first = await cloned(registry, "hash-a", create);
    const second = await cloned(registry, "hash-a", create);
    assert.equal(second.voiceId, first.voiceId);
    assert.equal(first.reused, false);
    assert.equal(second.reused, true);
    assert.equal(createdCount(), 1);
    const other = await cloned(registry, "hash-b", create);
    assert.notEqual(other.voiceId, first.voiceId);
  });

  test("concurrent clones of the same audio share one creation", async () => {
    const { registry, createdCount } = setup();
    let finish!: (id: string) => void;
    const slowCreate = () =>
      new Promise<string>((resolve) => {
        finish = resolve;
      });
    const one = registry.acquire("hash-a", slowCreate);
    const two = registry.acquire("hash-a", slowCreate);
    await new Promise((resolve) => setImmediate(resolve));
    finish("voice-x");
    assert.deepEqual((await one).voiceId, "voice-x");
    assert.deepEqual((await two).voiceId, "voice-x");
    assert.equal(createdCount(), 0);
  });

  test("a pipeline's own clone (no hash) is never shared", async () => {
    const { registry, create } = setup();
    const one = await cloned(registry, null, create);
    const two = await cloned(registry, null, create);
    assert.notEqual(one.voiceId, two.voiceId);
  });

  test("idle voices are deleted after the TTL; used ones are kept", async () => {
    const { registry, create, deleted, advance } = setup({ ttlMs: 30_000 });
    const idle = await cloned(registry, "a", create);
    const busy = await cloned(registry, "b", create);
    advance(20_000);
    assert.equal(registry.touch(busy.voiceId), true);
    advance(15_000);
    assert.equal(await registry.sweep(), 1);
    assert.deepEqual(deleted, [idle.voiceId]);
    assert.equal(registry.has(busy.voiceId), true);
    assert.equal(registry.wasReleased(idle.voiceId), true);
  });

  test("past the cap, the least recently used voice is deleted first", async () => {
    const { registry, create, deleted, advance } = setup({ maxLive: 2 });
    const a = await cloned(registry, "a", create);
    advance(10);
    const b = await cloned(registry, "b", create);
    advance(10);
    registry.touch(a.voiceId);
    advance(10);
    const c = await cloned(registry, "c", create);
    assert.deepEqual(deleted, [b.voiceId]);
    assert.equal(registry.size, 2);
    assert.ok(registry.has(a.voiceId) && registry.has(c.voiceId));
  });

  test("a voice whose client left before receiving it is deleted, unless someone else got it", async () => {
    const { registry, create, deleted } = setup();
    const lost = await registry.acquire("a", create);
    await registry.abandoned(lost.voiceId);
    assert.deepEqual(deleted, [lost.voiceId]);

    const kept = await cloned(registry, "b", create);
    const again = await registry.acquire("b", create);
    await registry.abandoned(again.voiceId);
    assert.deepEqual(deleted, [lost.voiceId]);
    assert.equal(registry.has(kept.voiceId), true);
  });

  test("release is idempotent and never touches a voice it did not create", async () => {
    const { registry, create, deleted } = setup();
    const { voiceId } = await cloned(registry, "a", create);
    assert.equal(await registry.release(voiceId), "released");
    assert.equal(await registry.release(voiceId), "already-released");
    assert.equal(await registry.release("someone-elses-voice"), "unknown");
    assert.equal(registry.touch("someone-elses-voice"), false);
    assert.deepEqual(deleted, [voiceId]);
    // Released audio is cloned afresh next time.
    const next = await cloned(registry, "a", create);
    assert.notEqual(next.voiceId, voiceId);
  });

  test("shutdown deletes every voice", async () => {
    const { registry, create, deleted } = setup();
    const ids = [
      (await cloned(registry, "a", create)).voiceId,
      (await cloned(registry, "b", create)).voiceId,
    ];
    await registry.releaseAll();
    assert.deepEqual([...deleted].sort(), [...ids].sort());
    assert.equal(registry.size, 0);
  });

  test("a failed upstream delete is logged, not thrown", async () => {
    const registry = new VoiceRegistry({
      deleteVoice: async () => {
        throw new Error("network down");
      },
      ttlMs: 1,
      maxLive: 3,
    });
    const { voiceId } = await registry.acquire("a", async () => "v1");
    const original = console.error;
    console.error = () => undefined;
    try {
      assert.equal(await registry.release(voiceId), "released");
    } finally {
      console.error = original;
    }
    assert.equal(registry.has(voiceId), false);
  });
});
