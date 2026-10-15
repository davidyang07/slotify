import assert from "node:assert/strict";
import { describe, test } from "node:test";
import type { Request } from "express";

import { internalToken } from "../config";
import { JobLimiter, refuseWhenBusy, runHeavyJob, BUSY_RETRY_AFTER_SECONDS } from "./concurrency";
import { ClientDisconnectedError, HttpError } from "./http";

/** A job that runs until `finish` is called. */
const gate = () => {
  let finish!: () => void;
  const done = new Promise<void>((resolve) => {
    finish = resolve;
  });
  return { finish, job: () => done };
};

const tick = () => new Promise((resolve) => setImmediate(resolve));

const isBusy = (error: unknown) =>
  error instanceof HttpError &&
  error.status === 503 &&
  error.headers["Retry-After"] === String(BUSY_RETRY_AFTER_SECONDS);

const fakeRequest = (authorization?: string) =>
  ({ headers: authorization ? { authorization } : {} }) as unknown as Request;

describe("JobLimiter", () => {
  test("runs up to the limit at once and queues the rest in order", async () => {
    const limiter = new JobLimiter({ maxConcurrent: 2, maxQueued: 5 });
    const gates = [gate(), gate(), gate()];
    const started: number[] = [];
    const runs = gates.map((entry, index) =>
      limiter.run(async () => {
        started.push(index);
        await entry.job();
      }),
    );
    await tick();
    assert.deepEqual(started, [0, 1]);
    assert.equal(limiter.running, 2);
    assert.equal(limiter.waiting, 1);

    gates[0].finish();
    await tick();
    assert.deepEqual(started, [0, 1, 2]);
    gates[1].finish();
    gates[2].finish();
    await Promise.all(runs);
    assert.equal(limiter.running, 0);
    assert.equal(limiter.waiting, 0);
  });

  test("a full queue is a 503 with Retry-After, and nothing is started", async () => {
    const limiter = new JobLimiter({ maxConcurrent: 1, maxQueued: 1 });
    const first = gate();
    const second = gate();
    const running = limiter.run(first.job);
    const queued = limiter.run(second.job);
    await tick();
    assert.equal(limiter.isFull(), true);
    let ran = false;
    await assert.rejects(
      limiter.run(async () => {
        ran = true;
      }),
      isBusy,
    );
    assert.equal(ran, false);
    first.finish();
    second.finish();
    await Promise.all([running, queued]);
    assert.equal(limiter.isFull(), false);
  });

  test("a queued job whose client disconnects leaves the queue", async () => {
    const limiter = new JobLimiter({ maxConcurrent: 1, maxQueued: 1 });
    const first = gate();
    const running = limiter.run(first.job);
    const controller = new AbortController();
    let ran = false;
    const queued = limiter.run(
      async () => {
        ran = true;
      },
      { signal: controller.signal },
    );
    await tick();
    assert.equal(limiter.waiting, 1);
    controller.abort();
    await assert.rejects(queued, (error: unknown) => error instanceof ClientDisconnectedError);
    assert.equal(limiter.waiting, 0);
    // Its place is free for another job.
    const next = gate();
    const replacement = limiter.run(next.job);
    await tick();
    assert.equal(limiter.waiting, 1);
    first.finish();
    next.finish();
    await Promise.all([running, replacement]);
    assert.equal(ran, false);
  });

  test("an already-aborted job is never queued", async () => {
    const limiter = new JobLimiter({ maxConcurrent: 1, maxQueued: 1 });
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      limiter.run(async () => undefined, { signal: controller.signal }),
      (error: unknown) => error instanceof ClientDisconnectedError,
    );
    assert.equal(limiter.running, 0);
  });

  test("a failing job frees its slot", async () => {
    const limiter = new JobLimiter({ maxConcurrent: 1, maxQueued: 0 });
    await assert.rejects(limiter.run(async () => Promise.reject(new Error("boom"))), /boom/);
    assert.equal(limiter.running, 0);
    assert.equal(await limiter.run(async () => "ok"), "ok");
  });

  test("a lease frees its slot once, however often it is released", async () => {
    const limiter = new JobLimiter({ maxConcurrent: 1, maxQueued: 0 });
    const release = await limiter.lease();
    assert.equal(limiter.isFull(), true);
    release();
    release();
    assert.equal(limiter.running, 0);
    const again = await limiter.lease();
    assert.equal(limiter.running, 1);
    again();
  });
});

describe("heavy-job admission", () => {
  test("our own pipelines' callbacks bypass the limiter, so a full one cannot deadlock", async () => {
    const limiter = new JobLimiter({ maxConcurrent: 1, maxQueued: 0 });
    const held = gate();
    const pipeline = limiter.run(held.job);
    await tick();
    const callback = await runHeavyJob(
      fakeRequest(`Bearer ${internalToken}`),
      async () => "merged",
      { limiter },
    );
    assert.equal(callback, "merged");
    await assert.rejects(runHeavyJob(fakeRequest(), async () => "x", { limiter }), isBusy);
    held.finish();
    await pipeline;
  });

  test("a full limiter refuses before the upload is read", async () => {
    const limiter = new JobLimiter({ maxConcurrent: 1, maxQueued: 0 });
    const middleware = refuseWhenBusy({ limiter, extra: { slots: [] } });
    /** "next", or the refusal the middleware sent. */
    const outcome = (req: Request) =>
      new Promise<unknown>((resolve) => {
        const headers: Record<string, string> = {};
        const res = {
          setHeader: (name: string, value: string) => {
            headers[name] = value;
          },
          status: (status: number) => ({
            json: (body: unknown) => resolve({ status, headers, body }),
          }),
        };
        middleware(req, res as never, () => resolve("next"));
      });

    assert.equal(await outcome(fakeRequest()), "next");
    const held = gate();
    const running = limiter.run(held.job);
    await tick();
    const refused = (await outcome(fakeRequest())) as { status: number; headers: Record<string, string>; body: any };
    assert.equal(refused.status, 503);
    assert.equal(refused.headers["Retry-After"], String(BUSY_RETRY_AFTER_SECONDS));
    assert.deepEqual(refused.body.slots, []);
    assert.match(refused.body.error, /busy/);
    assert.equal(await outcome(fakeRequest(`Bearer ${internalToken}`)), "next");
    held.finish();
    await running;
  });
});
