import assert from "node:assert/strict";
import { test, describe } from "node:test";

import { TaskScope } from "./tasks";

describe("work for a previous file cannot land on the next one", () => {
  test("a task begun before reset is stale and aborted", () => {
    const scope = new TaskScope();
    const renderOfA = scope.begin();
    assert.equal(renderOfA.isCurrent(), true);
    scope.reset(); // file B uploaded
    assert.equal(renderOfA.isCurrent(), false);
    assert.equal(renderOfA.signal.aborted, true);
    assert.equal(renderOfA.abortReason(), "superseded");
    renderOfA.finish();
  });

  test("a task begun after reset is current", () => {
    const scope = new TaskScope();
    scope.begin().finish();
    scope.reset();
    const renderOfB = scope.begin();
    assert.equal(renderOfB.isCurrent(), true);
    assert.equal(renderOfB.signal.aborted, false);
    renderOfB.finish();
  });

  test("a finished task is detached from later resets", () => {
    const scope = new TaskScope();
    const task = scope.begin();
    task.finish();
    scope.reset();
    assert.equal(task.signal.aborted, false);
    assert.equal(task.isCurrent(), false);
  });
});

describe("cancel and timeout", () => {
  test("cancel aborts with a reason, and the task stays current", () => {
    const scope = new TaskScope();
    const task = scope.begin(60_000);
    task.cancel();
    assert.equal(task.signal.aborted, true);
    assert.equal(task.abortReason(), "cancelled");
    assert.equal(task.isCurrent(), true);
    task.finish();
  });

  test("a timeout aborts with its own reason", async () => {
    const scope = new TaskScope();
    const task = scope.begin(5);
    await new Promise((resolve) => setTimeout(resolve, 30));
    assert.equal(task.signal.aborted, true);
    assert.equal(task.abortReason(), "timeout");
    task.finish();
  });

  test("the first reason wins", () => {
    const scope = new TaskScope();
    const task = scope.begin();
    task.cancel();
    scope.reset();
    assert.equal(task.abortReason(), "cancelled");
    task.finish();
  });
});
