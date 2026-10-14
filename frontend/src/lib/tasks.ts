/**
 * Async work that belongs to one uploaded file.
 *
 * Uploading a new file used to leave the previous file's requests running, and
 * whatever they resolved to was written into the new file's state: a render of
 * A finishing after B was uploaded showed A's merged audio as "Render
 * complete", and A's voice clone became the voice for B's ads. Each per-file
 * flow now begins a task from a `TaskScope`. `reset()` (on upload) aborts every
 * task's requests and makes its `isCurrent()` false, so nothing it resolves to
 * is written.
 */

export type TaskAbortReason = "cancelled" | "timeout" | "superseded";

export interface ScopedTask {
  /** Aborts on `cancel()`, on timeout, or when the scope is reset. */
  readonly signal: AbortSignal;
  /** False once the scope has been reset: the result belongs to an old file. */
  isCurrent(): boolean;
  cancel(): void;
  /** Why `signal` aborted, or null while it has not. */
  abortReason(): TaskAbortReason | null;
  /** Clear the timeout and detach from the scope. Call when the task settles. */
  finish(): void;
}

export class TaskScope {
  private generation = 0;
  private controller = new AbortController();

  /** Abandon everything begun so far. */
  reset(): void {
    this.generation += 1;
    this.controller.abort();
    this.controller = new AbortController();
  }

  begin(timeoutMs?: number): ScopedTask {
    const generation = this.generation;
    const scopeSignal = this.controller.signal;
    const controller = new AbortController();
    let reason: TaskAbortReason | null = null;
    const abort = (why: TaskAbortReason) => {
      if (controller.signal.aborted) return;
      reason = why;
      controller.abort();
    };
    const onScopeAbort = () => abort("superseded");
    scopeSignal.addEventListener("abort", onScopeAbort, { once: true });
    const timer =
      timeoutMs && timeoutMs > 0
        ? setTimeout(() => abort("timeout"), timeoutMs)
        : null;

    return {
      signal: controller.signal,
      isCurrent: () => generation === this.generation,
      cancel: () => abort("cancelled"),
      abortReason: () => reason,
      finish: () => {
        if (timer !== null) clearTimeout(timer);
        scopeSignal.removeEventListener("abort", onScopeAbort);
      },
    };
  }
}
