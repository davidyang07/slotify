/**
 * Process shutdown hooks.
 *
 * Anything that owns a long-lived resource the OS will not clean up for us -- a
 * child process, a persistent model worker, a cloned voice -- registers a hook
 * here, and the SIGTERM/SIGINT handler in index.ts runs them before exiting.
 * Hooks run concurrently and are bounded by one shared deadline, so a hook that
 * hangs cannot keep the process alive.
 */

export type ShutdownHook = () => void | Promise<void>;

const hooks = new Set<ShutdownHook>();
let shuttingDown = false;

/** Register a hook; returns a function that unregisters it. */
export const onShutdown = (hook: ShutdownHook): (() => void) => {
  hooks.add(hook);
  return () => {
    hooks.delete(hook);
  };
};

/** True once shutdown has begun: new long-running work should be refused. */
export const isShuttingDown = (): boolean => shuttingDown;

/**
 * Run every hook once, concurrently, giving up after `deadlineMs`. A failing
 * hook is logged, never thrown, so one bad hook cannot skip the others.
 */
export const runShutdownHooks = async (deadlineMs: number): Promise<void> => {
  if (shuttingDown) return;
  shuttingDown = true;
  const pending = [...hooks].map(async (hook) => {
    try {
      await hook();
    } catch (error) {
      console.error("Shutdown hook failed.", error);
    }
  });
  let timer: NodeJS.Timeout | undefined;
  const deadline = new Promise<void>((resolve) => {
    timer = setTimeout(resolve, deadlineMs);
    timer.unref();
  });
  await Promise.race([Promise.all(pending).then(() => undefined), deadline]);
  if (timer) clearTimeout(timer);
};
