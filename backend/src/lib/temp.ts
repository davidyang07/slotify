import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// Every per-request dir that has not been removed yet, so shutdown can remove
// what an interrupted request's `finally` did not get to.
const liveDirs = new Set<string>();

export const makeTempDir = async (prefix: string): Promise<string> => {
  const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), prefix));
  liveDirs.add(dir);
  return dir;
};

/** Remove a temp dir and everything in it; never throws. */
export const removeTempDir = async (dir: string | null | undefined): Promise<void> => {
  if (!dir) return;
  await fs.promises.rm(dir, { recursive: true, force: true }).catch(() => undefined);
  liveDirs.delete(dir);
};

/** Temp dirs made by `makeTempDir` and not yet removed. */
export const liveTempDirs = (): string[] => [...liveDirs];

/**
 * Remove every remaining temp dir, synchronously: the last step before the
 * process exits, after children have been stopped, when an async `rm` could be
 * cut off by `process.exit`.
 */
export const removeAllTempDirsSync = (): void => {
  for (const dir of liveDirs) {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch {
      // Best effort; the OS temp cleaner is the backstop.
    }
    liveDirs.delete(dir);
  }
};
