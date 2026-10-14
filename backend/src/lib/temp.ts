import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const makeTempDir = (prefix: string): Promise<string> =>
  fs.promises.mkdtemp(path.join(os.tmpdir(), prefix));

/** Remove a temp dir and everything in it; never throws. */
export const removeTempDir = async (dir: string | null): Promise<void> => {
  if (!dir) return;
  await fs.promises.rm(dir, { recursive: true, force: true }).catch(() => undefined);
};
