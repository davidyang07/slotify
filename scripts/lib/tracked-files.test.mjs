import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, test } from "node:test";

import { trackedFiles } from "./tracked-files.mjs";

const root = fs.mkdtempSync(path.join(os.tmpdir(), "slotify-tracked-"));
after(() => fs.rmSync(root, { recursive: true, force: true }));

const git = (...args) => execFileSync("git", args, { cwd: root, stdio: "pipe" });

test("reports the size of the staged blob, not of the working-tree file", () => {
  git("init", "-q");
  fs.mkdirSync(path.join(root, "assets"));
  fs.writeFileSync(path.join(root, "assets", "video file é.mp4"), Buffer.alloc(4096));
  fs.writeFileSync(path.join(root, "README.md"), "hello\n");
  fs.writeFileSync(path.join(root, "untracked.bin"), Buffer.alloc(10));
  git("add", "assets", "README.md");
  // Shrunk on disk but not re-staged: what a commit would carry is still 4 KiB.
  fs.writeFileSync(path.join(root, "assets", "video file é.mp4"), "");

  const sizes = Object.fromEntries(trackedFiles(root).map((f) => [f.path, f.size]));
  assert.deepEqual(sizes, {
    "README.md": 6,
    // -z keeps a path with a space and a non-ASCII character as itself.
    "assets/video file é.mp4": 4096,
  });
});
