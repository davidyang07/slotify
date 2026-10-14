/**
 * What git tracks, with the size of each tracked blob: the input
 * `hygieneFindings` judges.
 *
 * Sizes come from the object database -- blob ids from `git ls-files -s`, then
 * one `git cat-file --batch-check` for all of them -- not from the working tree.
 * The question is what a clone downloads, and a file shrunk on disk but never
 * re-staged is still its old size in the commit. On CI the index is the commit
 * under test; locally it is what the next commit would contain.
 */

import { execFileSync } from "node:child_process";

const git = (repoRoot, args, input) =>
  execFileSync("git", args, {
    cwd: repoRoot,
    encoding: "utf8",
    input,
    maxBuffer: 64 * 1024 * 1024,
  });

/**
 * @param {string} repoRoot
 * @returns {{path: string, size: number | null}[]}
 */
export const trackedFiles = (repoRoot) => {
  // -z: NUL-separated and unquoted, so a path with spaces or non-ASCII
  // characters arrives as itself rather than as a C-quoted string.
  const byPath = new Map();
  for (const record of git(repoRoot, ["ls-files", "-s", "-z"]).split("\0")) {
    const tab = record.indexOf("\t");
    if (tab === -1) continue;
    const [mode, object] = record.slice(0, tab).split(" ");
    // A path appears once per stage during a conflicted merge; one entry is enough.
    byPath.set(record.slice(tab + 1), { mode, object });
  }

  // Gitlinks (mode 160000) name a submodule commit, not a blob, so have no size.
  const blobIds = [
    ...new Set(
      [...byPath.values()].filter((e) => e.mode !== "160000").map((e) => e.object),
    ),
  ];
  const sizes = new Map();
  if (blobIds.length) {
    const report = git(
      repoRoot,
      ["cat-file", "--batch-check=%(objectname) %(objectsize)"],
      `${blobIds.join("\n")}\n`,
    );
    for (const line of report.split("\n")) {
      // A blob absent from the object database reads "<id> missing".
      const [object, size] = line.trim().split(" ");
      if (object && /^\d+$/.test(size ?? "")) sizes.set(object, Number(size));
    }
  }

  return [...byPath].map(([path, { object }]) => ({
    path,
    size: sizes.get(object) ?? null,
  }));
};
