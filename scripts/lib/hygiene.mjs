/**
 * Repository hygiene, as a function over a list of tracked files.
 *
 * Pure so it can be tested without a checkout. `npm run verify:ml` and
 * `scripts/check-hygiene.mjs` feed it `git ls-files` together with the size of
 * each staged blob (see `tracked-files.mjs`), so it judges what is COMMITTED --
 * not what happens to be sitting in a working tree. An editor's scratch file
 * nobody committed is not a repository problem; a committed one is, and stays
 * one forever.
 *
 * Each rule names a class of file that should never be in the history:
 *
 * - OS and editor metadata, which is noise in every clone but the one that made it;
 * - caches and compiled bytecode, which are regenerated and go stale;
 * - credentials, which cannot be un-published once pushed;
 * - audio and model binaries, because this corpus is reconstructed from a
 *   committed source registry rather than stored, and a checkout that carries
 *   gigabytes of it stops being clonable;
 * - anything over MAX_TRACKED_BYTES, whatever its name. The name-based rules
 *   cannot see that an exempt file has grown: a 71 MB background video passed
 *   all of them.
 *
 * The media rules have two deliberate exemptions, both product code rather than
 * data: `backend/audio_tests/` holds the small sample clips the product's own
 * manual and CLI testing needs, and `frontend/src/assets/` holds the UI's own
 * media. Neither is corpus, and neither is regenerable from a source registry.
 * The exemption covers audio, video and image files only -- a `.bin` or a `.zip`
 * in either directory is a heavy binary like anywhere else -- and never the size
 * rule.
 */

/** @typedef {{path: string, rule: string, reason: string}} Finding */
/**
 * A bare path, or a path with the size in bytes of its committed blob. Without
 * a size the size rule cannot apply, so callers judging a real checkout pass one.
 * @typedef {string | {path: string, size?: number | null}} TrackedFile
 */

const OS_METADATA = /(^|\/)(\.DS_Store|Thumbs\.db|desktop\.ini|\._[^/]+)$/i;
const EDITOR_SWAP = /(^|\/)([^/]+\.(swp|swo|orig|rej|bak)|.*~)$/i;
const CACHE = /(^|\/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|node_modules|\.venv|htmlcov)(\/|$)/;
const COMPILED = /\.(pyc|pyo)$/i;
const COVERAGE = /(^|\/)\.coverage(\.|$)/;
// `.env` and `.env.<anything>`, except the conventional committed templates
// (`.env.example`, `.env.sample`, `.env.template`), which hold variable names
// and placeholders rather than values -- flagging them would teach people to
// stop documenting configuration.
const CREDENTIAL =
  /(^|\/)(\.env(\.(?!(example|sample|template)$)[^/]+)?|.*\.pem|.*\.p12|id_rsa|.*\.keystore)$/i;
const AUDIO = /\.(mp3|wav|flac|m4a|ogg|opus|aac|mp4|mkv)$/i;
const IMAGE = /\.(png|jpe?g|gif|svg|webp|avif|ico)$/i;
const MEDIA_DIRS = /^(backend\/audio_tests|frontend\/src\/assets)\//;
const HEAVY_BINARY = /\.(pt|onnx|bin|ckpt|safetensors|npy|npz|parquet|zip|tar|gz)$/i;

/** Product media: an audio, video or image file in one of the two media directories. */
const isProductMedia = (p) => MEDIA_DIRS.test(p) && (AUDIO.test(p) || IMAGE.test(p));

/**
 * The one committed checkpoint the demo serves -- the path `.gitignore`
 * re-includes and `select-checkpoint.mjs` falls back to -- is exempt. Any other
 * committed weight file is a finding, including another run's
 * `best_checkpoint.pt`.
 */
export const COMMITTED_CHECKPOINT =
  "artifacts/training/gated-d8ed976101aa4c3b/best_checkpoint.pt";

/**
 * GitHub warns at 50 MB and refuses at 100 MB; this is well under both because
 * every clone pays for every byte forever. The largest legitimate files -- the
 * committed checkpoint (~6 MB) and the background video (~2.4 MB) -- fit under
 * it with room to spare.
 */
export const MAX_TRACKED_BYTES = 10 * 1024 * 1024;

const megabytes = (bytes) => `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

const RULES = [
  { rule: "os-metadata", test: (p) => OS_METADATA.test(p), reason: "operating-system metadata" },
  { rule: "editor-scratch", test: (p) => EDITOR_SWAP.test(p), reason: "editor swap or backup file" },
  { rule: "cache", test: (p) => CACHE.test(p), reason: "a regenerated cache directory" },
  { rule: "compiled", test: (p) => COMPILED.test(p), reason: "compiled Python bytecode" },
  { rule: "coverage", test: (p) => COVERAGE.test(p), reason: "a coverage database" },
  { rule: "credential", test: (p) => CREDENTIAL.test(p), reason: "a credential or private key" },
  {
    rule: "audio-corpus",
    test: (p) => AUDIO.test(p) && !isProductMedia(p),
    reason:
      "audio or video outside backend/audio_tests/ and frontend/src/assets/; " +
      "the corpus is reconstructed from its source registry, never stored",
  },
  {
    rule: "heavy-binary",
    // No media exemption: no heavy-binary extension is a media one.
    test: (p) => HEAVY_BINARY.test(p) && p !== COMMITTED_CHECKPOINT,
    reason: "a large binary with no documented reason to be committed",
  },
  {
    // Last, so a file that also breaks a more specific rule is reported under
    // that one. Applies to exempt files too.
    rule: "oversized",
    test: (p, size) => typeof size === "number" && size > MAX_TRACKED_BYTES,
    reason: (p, size) =>
      `${megabytes(size)} committed, over the ${megabytes(MAX_TRACKED_BYTES)} limit; ` +
      "every clone pays for it forever",
  },
];

/**
 * @param {TrackedFile[]} trackedFiles repository-relative, forward slashes.
 * @returns {Finding[]}
 */
export function hygieneFindings(trackedFiles) {
  /** @type {Finding[]} */
  const findings = [];
  for (const entry of trackedFiles) {
    const raw = typeof entry === "string" ? entry : entry?.path ?? "";
    const size = typeof entry === "string" ? null : entry?.size ?? null;
    const filePath = String(raw).trim().replaceAll("\\", "/");
    if (!filePath) continue;
    for (const rule of RULES) {
      if (rule.test(filePath, size)) {
        const reason =
          typeof rule.reason === "function" ? rule.reason(filePath, size) : rule.reason;
        findings.push({ path: filePath, rule: rule.rule, reason });
        break;
      }
    }
  }
  return findings;
}

export const HYGIENE_RULES = RULES.map((rule) => rule.rule);
