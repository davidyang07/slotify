/**
 * Repository hygiene, as a function over a list of tracked paths.
 *
 * Pure so it can be tested without a checkout. `npm run verify:ml` feeds it
 * `git ls-files`, so it judges what is COMMITTED -- not what happens to be
 * sitting in a working tree. An editor's scratch file nobody committed is not a
 * repository problem; a committed one is, and stays one forever.
 *
 * Each rule names a class of file that should never be in the history:
 *
 * - OS and editor metadata, which is noise in every clone but the one that made it;
 * - caches and compiled bytecode, which are regenerated and go stale;
 * - credentials, which cannot be un-published once pushed;
 * - audio and model binaries, because this corpus is reconstructed from a
 *   committed source registry rather than stored, and a checkout that carries
 *   gigabytes of it stops being clonable.
 *
 * The media rules have two deliberate exemptions, both product code rather than
 * data: `backend/audio_tests/` holds the small sample clips the product's own
 * manual and CLI testing needs, and `frontend/src/assets/` holds the UI's own
 * media. Neither is corpus, and neither is regenerable from a source registry.
 */

/** @typedef {{path: string, rule: string, reason: string}} Finding */

const OS_METADATA = /(^|\/)(\.DS_Store|Thumbs\.db|desktop\.ini|\._[^/]+)$/i;
const EDITOR_SWAP = /(^|\/)([^/]+\.(swp|swo|orig|rej|bak)|.*~)$/i;
const CACHE = /(^|\/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|node_modules|\.venv|htmlcov)(\/|$)/;
const COMPILED = /\.(pyc|pyo)$/i;
const COVERAGE = /(^|\/)\.coverage(\.|$)/;
const CREDENTIAL = /(^|\/)(\.env(\..+)?|.*\.pem|.*\.p12|id_rsa|.*\.keystore)$/i;
const AUDIO = /\.(mp3|wav|flac|m4a|ogg|opus|aac|mp4|mkv)$/i;
const MEDIA_EXEMPT = /^(backend\/audio_tests|frontend\/src\/assets)\//;
const HEAVY_BINARY = /\.(pt|onnx|bin|ckpt|safetensors|npy|npz|parquet|zip|tar|gz)$/i;

/**
 * Checkpoints are exempt only where a README explains why they are committed.
 * `artifacts/training/README.md` documents the single bootstrap checkpoint; any
 * other committed weight file is a finding.
 */
const CHECKPOINT_EXEMPT = /^artifacts\/training\/[^/]+\/best_checkpoint\.pt$/;

const RULES = [
  { rule: "os-metadata", test: (p) => OS_METADATA.test(p), reason: "operating-system metadata" },
  { rule: "editor-scratch", test: (p) => EDITOR_SWAP.test(p), reason: "editor swap or backup file" },
  { rule: "cache", test: (p) => CACHE.test(p), reason: "a regenerated cache directory" },
  { rule: "compiled", test: (p) => COMPILED.test(p), reason: "compiled Python bytecode" },
  { rule: "coverage", test: (p) => COVERAGE.test(p), reason: "a coverage database" },
  { rule: "credential", test: (p) => CREDENTIAL.test(p), reason: "a credential or private key" },
  {
    rule: "audio-corpus",
    test: (p) => AUDIO.test(p) && !MEDIA_EXEMPT.test(p),
    reason:
      "audio or video outside backend/audio_tests/ and frontend/src/assets/; " +
      "the corpus is reconstructed from its source registry, never stored",
  },
  {
    rule: "heavy-binary",
    test: (p) =>
      HEAVY_BINARY.test(p) && !CHECKPOINT_EXEMPT.test(p) && !MEDIA_EXEMPT.test(p),
    reason: "a large binary with no documented reason to be committed",
  },
];

/**
 * @param {string[]} trackedPaths repository-relative, forward slashes.
 * @returns {Finding[]}
 */
export function hygieneFindings(trackedPaths) {
  /** @type {Finding[]} */
  const findings = [];
  for (const raw of trackedPaths) {
    const filePath = String(raw).trim().replaceAll("\\", "/");
    if (!filePath) continue;
    for (const rule of RULES) {
      if (rule.test(filePath)) {
        findings.push({ path: filePath, rule: rule.rule, reason: rule.reason });
        break;
      }
    }
  }
  return findings;
}

export const HYGIENE_RULES = RULES.map((rule) => rule.rule);
