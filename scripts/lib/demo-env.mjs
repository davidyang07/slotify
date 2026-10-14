/**
 * The configuration preflight and the demo must agree on with the backend.
 *
 * The API reads its settings from `backend/.env` as well as from the
 * environment (`import "dotenv/config"`, which never overrides a variable that
 * is already set). A script that read only `process.env` would check `python`
 * while the API runs the PYTHON_BIN in backend/.env, or probe port 3001 while
 * the API listens on the PORT written there -- so both scripts resolve those
 * settings here, the same way the API does.
 */

import fs from "node:fs";
import path from "node:path";

/** Vite's port. The demo pins it with --strictPort, so it is fixed, not a default. */
export const UI_PORT = 5173;

/** The API's port when PORT is unset; backend/src/config.ts has the same default. */
export const DEFAULT_API_PORT = 3001;

/**
 * The subset of dotenv syntax a hand-written backend/.env uses: `KEY=value`,
 * an optional `export `, `#` comments, and single, double or backtick quotes.
 * As in dotenv, an unquoted value ends at the first `#`.
 * @param {string} text
 * @returns {Record<string, string>}
 */
export const parseDotenv = (text) => {
  /** @type {Record<string, string>} */
  const values = {};
  for (const line of String(text).split(/\r?\n/)) {
    const match = /^\s*(?:export\s+)?([\w.-]+)\s*=\s*(.*)$/.exec(line);
    if (!match) continue;
    const value = match[2].trim();
    const quoted = /^(['"`])(.*?)\1\s*(#.*)?$/.exec(value);
    values[match[1]] = quoted ? quoted[2] : value.replace(/#.*$/, "").trim();
  }
  return values;
};

/**
 * The environment the API will see: backend/.env underneath, the real
 * environment on top.
 * @param {string} repoRoot
 * @param {NodeJS.ProcessEnv} [env]
 */
export const backendEnv = (repoRoot, env = process.env) => {
  let fromFile = {};
  try {
    fromFile = parseDotenv(fs.readFileSync(path.join(repoRoot, "backend", ".env"), "utf8"));
  } catch {
    fromFile = {};
  }
  return { ...fromFile, ...env };
};

/**
 * The port the API will listen on, parsed exactly as backend/src/config.ts does.
 * @param {Record<string, string | undefined>} env
 */
export const apiPort = (env) => Number.parseInt(env.PORT ?? String(DEFAULT_API_PORT), 10);
