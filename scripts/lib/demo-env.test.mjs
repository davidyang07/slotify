import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, test } from "node:test";

import { apiPort, backendEnv, DEFAULT_API_PORT, parseDotenv } from "./demo-env.mjs";

const roots = [];
after(() => {
  for (const root of roots) fs.rmSync(root, { recursive: true, force: true });
});

const repoWithEnv = (text) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "slotify-env-"));
  roots.push(root);
  fs.mkdirSync(path.join(root, "backend"));
  if (text !== null) fs.writeFileSync(path.join(root, "backend", ".env"), text);
  return root;
};

test("parses the dotenv syntax a hand-written backend/.env uses", () => {
  assert.deepEqual(
    parseDotenv(
      [
        "# a comment",
        "",
        "PORT=3002",
        "export PYTHON_BIN=/opt/venv/bin/python",
        'OPENAI_API_KEY="sk-quoted # not a comment"',
        "ELEVENLABS_API_KEY='single' # comment after quotes",
        "RANKER_MODE=heuristic   # trailing comment",
        "EMPTY=",
        "not a line",
      ].join("\r\n"),
    ),
    {
      PORT: "3002",
      PYTHON_BIN: "/opt/venv/bin/python",
      OPENAI_API_KEY: "sk-quoted # not a comment",
      ELEVENLABS_API_KEY: "single",
      RANKER_MODE: "heuristic",
      EMPTY: "",
    },
  );
});

test("the real environment wins over backend/.env, as dotenv does", () => {
  const root = repoWithEnv("PORT=4000\nPYTHON_BIN=/from/file/python\n");
  const env = backendEnv(root, { PORT: "5000" });
  assert.equal(env.PORT, "5000");
  assert.equal(env.PYTHON_BIN, "/from/file/python");
  assert.equal(apiPort(env), 5000);
});

test("no backend/.env means the environment alone, and the default port", () => {
  const env = backendEnv(repoWithEnv(null), {});
  assert.deepEqual(env, {});
  assert.equal(apiPort(env), DEFAULT_API_PORT);
});
