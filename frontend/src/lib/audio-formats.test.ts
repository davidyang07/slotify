import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  AUDIO_ACCEPT,
  SUPPORTED_AUDIO_EXTENSIONS,
  SUPPORTED_AUDIO_LABEL,
  isSupportedAudioFile,
  unsupportedAudioMessage,
} from "./audio-formats";

const BACKEND_UPLOAD = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../backend/src/middleware/upload.ts",
);

describe("the file picker offers what the API accepts", () => {
  test("never audio/*, and none of the formats the API refuses", () => {
    assert.ok(!AUDIO_ACCEPT.includes("audio/*"));
    for (const refused of [".aif", ".aiff", ".caf", ".wma", ".m3u8", ".m3u", ".pls"]) {
      assert.ok(!AUDIO_ACCEPT.split(",").includes(refused), refused);
      assert.equal(isSupportedAudioFile({ name: `episode${refused}` }), false, refused);
    }
  });

  test(
    "matches the backend's extension allow-list exactly",
    { skip: !fs.existsSync(BACKEND_UPLOAD) && "backend sources not present" },
    () => {
      const source = fs.readFileSync(BACKEND_UPLOAD, "utf8");
      const block = /const AUDIO_EXTENSIONS = new Set\(\[([\s\S]*?)\]\)/.exec(source)?.[1] ?? "";
      const backend = [...block.matchAll(/"(\.[a-z0-9]+)"/g)].map((match) => match[1]);
      assert.ok(backend.length > 5, "could not read the backend list");
      assert.deepEqual([...SUPPORTED_AUDIO_EXTENSIONS].sort(), [...backend].sort());
    },
  );

  test("supported names pass, whatever their case; a bare name is left to the server", () => {
    assert.equal(isSupportedAudioFile({ name: "Episode 12.MP3" }), true);
    assert.equal(isSupportedAudioFile({ name: "voice memo.m4a" }), true);
    assert.equal(isSupportedAudioFile({ name: "recording" }), true);
  });

  test("the caption and the refusal name the formats", () => {
    assert.equal(SUPPORTED_AUDIO_LABEL, "MP3, WAV, FLAC, OGG/Opus, M4A/MP4, AAC or WebM");
    assert.match(unsupportedAudioMessage("song.aiff"), /"song.aiff" is not a supported audio format\. Use MP3/);
  });
});
