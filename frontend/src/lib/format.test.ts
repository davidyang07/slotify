import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  UNKNOWN_DURATION,
  describeFile,
  downloadName,
  fileFormatLabel,
  formatBytes,
  formatTimestamp,
} from "./format";

describe("formatTimestamp", () => {
  test("m:ss with zero-padded seconds", () => {
    assert.equal(formatTimestamp(0), "0:00");
    assert.equal(formatTimestamp(8.471), "0:08");
    assert.equal(formatTimestamp(22.9), "0:22");
    assert.equal(formatTimestamp(65), "1:05");
  });

  test("minutes are not wrapped into hours, matching the mm:ss inputs", () => {
    assert.equal(formatTimestamp(75 * 60 + 3), "75:03");
  });

  test("garbage reads as 0:00, never NaN", () => {
    assert.equal(formatTimestamp(Number.NaN), "0:00");
    assert.equal(formatTimestamp(null), "0:00");
    assert.equal(formatTimestamp(-4), "0:00");
    assert.equal(formatTimestamp(Number.POSITIVE_INFINITY), "0:00");
  });
});

describe("formatBytes", () => {
  test("the DESIGN.md example: ryan-test1.mp3 is 358 KB", () => {
    assert.equal(formatBytes(367101), "358 KB");
  });

  test("bytes, kilobytes, megabytes, gigabytes", () => {
    assert.equal(formatBytes(512), "512 B");
    assert.equal(formatBytes(1024), "1 KB");
    assert.equal(formatBytes(5.5 * 1024 * 1024), "5.5 MB");
    assert.equal(formatBytes(2 * 1024 * 1024 * 1024), "2.0 GB");
  });

  test("an impossible size is not printed as NaN", () => {
    assert.equal(formatBytes(Number.NaN), "0 B");
  });
});

describe("fileFormatLabel", () => {
  test("the extension, upper-cased", () => {
    assert.equal(fileFormatLabel("ryan-test1.mp3", "audio/mpeg"), "MP3");
    assert.equal(fileFormatLabel("episode.final.wav"), "WAV");
  });

  test("falls back to the MIME subtype without an extension", () => {
    assert.equal(fileFormatLabel("recording", "audio/mpeg"), "MP3");
    assert.equal(fileFormatLabel("recording", "audio/x-wav"), "WAV");
    assert.equal(fileFormatLabel("recording", "audio/ogg; codecs=opus"), "OGG");
  });

  test("says AUDIO when nothing identifies the format", () => {
    assert.equal(fileFormatLabel("recording", ""), "AUDIO");
    assert.equal(fileFormatLabel(".hidden", ""), "AUDIO");
  });
});

describe("describeFile", () => {
  test("name · format · size · duration, as the header shows them", () => {
    assert.deepEqual(
      describeFile({ name: "ryan-test1.mp3", size: 367101, type: "audio/mpeg" }, 22.909),
      { name: "ryan-test1.mp3", format: "MP3", size: "358 KB", duration: "0:22" },
    );
  });

  test("an unknown duration is shown as unknown, not as 0:00", () => {
    const parts = describeFile({ name: "a.mp3", size: 10 }, null);
    assert.equal(parts.duration, UNKNOWN_DURATION);
    assert.equal(describeFile({ name: "a.mp3", size: 10 }, 0).duration, UNKNOWN_DURATION);
  });
});

describe("downloadName", () => {
  test("keeps the episode's name and says what changed", () => {
    assert.equal(downloadName("ryan-test1.mp3"), "ryan-test1-with-ads.mp3");
    assert.equal(downloadName("show.final.wav"), "show.final-with-ads.mp3");
  });

  test("never produces a nameless file", () => {
    assert.equal(downloadName(""), "slotify-with-ads.mp3");
    assert.equal(downloadName(".mp3"), ".mp3-with-ads.mp3");
  });
});
