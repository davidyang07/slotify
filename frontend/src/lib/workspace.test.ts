import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { makeManualSlot, type PlacementSignal } from "./recommendations";
import {
  parseBreakTime,
  reasonTags,
  renderBlockedReason,
  renderButtonLabel,
  shortSignalLabel,
} from "./workspace";

const supporting = (label: string): PlacementSignal => ({ kind: "supporting", label });
const caution = (label: string): PlacementSignal => ({ kind: "caution", label });
const heuristic = (signals: PlacementSignal[]) => ({
  source: "heuristic_offline_v1" as const,
  signals,
});
const texts = (tags: { text: string }[]) => tags.map((tag) => tag.text);

describe("reason tags", () => {
  test("the DESIGN.md example: pause 1.2s · mid-episode", () => {
    const tags = reasonTags(
      heuristic([
        supporting("Pause of 1.2s detected"),
        caution("No transcript context available here"),
        supporting("Mid-episode (37% through)"),
      ]),
    );
    assert.deepEqual(texts(tags), ["pause 1.2s", "mid-episode"]);
  });

  test("a positional caution is shown on the row, not only in its details", () => {
    const tags = reasonTags(
      heuristic([
        supporting("Pause of 2.9s detected"),
        caution("No transcript context available here"),
        caution("Very close to the end"),
      ]),
    );
    assert.deepEqual(texts(tags), ["pause 2.9s", "near the end"]);
    assert.equal(tags[1].kind, "caution");
  });

  test("the generic transcript caution only fills a row that would show one tag", () => {
    const tags = reasonTags(
      heuristic([
        supporting("Pause of 0.9s detected"),
        caution("No transcript context available here"),
      ]),
    );
    assert.deepEqual(texts(tags), ["pause 0.9s", "no transcript"]);
  });

  test("never more than three tags", () => {
    const tags = reasonTags(
      heuristic([
        supporting("Pause of 1.8s detected"),
        supporting("Falls on a sentence boundary"),
        supporting("Mid-episode (50% through)"),
        caution("Short pause may feel abrupt"),
      ]),
    );
    assert.deepEqual(texts(tags), ["pause 1.8s", "sentence end", "mid-episode"]);
  });

  test("nothing measured means no tags, not invented ones", () => {
    assert.deepEqual(reasonTags(heuristic([])), []);
  });

  test("a break the user added says so", () => {
    const manual = makeManualSlot(12, []);
    assert.deepEqual(reasonTags(manual), [{ text: "added by you", kind: "manual" }]);
  });

  test("an unknown signal keeps its own words", () => {
    assert.equal(shortSignalLabel("Music bed fades out (3s)"), "music bed fades out");
    assert.equal(shortSignalLabel("Beat-aligned low-energy valley"), "beat-aligned valley");
    assert.equal(
      shortSignalLabel("Low-energy valley (no beat grid detected)"),
      "low-energy valley",
    );
  });
});

describe("Add a break at mm:ss", () => {
  test("minutes and seconds become seconds", () => {
    assert.deepEqual(parseBreakTime("1", "05", 600), { seconds: 65 });
    assert.deepEqual(parseBreakTime("", "8", 22.9), { seconds: 8 });
    assert.deepEqual(parseBreakTime("2", "", null), { seconds: 120 });
  });

  test("an empty time is asked for, not read as 0:00", () => {
    assert.ok("error" in parseBreakTime("", "", 22));
  });

  test("seconds stop at 59 and must be whole numbers", () => {
    assert.ok("error" in parseBreakTime("0", "60", 600));
    assert.ok("error" in parseBreakTime("0", "1.5", 600));
    assert.ok("error" in parseBreakTime("-1", "0", 600));
  });

  test("a time past the end of the episode is refused with its length", () => {
    const result = parseBreakTime("0", "30", 22.9);
    assert.deepEqual(result, { error: "That is past the end of the episode (0:22)." });
  });

  test("with no known duration, any time is accepted", () => {
    assert.deepEqual(parseBreakTime("90", "0", null), { seconds: 5400 });
  });
});

describe("the Render button", () => {
  test("counts the selection", () => {
    assert.equal(renderButtonLabel(1, false), "Render 1 ad break");
    assert.equal(renderButtonLabel(2, false), "Render 2 ad breaks");
    assert.equal(renderButtonLabel(0, false), "Render");
    assert.equal(renderButtonLabel(3, true), "Rendering 3 ad breaks…");
  });

  const ready = {
    generationBlocked: null,
    selectedCount: 1,
    hasSponsorName: true,
    analyzing: false,
    previewing: false,
  };

  test("is on when everything it needs is there", () => {
    assert.equal(renderBlockedReason(ready), null);
  });

  test("a missing capability is the reason, whatever else is missing", () => {
    const reason = "Ad generation needs an ElevenLabs API key on the server.";
    assert.equal(
      renderBlockedReason({ ...ready, generationBlocked: reason, selectedCount: 0 }),
      reason,
    );
  });

  test("says what to do next", () => {
    assert.match(renderBlockedReason({ ...ready, selectedCount: 0 }) ?? "", /Select at least one/);
    assert.match(
      renderBlockedReason({ ...ready, selectedCount: 0, analyzing: true }) ?? "",
      /still being found/,
    );
    assert.match(renderBlockedReason({ ...ready, hasSponsorName: false }) ?? "", /sponsor name/);
    assert.match(renderBlockedReason({ ...ready, previewing: true }) ?? "", /preview/);
  });
});
