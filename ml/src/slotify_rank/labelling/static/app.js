/* Labelling client. No framework, no build step.
 *
 * The job is throughput: a person has a few thousand clips to judge, so the only
 * thing that matters is the time between hearing a clip and hearing the next
 * one. Three decisions follow from that.
 *
 * 1. **A rating is one keystroke.** `1`-`5` saves and advances. There is no
 *    confirm button and no "next" button, because either would double the
 *    keystrokes for the whole session.
 *
 * 2. **The next few items are already here.** `/api/batch` returns a small
 *    window of upcoming items and their audio is fetched into the browser cache
 *    while the current one plays, so advancing is a repaint rather than a round
 *    trip. Advancing therefore does not wait on the save: the POST is fired and
 *    the UI moves on, and a failure surfaces as a visible error with the item
 *    pushed back onto the front of the queue rather than lost.
 *
 * 3. **The server owns "where was I".** The annotator id is kept in
 *    localStorage so reopening the tab resumes the same session, but no
 *    judgement is ever buffered here. Everything else is recomputed from the
 *    database, so a crashed tab loses at most the item on screen.
 */

const $ = (id) => document.getElementById(id);

const BATCH_SIZE = 6;
/* Kept in the DOM rather than a variable so the browser actually holds the
 * bytes: an <audio> element with a src is a real cache entry. */
const prefetchNodes = new Map();

const state = {
  annotator: localStorage.getItem("slotify.annotator") || "",
  queue: [],
  current: null,
  shownAt: 0,
  loading: false,
  inflight: 0,
};

const setStatus = (message, isError = false) => {
  const node = $("status");
  node.textContent = message;
  node.classList.toggle("error", isError);
};

const request = async (url, options) => {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch (error) {
      /* response had no JSON body; the status line is the best we have */
    }
    throw new Error(detail);
  }
  return response.json();
};

const renderProgress = (progress) => {
  if (!progress) return;
  const target = progress.target || progress.total_candidates || 0;
  const done = progress.labelled || 0;
  const pct = target ? Math.min(100, Math.round((done / target) * 100)) : 0;
  $("progress-fill").style.width = `${pct}%`;
  $("counter").textContent = `${done} / ${target}`;
  const bits = [`${progress.remaining} to go`];
  if (progress.marked_unusable) bits.push(`${progress.marked_unusable} broken`);
  if (progress.skipped) bits.push(`${progress.skipped} skipped`);
  $("progress-text").textContent = `${pct}% · ${bits.join(" · ")}`;
};

const formatOffset = (ms) => {
  const seconds = Math.floor(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
};

const selectScore = (score) => {
  document.querySelectorAll("#scores button").forEach((button) => {
    button.classList.toggle("selected", Number(button.dataset.score) === score);
  });
};

/* Pull upcoming clips into the browser cache. Only the queue window is held, so
 * a long session never accumulates audio elements. */
const prefetch = () => {
  const wanted = new Set(state.queue.map((item) => item.clip_url));
  for (const [url, node] of prefetchNodes) {
    if (!wanted.has(url)) {
      node.removeAttribute("src");
      prefetchNodes.delete(url);
    }
  }
  for (const item of state.queue) {
    if (prefetchNodes.has(item.clip_url)) continue;
    const node = new Audio();
    node.preload = "auto";
    node.src = item.clip_url;
    prefetchNodes.set(item.clip_url, node);
  }
};

const renderCandidate = (item) => {
  state.current = item;
  state.shownAt = performance.now();
  selectScore(item.existing_quality_score ?? null);
  $("unusable").checked = Boolean(item.existing_is_unusable);
  $("notes").value = item.existing_notes || "";

  $("episode-title").textContent = item.episode_title;
  $("timestamp").textContent = item.timestamp_label;
  $("boundary-label").textContent = formatOffset(item.boundary_offset_ms);

  const audio = $("audio");
  audio.src = item.clip_url;
  audio.load();
  /* Autoplay is best-effort: a browser that refuses it leaves the clip cued and
   * Space or R plays it, which is why neither is the only way to hear a clip. */
  const attempt = audio.play();
  if (attempt && typeof attempt.catch === "function") attempt.catch(() => {});

  const hasTranscript = item.has_transcript;
  $("transcript-block").hidden = !hasTranscript;
  $("transcript-missing").hidden = hasTranscript;
  if (hasTranscript) {
    $("transcript-before").textContent = item.transcript_before || "—";
    $("transcript-after").textContent = item.transcript_after || "—";
  }
  $("empty").hidden = true;
  $("workspace").hidden = false;
};

const showEmpty = () => {
  state.current = null;
  $("workspace").hidden = true;
  $("empty").hidden = false;
};

const refill = async () => {
  if (state.loading) return;
  state.loading = true;
  try {
    const payload = await request(
      `/api/batch?annotator_id=${encodeURIComponent(state.annotator)}&count=${BATCH_SIZE}`,
    );
    renderProgress(payload.progress);
    /* The server excludes what is already judged or skipped, so a refill is
     * authoritative: anything held locally that it did not return is stale. */
    const seen = new Set();
    state.queue = payload.items.filter((item) => {
      if (seen.has(item.presentation_id)) return false;
      seen.add(item.presentation_id);
      return item.presentation_id !== state.current?.presentation_id;
    });
    prefetch();
  } finally {
    state.loading = false;
  }
};

const advance = async () => {
  if (state.queue.length === 0) await refill();
  const next = state.queue.shift();
  if (!next) {
    if (state.inflight > 0) {
      /* Saves still in flight; the queue may refill once they land. */
      setStatus("Saving…");
      return;
    }
    showEmpty();
    return;
  }
  renderCandidate(next);
  prefetch();
  if (state.queue.length <= 2) refill().catch(() => {});
};

const submit = async (score) => {
  const item = state.current;
  if (!item) return;
  const body = {
    presentation_id: item.presentation_id,
    annotator_id: state.annotator,
    quality_score: score,
    is_unusable: $("unusable").checked,
    notes: $("notes").value.trim() || null,
    elapsed_ms: Math.round(performance.now() - state.shownAt),
  };
  selectScore(score);
  setStatus("");
  await advance();

  state.inflight += 1;
  try {
    const payload = await request("/api/label", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderProgress(payload.progress);
  } catch (error) {
    /* A judgement that did not persist must not be silently dropped: put the
     * item back at the front and say so. */
    state.queue.unshift(item);
    setStatus(
      `Could not save that rating (${error.message}). It has been put back — re-rate it.`,
      true,
    );
  } finally {
    state.inflight -= 1;
  }
};

const skip = async () => {
  const item = state.current;
  if (!item) return;
  await advance();
  try {
    const payload = await request("/api/skip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        presentation_id: item.presentation_id,
        annotator_id: state.annotator,
        reason: $("notes").value.trim() || null,
      }),
    });
    renderProgress(payload.progress);
    setStatus("Skipped — it stays unjudged and comes back when you clear skips.");
  } catch (error) {
    setStatus(`Could not skip: ${error.message}`, true);
  }
};

const replayWindow = () => {
  const audio = $("audio");
  audio.currentTime = 0;
  audio.play();
};

const replayBoundary = () => {
  const audio = $("audio");
  const offsetSeconds = (state.current?.boundary_offset_ms || 0) / 1000;
  audio.currentTime = Math.max(0, offsetSeconds - 3);
  audio.play();
};

const start = async () => {
  const annotator = $("annotator").value.trim();
  if (!annotator) {
    setStatus("Enter an annotator id (a pseudonym is fine).", true);
    return;
  }
  state.annotator = annotator;
  localStorage.setItem("slotify.annotator", annotator);
  state.queue = [];
  setStatus("");
  try {
    await refill();
    await advance();
  } catch (error) {
    setStatus(error.message, true);
  }
};

document.querySelectorAll("#scores button").forEach((button) => {
  button.addEventListener("click", () => submit(Number(button.dataset.score)));
});

$("start").addEventListener("click", start);
$("replay-all").addEventListener("click", replayWindow);
$("replay-boundary").addEventListener("click", replayBoundary);
$("skip").addEventListener("click", skip);

$("notes").addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("notes").blur();
});

document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  const key = event.key.toLowerCase();
  if (["1", "2", "3", "4", "5"].includes(key)) {
    submit(Number(key));
  } else if (key === "r") {
    replayWindow();
  } else if (key === "b") {
    replayBoundary();
  } else if (key === "u") {
    $("unusable").checked = !$("unusable").checked;
  } else if (key === "s") {
    skip();
  } else if (key === "n") {
    event.preventDefault();
    $("notes").focus();
  } else if (key === " ") {
    event.preventDefault();
    const audio = $("audio");
    audio.paused ? audio.play() : audio.pause();
  }
});

if (state.annotator) {
  $("annotator").value = state.annotator;
  start();
}
