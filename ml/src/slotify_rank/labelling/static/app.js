/* Labelling client. No framework, no build step.
 *
 * The annotator id is kept in localStorage so closing the tab and reopening it
 * resumes the same session; the server decides what "next" means, so resumption
 * is really the server's stable per-annotator ordering plus the set of labels
 * already stored.
 */

const $ = (id) => document.getElementById(id);

const state = {
  annotator: localStorage.getItem("slotify.annotator") || "",
  candidate: null,
  score: null,
  busy: false,
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
  const total = progress.total_candidates || 0;
  const done = progress.labelled || 0;
  const pct = total ? Math.round((done / total) * 100) : 0;
  $("progress-fill").style.width = `${pct}%`;
  $("progress-text").textContent =
    `${done} of ${total} labelled (${pct}%) · ${progress.remaining} remaining` +
    (progress.marked_unusable ? ` · ${progress.marked_unusable} unusable` : "");
};

const formatOffset = (ms) => {
  const seconds = Math.floor(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
};

const selectScore = (score) => {
  state.score = score;
  document.querySelectorAll("#scores button").forEach((button) => {
    button.classList.toggle("selected", Number(button.dataset.score) === score);
  });
};

const renderCandidate = (candidate) => {
  state.candidate = candidate;
  selectScore(candidate.existing_quality_score ?? null);
  $("unusable").checked = Boolean(candidate.existing_is_unusable);
  $("notes").value = candidate.existing_notes || "";

  $("episode-title").textContent = candidate.episode_title;
  $("timestamp").textContent = candidate.timestamp_label;
  $("boundary-label").textContent = formatOffset(candidate.boundary_offset_ms);

  const audio = $("audio");
  audio.src = `/api/clip/${encodeURIComponent(candidate.candidate_id)}`;
  audio.load();

  const hasTranscript = candidate.has_transcript;
  $("transcript-block").hidden = !hasTranscript;
  $("transcript-missing").hidden = hasTranscript;
  if (hasTranscript) {
    $("transcript-before").textContent = candidate.transcript_before || "—";
    $("transcript-after").textContent = candidate.transcript_after || "—";
  }
};

const loadNext = async () => {
  const payload = await request(
    `/api/next?annotator_id=${encodeURIComponent(state.annotator)}`,
  );
  renderProgress(payload.progress);
  if (!payload.candidate) {
    $("workspace").hidden = true;
    $("empty").hidden = false;
    return;
  }
  $("empty").hidden = true;
  $("workspace").hidden = false;
  renderCandidate(payload.candidate);
};

const submit = async () => {
  if (state.busy) return;
  if (state.score === null) {
    setStatus("Pick a score from 1 to 5 first.", true);
    return;
  }
  state.busy = true;
  try {
    // Saved immediately on submit: nothing is buffered client-side, so a closed
    // tab never loses a judgement.
    const payload = await request("/api/label", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        candidate_id: state.candidate.candidate_id,
        annotator_id: state.annotator,
        quality_score: state.score,
        is_unusable: $("unusable").checked,
        notes: $("notes").value.trim() || null,
      }),
    });
    renderProgress(payload.progress);
    setStatus("Saved.");
    await loadNext();
  } catch (error) {
    setStatus(`Could not save: ${error.message}`, true);
  } finally {
    state.busy = false;
  }
};

const replayWindow = () => {
  const audio = $("audio");
  audio.currentTime = 0;
  audio.play();
};

const replayBoundary = () => {
  const audio = $("audio");
  const offsetSeconds = (state.candidate?.boundary_offset_ms || 0) / 1000;
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
  setStatus("");
  try {
    await loadNext();
  } catch (error) {
    setStatus(error.message, true);
  }
};

document.querySelectorAll("#scores button").forEach((button) => {
  button.addEventListener("click", () => {
    selectScore(Number(button.dataset.score));
    submit();
  });
});

$("start").addEventListener("click", start);
$("replay-all").addEventListener("click", replayWindow);
$("replay-boundary").addEventListener("click", replayBoundary);

document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  const key = event.key.toLowerCase();
  if (["1", "2", "3", "4", "5"].includes(key)) {
    selectScore(Number(key));
    submit();
  } else if (key === "r") {
    replayWindow();
  } else if (key === "b") {
    replayBoundary();
  } else if (key === "u") {
    $("unusable").checked = !$("unusable").checked;
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
