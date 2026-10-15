import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import AdPanel from "./components/AdPanel";
import AppHeader from "./components/AppHeader";
import BreakList, { type AnalysisNotice } from "./components/BreakList";
import EmptyState from "./components/EmptyState";
import ProcessingOverlay from "./components/ProcessingOverlay";
import RightsModal from "./components/RightsModal";
import WaveformPanel from "./components/WaveformPanel";
import {
  apiErrorMessage,
  createApiFetch,
  readApiError,
  resolveApiConfig,
} from "./lib/api";
import {
  BACKEND_UNREACHABLE_MESSAGE,
  UNKNOWN_CAPABILITIES,
  fetchCapabilities,
  generationBlockedReason,
  unreadableCapabilities,
  type Capabilities,
} from "./lib/capabilities";
import { downloadName } from "./lib/format";
import {
  combineSlots,
  emptyStateMessage,
  makeManualSlot,
  parsePlacementResponse,
  selectionAfterPlacement,
  slotLabel,
  type PlacementResult,
  type Slot,
} from "./lib/recommendations";
import { TaskScope, type ScopedTask } from "./lib/tasks";
import { releaseVoice } from "./lib/voice";
import {
  MAX_SPONSORS,
  renderBlockedReason,
  renderButtonLabel,
  type GenerationMode,
  type GenerationNote,
  type Sponsor,
} from "./lib/workspace";

// Same-origin `/api/...` by default: `vite` and `vite preview` proxy /api to
// the backend (vite.config.ts), and a deployment serves both from one origin.
// VITE_API_BASE_URL overrides the base; VITE_API_TOKEN, when set, is sent as a
// bearer token on every request.
const API_CONFIG = resolveApiConfig({
  VITE_API_BASE_URL: import.meta.env.VITE_API_BASE_URL,
  VITE_API_TOKEN: import.meta.env.VITE_API_TOKEN,
});
const apiFetch = createApiFetch(API_CONFIG);

/** Generous, but a hung analyser must not hold the UI until a reload. */
const ANALYSIS_TIMEOUT_MS = 10 * 60 * 1000;

/** What a placement result replaces: the state it was asked for from. */
type AnalysisBase = { manualSlots: Slot[]; selectedSlotIds: string[] };

/**
 * Slotify is one screen. With no file it is the empty state; with one, the
 * header's file bar, the waveform and break list, and the ad panel. Dropping
 * or choosing a file starts analysis at once.
 */
function App() {
  /** Per-file async work. Reset whenever the uploaded file changes. */
  const fileTasksRef = useRef<TaskScope | null>(null);
  const analysisTaskRef = useRef<ScopedTask | null>(null);
  const generationTaskRef = useRef<ScopedTask | null>(null);
  /** Monotonic, so a removed sponsor's id is never handed out again. */
  const sponsorCounterRef = useRef(1);

  const [baseAudio, setBaseAudio] = useState<File | null>(null);
  const [voiceId, setVoiceId] = useState("");
  /** The voice to release on `pagehide`, which must not read stale state. */
  const voiceIdRef = useRef("");
  const [isCloningVoice, setIsCloningVoice] = useState(false);
  const [voiceCloneError, setVoiceCloneError] = useState("");
  const [analysisNotice, setAnalysisNotice] = useState<AnalysisNotice | null>(null);
  const [placement, setPlacement] = useState<PlacementResult | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities>(
    UNKNOWN_CAPABILITIES,
  );
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  /** Decoded once per file; drawing reads these and never re-decodes. */
  const [waveformPeaks, setWaveformPeaks] = useState<Float32Array | null>(null);
  const [waveformFailed, setWaveformFailed] = useState(false);
  const [manualSlots, setManualSlots] = useState<Slot[]>([]);
  const [selectedSlotIds, setSelectedSlotIds] = useState<string[]>([]);
  /** The break the waveform's insertion cursor sits on. */
  const [focusedSlotId, setFocusedSlotId] = useState<string | null>(null);
  const [sponsors, setSponsors] = useState<Sponsor[]>([
    { id: "sponsor-1", name: "", script: "" },
  ]);
  const [slotAssignments, setSlotAssignments] = useState<Record<string, string>>(
    {},
  );
  /** The last preview's or render's message, shown where it happened. */
  const [note, setNote] = useState<GenerationNote | null>(null);
  /** The last preview, played inline in the row that asked for it. */
  const [preview, setPreview] = useState<{ slotId: string; url: string } | null>(null);
  const [renderUrl, setRenderUrl] = useState("");
  const [baseAudioUrl, setBaseAudioUrl] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [previewingSlotId, setPreviewingSlotId] = useState<string | null>(null);
  const [isRendering, setIsRendering] = useState(false);
  const [showRightsModal, setShowRightsModal] = useState(false);
  const [rightsAccepted, setRightsAccepted] = useState(false);
  const [rightsCertified, setRightsCertified] = useState(false);
  /** Set while the rights modal is open, so confirming resumes the right action. */
  const [pendingGeneration, setPendingGeneration] = useState<{
    mode: GenerationMode;
    slotIds: string[];
  } | null>(null);
  const [selectedTone, setSelectedTone] = useState("professional");
  const [selectedLanguage, setSelectedLanguage] = useState("en");

  // Slots are exactly what the server returned, plus the ones the user added.
  // If it returned none, none are shown -- the UI used to top the list up to
  // three with slots at 22/48/72 % of the duration and hard-coded scores of
  // 92/85/78, which made an analysis that found nothing indistinguishable from
  // one that found three good breaks. Derived rather than copied into state by
  // an effect, which used to wipe manual slots whenever the selection changed.
  const slots = useMemo(
    () => combineSlots(placement?.slots ?? [], manualSlots, audioDuration),
    [placement, manualSlots, audioDuration],
  );
  const selectedCount = slots.filter((slot) => selectedSlotIds.includes(slot.id)).length;
  const hasSponsorName = sponsors.some((entry) => entry.name.trim());
  /** Generation needs a sponsor to read out; analysis does not. */
  const isSponsorReady = Boolean(baseAudio && hasSponsorName);

  useEffect(() => {
    voiceIdRef.current = voiceId;
  }, [voiceId]);

  // Closing or leaving the page releases its cloned voice, so it does not sit
  // in the ElevenLabs account until the server's idle timeout. A page restored
  // from the back/forward cache then clones again when it next needs to.
  useEffect(() => {
    const onPageHide = () => {
      const current = voiceIdRef.current;
      if (!current) return;
      voiceIdRef.current = "";
      setVoiceId("");
      void releaseVoice(apiFetch, current);
    };
    window.addEventListener("pagehide", onPageHide);
    return () => window.removeEventListener("pagehide", onPageHide);
  }, []);

  // Each generated file is released when it is replaced or the file changes.
  useEffect(() => {
    if (!renderUrl) return;
    return () => URL.revokeObjectURL(renderUrl);
  }, [renderUrl]);

  useEffect(() => {
    const url = preview?.url;
    if (!url) return;
    return () => URL.revokeObjectURL(url);
  }, [preview]);

  useEffect(() => {
    if (!baseAudio) {
      setBaseAudioUrl("");
      return;
    }
    const nextUrl = URL.createObjectURL(baseAudio);
    setBaseAudioUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return nextUrl;
    });
    return () => {
      URL.revokeObjectURL(nextUrl);
    };
  }, [baseAudio]);

  // One player at a time: the original, a row's preview and the render
  // otherwise talk over each other.
  useEffect(() => {
    const onPlay = (event: Event) => {
      document.querySelectorAll("audio").forEach((element) => {
        if (element !== event.target && !element.paused) element.pause();
      });
    };
    document.addEventListener("play", onPlay, true);
    return () => document.removeEventListener("play", onPlay, true);
  }, []);

  // Decode once per file. This effect used to depend on the drawing code, so
  // every slot click re-read the file and re-decoded all of it (about 1.3 GB of
  // Float32 for an hour of audio), and overlapping decodes could crash the tab.
  // The per-file reset of peaks and duration happens in startFile.
  useEffect(() => {
    if (!baseAudio) return;

    let cancelled = false;
    const buildWaveform = async () => {
      let context: AudioContext | null = null;
      try {
        const arrayBuffer = await baseAudio.arrayBuffer();
        if (cancelled) return;
        const AudioContextCtor =
          window.AudioContext ||
          (window as typeof window & {
            webkitAudioContext?: typeof AudioContext;
          }).webkitAudioContext;
        if (!AudioContextCtor) throw new Error("Web Audio is unavailable.");
        context = new AudioContextCtor();
        const audioBuffer = await context.decodeAudioData(arrayBuffer);
        // A newer file may have been selected while this one decoded.
        if (cancelled) return;
        const channel = audioBuffer.getChannelData(0);
        const samples = 900;
        const blockSize = Math.max(1, Math.floor(channel.length / samples));
        const peaks = new Float32Array(samples);
        for (let i = 0; i < samples; i += 1) {
          const start = i * blockSize;
          const end = Math.min(start + blockSize, channel.length);
          let max = 0;
          for (let j = start; j < end; j += 1) {
            const value = Math.abs(channel[j]);
            if (value > max) max = value;
          }
          peaks[i] = max;
        }
        setAudioDuration(audioBuffer.duration);
        setWaveformPeaks(peaks);
      } catch {
        if (cancelled) return;
        setAudioDuration(null);
        setWaveformPeaks(null);
        setWaveformFailed(true);
      } finally {
        // Also on a failed decode: an unclosed context holds audio resources.
        context?.close().catch(() => undefined);
      }
    };

    void buildWaveform();
    return () => {
      cancelled = true;
    };
  }, [baseAudio]);

  // What this deployment can do. Placement never depends on it; only ad
  // generation is gated, and it says why it is gated.
  useEffect(() => {
    let cancelled = false;
    fetchCapabilities(apiFetch)
      .then((next) => {
        if (!cancelled) setCapabilities(next);
      })
      .catch(() => {
        if (!cancelled) {
          setCapabilities(unreadableCapabilities(BACKEND_UNREACHABLE_MESSAGE));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setRightsCertified(false);
  }, [baseAudio, sponsors]);

  useEffect(() => {
    setSlotAssignments((prev) => {
      const sponsorIds = sponsors.map((entry) => entry.id);
      if (sponsorIds.length === 0) return {};
      const next = { ...prev };
      slots.forEach((slot, index) => {
        const current = next[slot.id];
        if (!current || !sponsorIds.includes(current)) {
          next[slot.id] =
            sponsorIds[Math.min(index, sponsorIds.length - 1)] ?? sponsorIds[0];
        }
      });
      Object.keys(next).forEach((key) => {
        if (!slots.some((slot) => slot.id === key)) {
          delete next[key];
        }
      });
      return next;
    });
  }, [slots, sponsors]);

  const fileTasks = () => (fileTasksRef.current ??= new TaskScope());

  const generationBlocked = generationBlockedReason(capabilities);
  const authHint =
    capabilities.authRequired && !API_CONFIG.token
      ? "This server requires an API token. Set VITE_API_TOKEN for the frontend and restart it."
      : null;
  const capabilitiesKnown =
    capabilities !== UNKNOWN_CAPABILITIES && !capabilities.loadError;
  const scriptWriterNote =
    capabilitiesKnown && !capabilities.openaiEnhancement
      ? "With no OpenAI key on the server, it writes from a fixed template and tone is not applied."
      : null;
  const durationForUi = audioDuration ?? placement?.duration ?? null;

  /**
   * Show a new placement. Manual slots survive it; selections of slots it no
   * longer has are dropped rather than left to be silently skipped at render
   * time, and with nothing left selected the top-ranked slot is selected.
   * `base` is the state the analysis was started from: for a new file that is
   * nothing, whatever the previous file had selected.
   */
  const applyPlacement = (result: PlacementResult, base: AnalysisBase) => {
    const nextSlots = combineSlots(result.slots, base.manualSlots, null);
    const nextSelection = selectionAfterPlacement(base.selectedSlotIds, nextSlots);
    setPlacement(result);
    setSelectedSlotIds(nextSelection);
    setFocusedSlotId((prev) =>
      prev && nextSlots.some((slot) => slot.id === prev)
        ? prev
        : (nextSelection[0] ?? null),
    );
  };

  /**
   * Analysis is the first thing that happens after upload and needs no
   * credentials. Voice cloning used to gate it -- the rights modal ran, the
   * ElevenLabs clone ran, and placement only happened if that succeeded, so the
   * whole ranking demo depended on a paid API. Cloning now happens lazily, at
   * the point where audio is actually generated. Progress is shown inline in
   * the break list, so the ad panel stays usable meanwhile.
   */
  const runPlacementAnalysis = async (file: File, base: AnalysisBase) => {
    setAnalysisNotice(null);
    const task = fileTasks().begin(ANALYSIS_TIMEOUT_MS);
    analysisTaskRef.current = task;
    setIsAnalyzing(true);
    try {
      const form = new FormData();
      form.append("audio", file);
      form.append("count", "3");
      form.append("tone", selectedTone);
      form.append("language", selectedLanguage);

      const response = await apiFetch("/api/insert-sections", {
        method: "POST",
        body: form,
        signal: task.signal,
      });

      // A failed analysis still answers in the placement schema, so the honest
      // degraded state is rendered from the body rather than from a bare status.
      // Anything else (a 401 or 429 `{error}`, a proxy error) is a request that
      // never reached the analyser, and its message is shown as is.
      const body = await response.text();
      let data: unknown = null;
      try {
        data = JSON.parse(body);
      } catch {
        data = null;
      }
      const isPlacementBody =
        typeof data === "object" && data !== null && "placementStatus" in data;
      if (!response.ok && !isPlacementBody) {
        throw new Error(
          apiErrorMessage(
            response.status,
            body,
            `Insert analysis failed (HTTP ${response.status}).`,
          ),
        );
      }
      const result = parsePlacementResponse(data);
      if (!task.isCurrent()) return;
      applyPlacement(result, base);
      if (result.status === "unavailable") {
        setAnalysisNotice({ kind: "error", text: emptyStateMessage(result) });
      } else if (result.status !== "ok" || result.slots.length === 0) {
        setAnalysisNotice({ kind: "empty", text: emptyStateMessage(result) });
      }
    } catch (analysisErr) {
      if (!task.isCurrent()) return;
      const aborted = task.abortReason();
      if (aborted === "cancelled") {
        setAnalysisNotice({ kind: "cancelled", text: "Analysis cancelled." });
        return;
      }
      if (aborted === "timeout") {
        setAnalysisNotice({
          kind: "error",
          text: `Analysis took longer than ${ANALYSIS_TIMEOUT_MS / 60000} minutes and was stopped. Try again, or try a shorter clip.`,
        });
        return;
      }
      // fetch() rejects with a TypeError when the backend cannot be reached.
      const message =
        analysisErr instanceof TypeError
          ? BACKEND_UNREACHABLE_MESSAGE
          : analysisErr instanceof Error
            ? analysisErr.message
            : "Insert analysis failed.";
      applyPlacement(
        {
          status: "unavailable",
          slots: [],
          duration: null,
          candidateCount: null,
          provenance: null,
          message,
        },
        base,
      );
      setAnalysisNotice({ kind: "error", text: message });
    } finally {
      task.finish();
      if (analysisTaskRef.current === task) analysisTaskRef.current = null;
      if (task.isCurrent()) setIsAnalyzing(false);
    }
  };

  /**
   * A new file starts from nothing, and its analysis starts at once. The
   * previous file's in-flight requests are abandoned first, so a render, clone
   * or analysis of the old file cannot land in the new file's state, and then
   * everything derived from it -- placement, slots, voice, rendered output,
   * messages -- is cleared.
   */
  const startFile = (file: File) => {
    if (file === baseAudio) return;
    fileTasks().reset();
    // The previous file's clone is of no use to this one.
    if (voiceIdRef.current) void releaseVoice(apiFetch, voiceIdRef.current);
    voiceIdRef.current = "";
    setBaseAudio(file);
    setAudioDuration(null);
    setWaveformPeaks(null);
    setWaveformFailed(false);
    setVoiceId("");
    setVoiceCloneError("");
    setIsCloningVoice(false);
    setPlacement(null);
    setAnalysisNotice(null);
    setIsAnalyzing(false);
    setManualSlots([]);
    setSelectedSlotIds([]);
    setFocusedSlotId(null);
    setSlotAssignments({});
    setPreview(null);
    setRenderUrl("");
    setNote(null);
    setPreviewingSlotId(null);
    setIsRendering(false);
    setPendingGeneration(null);
    setShowRightsModal(false);
    void runPlacementAnalysis(file, { manualSlots: [], selectedSlotIds: [] });
  };

  const analyzeAgain = () => {
    if (!baseAudio || isAnalyzing) return;
    void runPlacementAnalysis(baseAudio, { manualSlots, selectedSlotIds });
  };

  const updateSponsor = (id: string, patch: Partial<Sponsor>) => {
    setSponsors((prev) =>
      prev.map((entry) =>
        entry.id === id ? { ...entry, ...patch } : entry,
      ),
    );
  };

  const addSponsor = () => {
    if (sponsors.length >= MAX_SPONSORS) return;
    // Built from a counter, not `prev.length + 1`: add, remove Sponsor 1, add
    // used to produce two `sponsor-2` entries sharing a key and input ids.
    sponsorCounterRef.current += 1;
    const id = `sponsor-${sponsorCounterRef.current}`;
    setSponsors((prev) =>
      prev.length >= MAX_SPONSORS ? prev : [...prev, { id, name: "", script: "" }],
    );
  };

  const removeSponsor = (id: string) => {
    setSponsors((prev) => prev.filter((entry) => entry.id !== id));
  };

  const toggleSlotSelection = (slotId: string) => {
    setSelectedSlotIds((prev) => {
      if (prev.includes(slotId)) {
        return prev.filter((id) => id !== slotId);
      }
      return [...prev, slotId];
    });
    setFocusedSlotId(slotId);
  };

  /** A waveform marker toggles its break and brings the row into view. */
  const onMarker = (slotId: string) => {
    toggleSlotSelection(slotId);
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    document
      .getElementById(`break-${slotId}`)
      ?.scrollIntoView({ block: "nearest", behavior: reduceMotion ? "auto" : "smooth" });
  };

  /**
   * A manual break carries no score, because nothing scored it. It is tagged
   * `manual` so it cannot be read as a detection, and it is selected, since
   * adding it is choosing it.
   */
  const addBreak = (seconds: number) => {
    const slot = makeManualSlot(seconds, manualSlots);
    setManualSlots((prev) => [...prev, slot]);
    setSelectedSlotIds((prev) => (prev.includes(slot.id) ? prev : [...prev, slot.id]));
    setFocusedSlotId(slot.id);
  };

  /** Clone the uploaded voice, once, on demand. Returns the voice id. */
  const ensureVoiceClone = async (task: ScopedTask): Promise<string> => {
    if (voiceId) return voiceId;
    if (!baseAudio) throw new Error("Upload a base audio file before generating audio.");
    setIsCloningVoice(true);
    try {
      const cloneForm = new FormData();
      cloneForm.append("files", baseAudio);
      cloneForm.append(
        "name",
        baseAudio.name ? `${baseAudio.name} Clone` : "Podcast Voice Clone",
      );
      const response = await apiFetch("/api/clone", {
        method: "POST",
        body: cloneForm,
        signal: task.signal,
      });
      if (!response.ok) {
        throw new Error(await readApiError(response, "Voice clone failed."));
      }
      const data = (await response.json()) as { voiceId?: string };
      if (!data.voiceId) throw new Error("Voice clone response missing voiceId.");
      // A clone of the previous file must never become this file's voice; the
      // server may delete it now rather than at its idle timeout.
      if (!task.isCurrent()) void releaseVoice(apiFetch, data.voiceId);
      task.signal.throwIfAborted();
      setVoiceId(data.voiceId);
      return data.voiceId;
    } finally {
      if (task.isCurrent()) setIsCloningVoice(false);
    }
  };

  /**
   * Ad generation is the step that clones a voice, so the rights certification
   * belongs here rather than in front of analysis. A preview's problems are
   * said in its row; a render's in the ad panel.
   */
  const requestGeneration = (mode: GenerationMode, slotIds: string[]) => {
    const slotId = mode === "preview" ? (slotIds[0] ?? null) : null;
    setNote(null);
    const blocked = generationBlockedReason(capabilities);
    if (blocked) {
      setNote({ mode, slotId, tone: "error", text: blocked });
      return;
    }
    if (!isSponsorReady) {
      setNote({ mode, slotId, tone: "error", text: "Add a sponsor name before generating audio." });
      return;
    }
    if (!rightsCertified) {
      setPendingGeneration({ mode, slotIds });
      setRightsAccepted(false);
      setShowRightsModal(true);
      return;
    }
    void handleMerge(mode, slotIds);
  };

  const handleConfirmRights = async () => {
    setShowRightsModal(false);
    setRightsCertified(true);
    setVoiceCloneError("");
    const pending = pendingGeneration;
    setPendingGeneration(null);
    if (pending) {
      await handleMerge(pending.mode, pending.slotIds);
    }
  };

  const cancelRights = () => {
    setShowRightsModal(false);
    setPendingGeneration(null);
  };

  const handleMerge = async (mode: GenerationMode, slotIds?: string[]) => {
    setNote(null);

    const slotIdsToUse =
      slotIds && slotIds.length
        ? slotIds
        : selectedSlotIds.length
          ? selectedSlotIds
          : focusedSlotId
            ? [focusedSlotId]
            : [];
    const slotId = mode === "preview" ? (slotIdsToUse[0] ?? null) : null;
    const fail = (text: string) => setNote({ mode, slotId, tone: "error", text });

    if (!baseAudio) {
      fail("Upload a base audio file for merging.");
      return;
    }
    const slotsToInsert = slotIdsToUse
      .map((id) => slots.find((entry) => entry.id === id))
      .filter((entry): entry is Slot => Boolean(entry));
    if (!slotsToInsert.length) {
      fail("Select at least one break point before generating audio.");
      return;
    }
    if (slotsToInsert.length !== slotIdsToUse.length) {
      // Never render fewer slots than were selected without saying so.
      fail("A selected break point no longer exists. Review your selection and try again.");
      return;
    }

    for (const slot of slotsToInsert) {
      const sponsorId = slotAssignments[slot.id];
      const sponsorEntry =
        sponsors.find((entry) => entry.id === sponsorId) ?? sponsors[0];
      const scriptText = sponsorEntry?.script ?? "";
      if (!scriptText.trim() && !sponsorEntry?.name?.trim()) {
        fail(`Add a sponsor name for ${slotLabel(slot)}.`);
        return;
      }
    }

    if (mode === "preview") {
      setPreviewingSlotId(slotId);
    } else {
      setIsRendering(true);
    }

    const task = fileTasks().begin();
    generationTaskRef.current = task;
    try {
      const activeVoiceId = await ensureVoiceClone(task);
      const slotsInOrder = [...slotsToInsert].sort((a, b) => b.time - a.time);
      let currentAudio: Blob | File = baseAudio;

      for (const slot of slotsInOrder) {
        const sponsorId = slotAssignments[slot.id];
        const sponsorEntry =
          sponsors.find((entry) => entry.id === sponsorId) ?? sponsors[0];
        const scriptText = sponsorEntry?.script ?? "";
        const sponsorName = sponsorEntry?.name ?? "";

        // Tone and language shape copy the server writes; a typed script is
        // spoken verbatim.
        const ttsPayload: Record<string, string | object> = {
          voiceId: activeVoiceId,
          modelId: "eleven_multilingual_v2",
          outputFormat: "mp3_44100_128",
          tone: selectedTone,
          language: selectedLanguage,
        };
        if (scriptText.trim()) {
          ttsPayload.text = scriptText;
        } else {
          ttsPayload.sponsor = { name: sponsorName };
        }

        const response = await apiFetch("/api/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(ttsPayload),
          signal: task.signal,
        });

        if (!response.ok) {
          // 410: the server released this clone (idle too long); forget it so
          // the next attempt clones again instead of failing the same way.
          if (response.status === 410) setVoiceId("");
          throw new Error(await readApiError(response, "TTS request failed."));
        }

        const ttsBlob = await response.blob();
        const mergeForm = new FormData();
        mergeForm.append("audio", currentAudio, "base.mp3");
        mergeForm.append("insert", ttsBlob, "insert.mp3");
        mergeForm.append("insertAt", slot.time.toString());
        mergeForm.append("pause", "0.12");
        if (mode === "preview") {
          mergeForm.append("preview", "1");
          mergeForm.append("previewSeconds", "3");
        }

        const mergeResponse = await apiFetch("/api/merge", {
          method: "POST",
          body: mergeForm,
          signal: task.signal,
        });

        if (!mergeResponse.ok) {
          throw new Error(
            await readApiError(mergeResponse, "Merge request failed."),
          );
        }

        currentAudio = await mergeResponse.blob();
      }

      // Cancelled, or a new file was uploaded, while the last request finished.
      task.signal.throwIfAborted();
      const nextUrl = URL.createObjectURL(currentAudio);
      if (mode === "preview" && slotId) {
        setPreview({ slotId, url: nextUrl });
      } else {
        setRenderUrl(nextUrl);
        const count = slotsToInsert.length;
        setNote({
          mode,
          slotId: null,
          tone: "ok",
          text: `Render complete: ${count} ad break${count === 1 ? "" : "s"} placed.`,
        });
      }
    } catch (ttsError) {
      // The output of an earlier file is discarded, never shown as this one's.
      if (!task.isCurrent()) return;
      if (task.abortReason() === "cancelled") {
        setNote({
          mode,
          slotId,
          tone: "info",
          text: mode === "preview" ? "Preview cancelled." : "Render cancelled.",
        });
        return;
      }
      const message =
        ttsError instanceof TypeError
          ? BACKEND_UNREACHABLE_MESSAGE
          : ttsError instanceof Error
            ? ttsError.message
            : "Merge failed.";
      fail(message);
      if (message.toLowerCase().includes("clone")) setVoiceCloneError(message);
    } finally {
      task.finish();
      if (generationTaskRef.current === task) generationTaskRef.current = null;
      if (task.isCurrent()) {
        if (mode === "preview") {
          setPreviewingSlotId(null);
        } else {
          setIsRendering(false);
        }
      }
    }
  };

  const previewBreak = (slotId: string) => {
    setSelectedSlotIds((prev) => (prev.includes(slotId) ? prev : [...prev, slotId]));
    setFocusedSlotId(slotId);
    requestGeneration("preview", [slotId]);
  };

  const previewing = previewingSlotId !== null;
  const renderBlocked = renderBlockedReason({
    generationBlocked,
    selectedCount,
    hasSponsorName,
    analyzing: isAnalyzing,
    previewing,
  });
  const emptyStateProblems = [capabilities.loadError, authHint].filter(
    (problem): problem is string => Boolean(problem),
  );

  return (
    <div className="app">
      <AppHeader file={baseAudio} durationSeconds={durationForUi} onReplace={startFile} />

      {!baseAudio ? (
        <EmptyState onFile={startFile} problems={emptyStateProblems} />
      ) : (
        <main className="workspace">
          <h1 className="visually-hidden">Ad breaks for {baseAudio.name}</h1>
          <div className="workspace-main">
            <WaveformPanel
              peaks={waveformPeaks}
              decodeFailed={waveformFailed}
              durationSeconds={durationForUi}
              slots={slots}
              selectedIds={selectedSlotIds}
              focusedId={focusedSlotId}
              onMarker={onMarker}
              originalUrl={baseAudioUrl}
            />
            <BreakList
              placement={placement}
              slots={slots}
              selectedIds={selectedSlotIds}
              isAnalyzing={isAnalyzing}
              analysisNotice={analysisNotice}
              authHint={authHint}
              onCancelAnalysis={() => analysisTaskRef.current?.cancel()}
              onAnalyzeAgain={analyzeAgain}
              onToggle={toggleSlotSelection}
              sponsors={sponsors}
              assignments={slotAssignments}
              onAssign={(slotId, sponsorId) =>
                setSlotAssignments((prev) => ({ ...prev, [slotId]: sponsorId }))
              }
              onPreview={previewBreak}
              previewBlocked={generationBlocked}
              generationBusy={previewing || isRendering}
              previewingId={previewingSlotId}
              preview={preview}
              note={note}
              durationSeconds={durationForUi}
              onAddBreak={addBreak}
            />
          </div>
          <AdPanel
            sponsors={sponsors}
            onUpdateSponsor={updateSponsor}
            onAddSponsor={addSponsor}
            onRemoveSponsor={removeSponsor}
            tone={selectedTone}
            onTone={setSelectedTone}
            language={selectedLanguage}
            onLanguage={setSelectedLanguage}
            scriptWriterNote={scriptWriterNote}
            renderLabel={renderButtonLabel(selectedCount, isRendering)}
            renderBlocked={renderBlocked}
            isRendering={isRendering}
            onRender={() => requestGeneration("render", selectedSlotIds)}
            note={note?.mode === "render" ? note : null}
            renderUrl={renderUrl}
            downloadName={downloadName(baseAudio.name)}
            voiceCloneError={voiceCloneError !== note?.text ? voiceCloneError : ""}
            authHint={authHint}
          />
        </main>
      )}

      {isCloningVoice && (
        <ProcessingOverlay onCancel={() => generationTaskRef.current?.cancel()} />
      )}

      {showRightsModal && (
        <RightsModal
          accepted={rightsAccepted}
          onAcceptedChange={setRightsAccepted}
          onCancel={cancelRights}
          onConfirm={() => void handleConfirmRights()}
        />
      )}
    </div>
  );
}

export default App;
