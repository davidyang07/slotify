import type { CSSProperties, DragEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import LandingHero from "./components/LandingHero";
import SlotifyLogo from "./components/SlotifyLogo";
import SoundwaveIcon from "./components/SoundwaveIcon";
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
import {
  combineSlots,
  describeScoreScale,
  describeSource,
  describeTrainingLabels,
  emptyStateMessage,
  makeManualSlot,
  parsePlacementResponse,
  selectionAfterPlacement,
  slotLabel,
  type PlacementResult,
  type Slot,
} from "./lib/recommendations";
import { TaskScope, type ScopedTask } from "./lib/tasks";

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

const timelineSteps = [
  { id: "upload", label: "Upload" },
  { id: "analyze", label: "Analyze" },
  { id: "export", label: "Export" },
];

type PageId = "landing" | "upload" | "analyze" | "export";

type UploadDropzoneProps = {
  id: string;
  title: string;
  subtitle: string;
  helper: string;
  accept?: string;
  multiple?: boolean;
  hasFile?: boolean;
  onFiles: (files: FileList) => void;
};

type Sponsor = {
  id: string;
  name: string;
  script: string;
};

// Pre-computed waveform bar heights using additive sine waves for natural shape
const WAVEFORM_BARS = Array.from({ length: 52 }, (_, i) => {
  const h =
    Math.abs(Math.sin(i / 5.2) * 28) +
    Math.abs(Math.sin(i / 2.1 + 1.3) * 18) +
    Math.abs(Math.sin(i / 8.7 + 0.5) * 16) +
    8;
  return Math.floor(Math.min(80, h));
});

function UploadDropzone({
  id,
  title,
  subtitle,
  helper,
  accept,
  multiple = false,
  hasFile = false,
  onFiles,
}: UploadDropzoneProps) {
  const [isDragActive, setIsDragActive] = useState(false);

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragActive(false);
    if (event.dataTransfer?.files?.length) {
      onFiles(event.dataTransfer.files);
    }
  };

  return (
    <label
      className={`upload-card${isDragActive ? " drag-active" : ""}${
        hasFile ? " has-file" : ""
      }`}
      htmlFor={id}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragActive(true);
      }}
      onDragLeave={() => setIsDragActive(false)}
      onDrop={handleDrop}
    >
      <input
        id={id}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={(event) => {
          if (event.target.files?.length) {
            onFiles(event.target.files);
          }
        }}
      />
      <div className="upload-icon" aria-hidden="true">
        {hasFile ? (
          <svg viewBox="0 0 24 24" role="presentation">
            <path
              d="M9.4 16.2 5.9 12.7a1 1 0 1 1 1.4-1.4l2.4 2.4 6.3-6.3a1 1 0 1 1 1.4 1.4l-7 7a1 1 0 0 1-1.4 0z"
              fill="currentColor"
            />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" role="presentation">
            <path
              d="M12 3a1 1 0 0 1 1 1v8.6l2.3-2.3a1 1 0 1 1 1.4 1.4l-4.01 4a1 1 0 0 1-1.38 0l-4.01-4a1 1 0 0 1 1.42-1.4L11 12.6V4a1 1 0 0 1 1-1z"
              fill="currentColor"
            />
            <path
              d="M5 15a1 1 0 0 1 1 1v2h12v-2a1 1 0 1 1 2 0v3a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1z"
              fill="currentColor"
            />
          </svg>
        )}
      </div>
      <div className="upload-title">{title}</div>
      <div className="upload-subtitle">{subtitle}</div>
      <div className="upload-helper">{helper}</div>
    </label>
  );
}

type FileInfoPanelProps = {
  file: File;
  duration: number | null;
  open: boolean;
  onToggle: () => void;
};

function FileInfoPanel({ file, duration, open, onToggle }: FileInfoPanelProps) {
  const fmtBytes = (b: number) =>
    b < 1024 * 1024 ? `${(b / 1024).toFixed(0)} KB` : `${(b / (1024 * 1024)).toFixed(1)} MB`;
  const fmtDur = (s: number) =>
    `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const ext = (file.name.split(".").pop() ?? "audio").toUpperCase();

  return (
    <div className={`file-info-panel${open ? " open" : ""}`}>
      <button type="button" className="file-info-header" onClick={onToggle}>
        <span className="file-info-name">
          <span className="file-info-led" />
          {file.name}
        </span>
        <span className="file-info-arrow" aria-hidden="true">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path
              d={open ? "M2 8L6 4L10 8" : "M2 4L6 8L10 4"}
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </button>
      <div className="file-info-body">
        <div className="file-info-grid">
          <div className="file-info-item">
            <span className="file-info-key">Format</span>
            <span className="file-info-val">
              <span className="file-info-tag">{ext}</span>
            </span>
          </div>
          <div className="file-info-item">
            <span className="file-info-key">Size</span>
            <span className="file-info-val">{fmtBytes(file.size)}</span>
          </div>
          <div className="file-info-item">
            <span className="file-info-key">Duration</span>
            <span className="file-info-val file-info-mono">
              {duration != null ? fmtDur(duration) : "—"}
            </span>
          </div>
          <div className="file-info-item">
            <span className="file-info-key">Status</span>
            <span className="file-info-val file-info-ready">
              <span className="file-info-led file-info-led-pulse" />
              Ready
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  /** Per-file async work. Reset whenever the uploaded file changes. */
  const fileTasksRef = useRef<TaskScope | null>(null);
  const analysisTaskRef = useRef<ScopedTask | null>(null);
  const generationTaskRef = useRef<ScopedTask | null>(null);
  /** Monotonic, so a removed sponsor's id is never handed out again. */
  const sponsorCounterRef = useRef(1);

  const [activePage, setActivePage] = useState<PageId>("landing");
  const [baseAudio, setBaseAudio] = useState<File | null>(null);
  const [voiceId, setVoiceId] = useState("");
  const [isCloningVoice, setIsCloningVoice] = useState(false);
  const [voiceCloneError, setVoiceCloneError] = useState("");
  const [insertAt, setInsertAt] = useState("0");
  const [analysisError, setAnalysisError] = useState("");
  const [placement, setPlacement] = useState<PlacementResult | null>(null);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const [capabilities, setCapabilities] = useState<Capabilities>(
    UNKNOWN_CAPABILITIES,
  );
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  /** Decoded once per file; drawing reads these and never re-decodes. */
  const [waveformPeaks, setWaveformPeaks] = useState<Float32Array | null>(null);
  /** State, not a ref, so a remounted canvas (back to Analyze) is redrawn. */
  const [waveformCanvas, setWaveformCanvas] =
    useState<HTMLCanvasElement | null>(null);
  const [manualSlots, setManualSlots] = useState<Slot[]>([]);
  const [selectedSlotIds, setSelectedSlotIds] = useState<string[]>([]);
  const [focusedSlotId, setFocusedSlotId] = useState<string | null>(null);
  const [sponsors, setSponsors] = useState<Sponsor[]>([
    { id: "sponsor-1", name: "", script: "" },
  ]);
  const [slotAssignments, setSlotAssignments] = useState<Record<string, string>>(
    {},
  );
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [baseAudioUrl, setBaseAudioUrl] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [isRendering, setIsRendering] = useState(false);
  const [showRightsModal, setShowRightsModal] = useState(false);
  const [rightsAccepted, setRightsAccepted] = useState(false);
  const [rightsCertified, setRightsCertified] = useState(false);
  /** Set while the rights modal is open, so confirming resumes the right action. */
  const [pendingGeneration, setPendingGeneration] = useState<{
    mode: "preview" | "render";
    slotIds: string[];
  } | null>(null);
  const [selectedTone, setSelectedTone] = useState("professional");
  const [selectedLanguage, setSelectedLanguage] = useState("en");
  const [showAddSlot, setShowAddSlot] = useState(false);
  const [newSlotMinutes, setNewSlotMinutes] = useState("0");
  const [newSlotSeconds, setNewSlotSeconds] = useState("0");
  const [showFileInfo, setShowFileInfo] = useState(true);

  const baseAudioName = useMemo(
    () => baseAudio?.name ?? "No file selected.",
    [baseAudio],
  );

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

  const selectedSlots = useMemo(
    () => slots.filter((slot) => selectedSlotIds.includes(slot.id)),
    [slots, selectedSlotIds],
  );
  const focusedSlot = useMemo(
    () => slots.find((slot) => slot.id === focusedSlotId) ?? null,
    [slots, focusedSlotId],
  );
  /** Generation needs a sponsor to read out; analysis does not. */
  const isSponsorReady = Boolean(
    baseAudio && sponsors.some((entry) => entry.name.trim()),
  );
  const formatTime = (seconds: number | null) => {
    if (!Number.isFinite(seconds)) return "0:00";
    const total = Math.max(0, Math.floor(seconds ?? 0));
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  useEffect(() => {
    if (!audioUrl) return;
    audioRef.current?.play().catch(() => undefined);
    return () => {
      URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

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

  const drawWaveform = useCallback(() => {
    const canvas = waveformCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const peaks = waveformPeaks;
    if (!peaks || peaks.length === 0) {
      ctx.fillStyle = "rgba(10, 10, 10, 0.45)";
      ctx.font = "12px 'Source Serif 4', serif";
      ctx.fillText("Upload a base audio file to render waveform.", 12, height / 2);
      return;
    }

    ctx.strokeStyle = "rgba(13, 102, 92, 0.9)";
    ctx.lineWidth = 1;
    const mid = height / 2;
    const step = width / peaks.length;

    for (let i = 0; i < peaks.length; i += 1) {
      const value = peaks[i] ?? 0;
      const barHeight = Math.max(1, value * height * 0.9);
      const x = i * step;
      ctx.beginPath();
      ctx.moveTo(x, mid - barHeight / 2);
      ctx.lineTo(x, mid + barHeight / 2);
      ctx.stroke();
    }

    if (audioDuration && audioDuration > 0) {
      ctx.lineWidth = 2;
      for (const slot of slots) {
        if (!audioDuration || audioDuration <= 0) continue;
        const ratio = slot.time / audioDuration;
        if (!Number.isFinite(ratio)) continue;
        const clamped = Math.max(0, Math.min(1, ratio));
        const x = clamped * (width - 1) + 0.5;
        if (Number.isFinite(x)) {
          ctx.strokeStyle = selectedSlotIds.includes(slot.id)
            ? "rgba(79, 181, 120, 0.85)"
            : "rgba(255, 193, 7, 0.75)";
          ctx.beginPath();
          ctx.moveTo(x, 8);
          ctx.lineTo(x, height - 8);
          ctx.stroke();
        }
      }

      const selectedInsert = Number.parseFloat(insertAt);
      if (
        audioDuration &&
        audioDuration > 0 &&
        Number.isFinite(selectedInsert) &&
        selectedInsert >= 0
      ) {
        const ratio = selectedInsert / audioDuration;
        if (!Number.isFinite(ratio)) return;
        const clamped = Math.max(0, Math.min(1, ratio));
        const x = clamped * (width - 1) + 0.5;
        if (Number.isFinite(x)) {
          ctx.strokeStyle = "rgba(245, 157, 0, 0.95)";
          ctx.beginPath();
          ctx.moveTo(x, 4);
          ctx.lineTo(x, height - 4);
          ctx.stroke();
        }
      }
    }
  }, [
    audioDuration,
    insertAt,
    selectedSlotIds,
    slots,
    waveformCanvas,
    waveformPeaks,
  ]);

  // Decode once per file. This effect used to depend on drawWaveform, so every
  // slot click re-read the file and re-decoded all of it (about 1.3 GB of
  // Float32 for an hour of audio), and overlapping decodes could crash the tab.
  // The per-file reset of peaks and duration happens in selectBaseAudio.
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
        if (!AudioContextCtor) return;
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

  useEffect(() => {
    const handleResize = () => drawWaveform();
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [drawWaveform]);

  // Also runs when the canvas remounts (drawWaveform depends on it), so the
  // waveform is back after Analyze -> Export -> Analyze.
  useEffect(() => {
    drawWaveform();
  }, [drawWaveform]);

  useEffect(() => {
    if (!focusedSlot) return;
    setInsertAt(focusedSlot.time.toFixed(2));
  }, [focusedSlot]);

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

  const fileTasks = () => (fileTasksRef.current ??= new TaskScope());

  /**
   * A new upload starts from nothing. The previous file's in-flight requests
   * are abandoned first, so a render, clone or analysis of the old file cannot
   * land in the new file's state, and then everything derived from it --
   * placement, slots, voice, rendered output, messages -- is cleared.
   */
  const selectBaseAudio = (file: File | null) => {
    if (file === baseAudio) return;
    fileTasks().reset();
    setBaseAudio(file);
    setShowFileInfo(true);
    setAudioDuration(null);
    setWaveformPeaks(null);
    setVoiceId("");
    setVoiceCloneError("");
    setIsCloningVoice(false);
    setPlacement(null);
    setHasAnalyzed(false);
    setAnalysisError("");
    setIsAnalyzing(false);
    setManualSlots([]);
    setSelectedSlotIds([]);
    setFocusedSlotId(null);
    setSlotAssignments({});
    setInsertAt("0");
    setAudioUrl("");
    setStatus("");
    setError("");
    setIsPreviewing(false);
    setIsRendering(false);
    setPendingGeneration(null);
    setShowRightsModal(false);
  };

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

  const selectedSponsorId = focusedSlot
    ? slotAssignments[focusedSlot.id]
    : sponsors[0]?.id;
  const selectedSponsor =
    sponsors.find((entry) => entry.id === selectedSponsorId) ?? sponsors[0];
  const selectedSponsorName = selectedSponsor?.name ?? "";
  const selectedSlotSummary = selectedSlots.length
    ? [...selectedSlots]
        .sort((a, b) => a.time - b.time)
        .map(slotLabel)
        .join(", ")
    : "Not selected";
  const selectedPlacementScore =
    selectedSlots.length === 1 && selectedSlots[0].placementScore !== null
      ? `${selectedSlots[0].placementScore} / 100`
      : "--";
  const generationBlocked = generationBlockedReason(capabilities);
  const authHint =
    capabilities.authRequired && !API_CONFIG.token
      ? "This server requires an API token. Set VITE_API_TOKEN for the frontend and restart it."
      : null;
  const trainingLabelNote = describeTrainingLabels(placement?.provenance ?? null);

  const updateSponsor = (id: string, patch: Partial<Sponsor>) => {
    setSponsors((prev) =>
      prev.map((entry) =>
        entry.id === id ? { ...entry, ...patch } : entry,
      ),
    );
  };

  const addSponsor = () => {
    if (sponsors.length >= 3) return;
    // Built from a counter, not `prev.length + 1`: add, remove Sponsor 1, add
    // used to produce two `sponsor-2` entries sharing a key and input ids.
    sponsorCounterRef.current += 1;
    const id = `sponsor-${sponsorCounterRef.current}`;
    setSponsors((prev) =>
      prev.length >= 3 ? prev : [...prev, { id, name: "", script: "" }],
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

  /**
   * Show a new placement. Manual slots survive it; selections of slots it no
   * longer has are dropped rather than left to be silently skipped at render
   * time, and with nothing left selected the top-ranked slot is selected.
   */
  const applyPlacement = (result: PlacementResult) => {
    const nextSlots = combineSlots(result.slots, manualSlots, audioDuration);
    const nextSelection = selectionAfterPlacement(selectedSlotIds, nextSlots);
    setPlacement(result);
    setHasAnalyzed(true);
    setSelectedSlotIds(nextSelection);
    setFocusedSlotId((prev) =>
      prev && nextSlots.some((slot) => slot.id === prev)
        ? prev
        : (nextSelection[0] ?? null),
    );
  };

  const runPlacementAnalysis = async () => {
    setAnalysisError("");
    if (!baseAudio) {
      setAnalysisError("Upload a base audio file first.");
      return false;
    }

    const task = fileTasks().begin(ANALYSIS_TIMEOUT_MS);
    analysisTaskRef.current = task;
    setIsAnalyzing(true);
    try {
      const form = new FormData();
      form.append("audio", baseAudio);
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
      if (!task.isCurrent()) return false;
      applyPlacement(result);
      if (result.status !== "ok") {
        setAnalysisError(emptyStateMessage(result));
        return true;
      }
      return true;
    } catch (analysisErr) {
      if (!task.isCurrent()) return false;
      const aborted = task.abortReason();
      if (aborted === "cancelled" || aborted === "timeout") {
        setAnalysisError(
          aborted === "timeout"
            ? `Analysis took longer than ${ANALYSIS_TIMEOUT_MS / 60000} minutes and was stopped. Try again, or try a shorter clip.`
            : "Analysis cancelled.",
        );
        return false;
      }
      // fetch() rejects with a TypeError when the backend cannot be reached.
      const message =
        analysisErr instanceof TypeError
          ? BACKEND_UNREACHABLE_MESSAGE
          : analysisErr instanceof Error
            ? analysisErr.message
            : "Insert analysis failed.";
      applyPlacement({
        status: "unavailable",
        slots: [],
        duration: null,
        candidateCount: null,
        provenance: null,
        message,
      });
      setAnalysisError(message);
      return false;
    } finally {
      task.finish();
      if (analysisTaskRef.current === task) analysisTaskRef.current = null;
      if (task.isCurrent()) setIsAnalyzing(false);
    }
  };

  /**
   * Analysis is the first thing that happens after upload and needs no
   * credentials. Voice cloning used to gate it -- the rights modal ran, the
   * ElevenLabs clone ran, and placement only happened if that succeeded, so the
   * whole ranking demo depended on a paid API. Cloning now happens lazily, at
   * the point where audio is actually generated.
   */
  const handleRequestAnalyze = async () => {
    setAnalysisError("");
    if (!baseAudio) {
      setAnalysisError("Add an audio file before continuing.");
      return;
    }
    const ran = await runPlacementAnalysis();
    if (ran) setActivePage("analyze");
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
      // A clone of the previous file must never become this file's voice.
      task.signal.throwIfAborted();
      setVoiceId(data.voiceId);
      return data.voiceId;
    } finally {
      if (task.isCurrent()) setIsCloningVoice(false);
    }
  };

  /**
   * Ad generation is the step that clones a voice, so the rights certification
   * belongs here rather than in front of analysis.
   */
  const requestGeneration = (mode: "preview" | "render", slotIds: string[]) => {
    setError("");
    const blocked = generationBlockedReason(capabilities);
    if (blocked) {
      setError(blocked);
      return;
    }
    if (!isSponsorReady) {
      setError("Add a sponsor name before generating audio.");
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

  const handleMerge = async (
    mode: "preview" | "render",
    slotIds?: string[],
  ) => {
    setError("");
    setStatus("");

    if (!baseAudio) {
      setError("Upload a base audio file for merging.");
      return;
    }

    const slotIdsToUse =
      slotIds && slotIds.length
        ? slotIds
        : selectedSlotIds.length
          ? selectedSlotIds
          : focusedSlotId
            ? [focusedSlotId]
            : [];
    const slotsToInsert = slotIdsToUse
      .map((id) => slots.find((entry) => entry.id === id))
      .filter((entry): entry is Slot => Boolean(entry));
    if (!slotsToInsert.length) {
      setError("Select at least one insertion slot before generating audio.");
      return;
    }
    if (slotsToInsert.length !== slotIdsToUse.length) {
      // Never render fewer slots than were selected without saying so.
      setError("A selected slot no longer exists. Review your selection and try again.");
      return;
    }

    for (const slot of slotsToInsert) {
      const sponsorId = slotAssignments[slot.id];
      const sponsorEntry =
        sponsors.find((entry) => entry.id === sponsorId) ?? sponsors[0];
      const scriptText = sponsorEntry?.script ?? "";
      if (!scriptText.trim() && !sponsorEntry?.name?.trim()) {
        setError(`Add a sponsor name for ${slotLabel(slot)}.`);
        return;
      }
    }

    if (mode === "preview") {
      setIsPreviewing(true);
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
      setAudioUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return nextUrl;
      });
      setStatus(
        mode === "preview"
          ? "Preview generated."
          : "Render complete. Ready to export.",
      );
    } catch (ttsError) {
      // The output of an earlier file is discarded, never shown as this one's.
      if (!task.isCurrent()) return;
      if (task.abortReason() === "cancelled") {
        setError(mode === "preview" ? "Preview cancelled." : "Render cancelled.");
        return;
      }
      const message =
        ttsError instanceof TypeError
          ? BACKEND_UNREACHABLE_MESSAGE
          : ttsError instanceof Error
            ? ttsError.message
            : "Merge failed.";
      setError(message);
      if (message.toLowerCase().includes("clone")) setVoiceCloneError(message);
    } finally {
      task.finish();
      if (generationTaskRef.current === task) generationTaskRef.current = null;
      if (task.isCurrent()) {
        if (mode === "preview") {
          setIsPreviewing(false);
        } else {
          setIsRendering(false);
        }
      }
    }
  };

  /** The processing overlay's Cancel: stop whatever it is waiting on. */
  const cancelProcessing = () => {
    if (isAnalyzing) analysisTaskRef.current?.cancel();
    if (isCloningVoice) generationTaskRef.current?.cancel();
  };

  return (
    <div className={`app page-${activePage}`}>
      <header className="topbar">
        <a href='/' className="logo">
          <SoundwaveIcon size="md" variant={activePage === "landing" ? "light" : "light"} />
          <SlotifyLogo size="md" variant={activePage === "landing" ? "light" : "light"} />
        </a>
        <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
          <div className="nav-steps">
            {timelineSteps.map((step) => (
              <button
                key={step.id}
                type="button"
                className={`nav-button${
                  activePage === step.id ? " active" : ""
                }`}
                disabled={
                  step.id !== "upload" &&
                  step.id !== "landing" &&
                  (!baseAudio || !hasAnalyzed)
                }
                onClick={() => {
                  if (
                    step.id !== "upload" &&
                    step.id !== "landing" &&
                    (!baseAudio || !hasAnalyzed)
                  ) {
                    return;
                  }
                  setActivePage(step.id as PageId);
                }}
              >
                {step.label}
              </button>
            ))}
          </div>
          <a
            href="https://github.com/jweng121/uoft-winners"
            target="_blank"
            rel="noopener noreferrer"
            className="github-link"
            aria-label="View on GitHub"
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="currentColor"
              style={{ display: "block" }}
            >
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
          </a>
        </div>
      </header>

      
      {activePage === "landing" && (
        <LandingHero
          onPrimaryAction={() => setActivePage("upload")}
          onSecondaryAction={() => setActivePage("analyze")}
        />
      )}


      {activePage === "upload" && (
        <section className="page">
          <div className="upload-flow">
            {timelineSteps.map((step, index) => (
              <div key={step.id} className="flow-step">
                <div
                  className={`flow-dot${
                    index === 0 ? " active" : ""
                  }`}
                >
                  {index === 0 ? "↑" : index === 1 ? "✦" : index === 2 ? "▶" : "↓"}
                </div>
                <span>{step.label}</span>
              </div>
            ))}
            <div className="flow-line" />
          </div>
          <div className="page-card">
            <div className="page-header upload-header-row">
              <div className="upload-header">
                <p className="eyebrow">Upload</p>
                <h2>Create a new insertion job.</h2>
                <p className="subtitle">
                  Upload your audio and paste the sponsor script to get started.
                </p>
              </div>
            </div>

            <div className="upload-grid">
              <UploadDropzone
                id="baseAudio"
                title="Upload files"
                subtitle="Drop your audio file here"
                helper={`or click to browse • ${baseAudioName}`}
                accept="audio/*"
                hasFile={Boolean(baseAudio)}
                onFiles={(nextFiles) => selectBaseAudio(nextFiles?.[0] ?? null)}
              />
              {baseAudio && (
                <FileInfoPanel
                  file={baseAudio}
                  duration={audioDuration}
                  open={showFileInfo}
                  onToggle={() => setShowFileInfo((p) => !p)}
                />
              )}
              {baseAudioUrl && (
                <div className="inline-preview">
                  <div className="inline-preview-title">Original audio</div>
                  <audio controls src={baseAudioUrl} />
                </div>
              )}
              <div className="upload-details">
                <div className="field">
                  <label>Sponsor companies</label>
                  <div className="sponsor-list">
                    {sponsors.map((entry, index) => (
                      <div key={entry.id} className="sponsor-card">
                        <div className="sponsor-header">
                          <span>Sponsor {index + 1}</span>
                          {sponsors.length > 1 && (
                            <button
                              type="button"
                              className="ghost small"
                              onClick={() => removeSponsor(entry.id)}
                            >
                              Remove
                            </button>
                          )}
                        </div>
                        <div className="field">
                          <label htmlFor={`${entry.id}-name`}>
                            Company name
                          </label>
                          <input
                            id={`${entry.id}-name`}
                            type="text"
                            placeholder="e.g. Morning Roast Coffee"
                            value={entry.name}
                            onChange={(event) =>
                              updateSponsor(entry.id, {
                                name: event.target.value,
                              })
                            }
                          />
                        </div>
                        <div className="field">
                          <label htmlFor={`${entry.id}-script`}>
                            Brand statement{" "}
                            <span className="optional">(optional)</span>
                          </label>
                          <textarea
                            id={`${entry.id}-script`}
                            placeholder="But before that..."
                            rows={3}
                            value={entry.script}
                            onChange={(event) =>
                              updateSponsor(entry.id, {
                                script: event.target.value,
                              })
                            }
                          />
                          <span className="helper">
                            Keep it to one sentence (~8-12 seconds spoken).
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <button
                    type="button"
                    className="secondary small"
                    onClick={addSponsor}
                    disabled={sponsors.length >= 3}
                  >
                    Add another sponsor
                  </button>
                </div>

                <button
                  type="button"
                  className="primary wide"
                  onClick={() => void handleRequestAnalyze()}
                  disabled={isAnalyzing || !baseAudio}
                >
                  {isAnalyzing ? "Analyzing..." : "Analyze & recommend slots"}
                </button>
                <span className="helper">
                  Analysis runs locally on the server and needs no API keys.
                  Sponsor names are only needed later, when generating audio.
                </span>
                {generationBlocked && (
                  <span className="helper">{generationBlocked}</span>
                )}
                {authHint && (
                  <span className="helper helper-error">{authHint}</span>
                )}
                {analysisError && (
                  <span className="helper helper-error">{analysisError}</span>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {activePage === "analyze" && (
        <section className="page">
          <div className="task-timeline">
            {timelineSteps.map((step, index) => {
              const isActive = step.id === "analyze";
              const isComplete = index === 0;
              return (
                <div key={step.id} className="task-step">
                  <div
                    className={`task-dot${isActive ? " active" : ""}${
                      isComplete ? " complete" : ""
                    }`}
                  >
                    {isComplete ? "✓" : index === 1 ? "✦" : index === 2 ? "▶" : "↓"}
                  </div>
                  <span>{step.label}</span>
                </div>
              );
            })}
            <div className="task-line" />
          </div>
          <div className="page-card">
            <div className="page-header">
              <div>
                <p className="eyebrow">Analyze</p>
                <h2>Recommended insertion points</h2>
                <p className="subtitle">
                  Ranked from the audio's own signals. Select slots to continue.
                </p>
                {placement?.provenance && (
                  <div className="provenance-strip">
                    <span className="provenance-chip">
                      {describeSource(placement.provenance)}
                    </span>
                    <span className="provenance-note">
                      {describeScoreScale(placement.provenance)}
                    </span>
                    {placement.candidateCount !== null && (
                      <span className="provenance-note">
                        {placement.candidateCount} candidate
                        {placement.candidateCount === 1 ? "" : "s"} considered,
                        {" "}
                        {slots.length} shown
                      </span>
                    )}
                    {trainingLabelNote && (
                      <span className="provenance-note provenance-caveat">
                        {trainingLabelNote}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="timeline-card">
              <div className="timeline-title">Timeline Visualization</div>
              <div className="timeline-waveform">
                <div className="waveform timeline-canvas">
                  <canvas ref={setWaveformCanvas} />
                </div>
                <div className="timeline-markers">
                  {slots.map((slot) => (
                    <button
                      key={slot.id}
                      type="button"
                      className={`timeline-dot${
                        selectedSlotIds.includes(slot.id) ? " active" : ""
                      }`}
                      style={{
                        left: audioDuration
                          ? `calc(${Math.min(
                              100,
                              Math.max(0, (slot.time / audioDuration) * 100),
                            )}% - 8px)`
                          : "50%",
                      }}
                      onClick={() => toggleSlotSelection(slot.id)}
                      aria-label={`Select ${slotLabel(slot)}`}
                    />
                  ))}
                </div>
              </div>
              <div className="timeline-labels">
                <span>{formatTime(0)}</span>
                <span>{formatTime(audioDuration ?? 60)}</span>
              </div>
            </div>
            {analysisError && (
              <span className="helper helper-error">{analysisError}</span>
            )}

            {/* Add Slot Section */}
            <div className="add-slot-section">
              <button
                type="button"
                className="secondary"
                onClick={() => setShowAddSlot(!showAddSlot)}
              >
                {showAddSlot ? "Cancel" : "+ Add Custom Slot"}
              </button>
              {showAddSlot && (
                <div className="add-slot-form">
                  <div className="add-slot-inputs">
                    <div className="field">
                      <label htmlFor="slot-minutes">Minutes</label>
                      <input
                        id="slot-minutes"
                        type="number"
                        min="0"
                        value={newSlotMinutes}
                        onChange={(e) => setNewSlotMinutes(e.target.value)}
                        placeholder="0"
                      />
                    </div>
                    <div className="field">
                      <label htmlFor="slot-seconds">Seconds</label>
                      <input
                        id="slot-seconds"
                        type="number"
                        min="0"
                        max="59"
                        value={newSlotSeconds}
                        onChange={(e) => {
                          const val = e.target.value;
                          if (val === "" || (parseInt(val) >= 0 && parseInt(val) <= 59)) {
                            setNewSlotSeconds(val);
                          }
                        }}
                        placeholder="0"
                      />
                    </div>
                  </div>
                  <button
                    type="button"
                    className="primary"
                    onClick={() => {
                      const minutes = parseInt(newSlotMinutes) || 0;
                      const seconds = parseInt(newSlotSeconds) || 0;
                      const totalSeconds = minutes * 60 + seconds;
                      
                      if (totalSeconds < 0) {
                        return;
                      }
                      
                      if (audioDuration && totalSeconds > audioDuration) {
                        alert(`Time cannot exceed audio duration (${formatTime(audioDuration)})`);
                        return;
                      }

                      // A manual slot carries no score, because nothing scored
                      // it. It is tagged `manual` so it cannot be read as a
                      // detection.
                      setManualSlots((prev) => [
                        ...prev,
                        makeManualSlot(totalSeconds, prev),
                      ]);
                      setNewSlotMinutes("0");
                      setNewSlotSeconds("0");
                      setShowAddSlot(false);
                    }}
                    disabled={!newSlotMinutes && !newSlotSeconds}
                  >
                    Add Slot
                  </button>
                </div>
              )}
            </div>

            {slots.length === 0 ? (
              <div className="empty-state">
                <h3>No insertion points to show</h3>
                <p>{emptyStateMessage(placement)}</p>
                <p className="empty-state-note">
                  Nothing is filled in here. When analysis finds one point, one
                  is shown; when it finds none, none are.
                </p>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => void runPlacementAnalysis()}
                  disabled={isAnalyzing || !baseAudio}
                >
                  {isAnalyzing ? "Analyzing..." : "Run analysis again"}
                </button>
              </div>
            ) : (
              <div className="slot-grid">
                {slots.map((slot) => {
                  const scoreClass =
                    slot.placementScore === null
                      ? "badge-manual"
                      : slot.placementScore >= 67
                        ? "badge-high"
                        : slot.placementScore >= 34
                          ? "badge-mid"
                          : "badge-low";
                  return (
                    <div
                      key={slot.id}
                      className={`slot-preview${
                        selectedSlotIds.includes(slot.id) ? " active" : ""
                      }`}
                    >
                      <div className="slot-preview-top">
                        <div>
                          <div className="slot-preview-label">
                            {slot.rank ? `Rank ${slot.rank}` : "Manual slot"}
                          </div>
                          <div className="slot-preview-time">
                            {formatTime(slot.time)}
                          </div>
                        </div>
                        <span className={`slot-badge ${scoreClass}`}>
                          {slot.placementScore === null
                            ? "manual"
                            : `${slot.placementScore}/100`}
                        </span>
                      </div>
                      <div className="slot-source">
                        {slot.source === "learned_ranker"
                          ? "Learned ranker"
                          : slot.source === "manual"
                            ? "Added by you"
                            : "Heuristic baseline"}
                      </div>
                      <div className="slot-select">
                        <label htmlFor={`${slot.id}-sponsor`}>
                          Insert brand statement
                        </label>
                        <select
                          id={`${slot.id}-sponsor`}
                          value={slotAssignments[slot.id] ?? sponsors[0]?.id ?? ""}
                          onChange={(event) =>
                            setSlotAssignments((prev) => ({
                              ...prev,
                              [slot.id]: event.target.value,
                            }))
                          }
                        >
                          {sponsors.map((entry, index) => (
                            <option key={entry.id} value={entry.id}>
                              {entry.name.trim()
                                ? entry.name
                                : `Sponsor ${index + 1}`}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="slot-preview-notes">
                        {slot.signals.length === 0 && (
                          <div className="note">
                            No signal was measured for this point.
                          </div>
                        )}
                        {slot.signals.map((signal, signalIndex) => (
                          <div
                            key={`${slot.id}-${signalIndex}`}
                            className={`note note-${signal.kind}`}
                          >
                            <span
                              className={`note-icon ${
                                signal.kind === "supporting" ? "up" : "down"
                              }`}
                            >
                              {signal.kind === "supporting" ? "OK" : "!"}
                            </span>
                            {signal.label}
                          </div>
                        ))}
                      </div>
                      {slot.rationale && (
                        <p className="slot-rationale">{slot.rationale}</p>
                      )}
                      <div className="slot-preview-actions">
                        <button
                          type="button"
                          className="ghost"
                          onClick={() => {
                            setSelectedSlotIds((prev) =>
                              prev.includes(slot.id) ? prev : [...prev, slot.id],
                            );
                            setFocusedSlotId(slot.id);
                            requestGeneration("preview", [slot.id]);
                          }}
                          disabled={isPreviewing || Boolean(generationBlocked)}
                          title={generationBlocked ?? undefined}
                        >
                          Preview
                        </button>
                        <button
                          type="button"
                          className={`primary select-slot${
                            selectedSlotIds.includes(slot.id) ? " selected" : ""
                          }`}
                          onClick={() => toggleSlotSelection(slot.id)}
                        >
                          {selectedSlotIds.includes(slot.id)
                            ? "Selected"
                            : "Select Slot"}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Voice Settings Section */}
            <div className="voice-settings-section">
              <div className="voice-settings-header">
                <h3>Voice Settings</h3>
                <p className="voice-settings-subtitle">
                  Customize the tone and language of your sponsor reads
                </p>
              </div>
              <div className="voice-settings-grid">
                <div className="voice-setting-field">
                  <label htmlFor="tone-select">Tone</label>
                  <select
                    id="tone-select"
                    value={selectedTone}
                    onChange={(e) => setSelectedTone(e.target.value)}
                    className="voice-setting-select"
                  >
                    <option value="professional">Professional</option>
                    <option value="friendly-casual">Friendly & Casual</option>
                    <option value="energetic">Energetic</option>
                    <option value="serious">Serious</option>
                    <option value="warm">Warm</option>
                    <option value="conversational">Conversational</option>
                    <option value="enthusiastic">Enthusiastic</option>
                  </select>
                  <span className="helper">
                    Applies to scripts Slotify writes for you. A brand
                    statement you typed is read exactly as written.
                  </span>
                </div>
                <div className="voice-setting-field">
                  <label htmlFor="language-select">Language</label>
                  <select
                    id="language-select"
                    value={selectedLanguage}
                    onChange={(e) => setSelectedLanguage(e.target.value)}
                    className="voice-setting-select"
                  >
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="it">Italian</option>
                    <option value="pt">Portuguese</option>
                    <option value="ja">Japanese</option>
                    <option value="ko">Korean</option>
                    <option value="zh">Chinese</option>
                  </select>
                  <span className="helper">
                    Applies to scripts Slotify writes for you. A brand
                    statement you typed is spoken verbatim, in the language
                    you wrote it in.
                  </span>
                </div>
              </div>
            </div>

            {audioUrl && (
              <div className="inline-preview">
                <div className="inline-preview-title">Preview playback</div>
                <audio ref={audioRef} controls src={audioUrl} />
              </div>
            )}
            <button
              type="button"
              className="primary wide"
              onClick={() => setActivePage("export")}
            >
              Confirm selections
            </button>
          </div>
        </section>
      )}

      {activePage === "export" && (
        <section className="page">
          <div className="page-card">
            <div className="page-header">
              <div>
                <p className="eyebrow">Export</p>
                <h2>Render the final placement.</h2>
                <p className="subtitle">
                  Review the ranking and export the final merged file.
                </p>
              </div>
              <button
                type="button"
                className="ghost"
                onClick={() => setActivePage("landing")}
              >
                Back to start
              </button>
            </div>

            <div className="export-grid">
              <div className="export-card">
                <div className="summary-item">
                  <span>Selected slots</span>
                  <strong>{selectedSlotSummary}</strong>
                </div>
                <div className="summary-item">
                  <span>Placement score</span>
                  <strong>{selectedPlacementScore}</strong>
                </div>
                <div className="summary-item">
                  <span>Ranked by</span>
                  <strong>{describeSource(placement?.provenance ?? null)}</strong>
                </div>
                <div className="summary-item">
                  <span>Sponsor</span>
                  <strong>
                    {selectedSponsorName || "Add sponsor name"}
                  </strong>
                </div>
                <button
                  type="button"
                  className="primary"
                  onClick={() => requestGeneration("render", selectedSlotIds)}
                  disabled={isRendering || Boolean(generationBlocked)}
                  title={generationBlocked ?? undefined}
                >
                  {isRendering ? "Rendering..." : "Render & export"}
                </button>
                {generationBlocked && (
                  <div className="helper">{generationBlocked}</div>
                )}
                {voiceCloneError && (
                  <div className="helper helper-error">{voiceCloneError}</div>
                )}
                {isRendering && <div className="loader" />}
                {status && <div className="helper">{status}</div>}
                
                {/* Preview and Download section - shown after rendering */}
                {audioUrl && !isRendering && (
                  <div style={{ marginTop: "24px", paddingTop: "24px", borderTop: "1px solid #e0e0e0" }}>
                    <div style={{ marginBottom: "16px" }}>
                      <h3 style={{ margin: "0 0 12px 0", fontSize: "16px", fontWeight: "600" }}>
                        Final Audio Preview
                      </h3>
                      <audio 
                        ref={audioRef} 
                        controls 
                        src={audioUrl}
                        style={{ width: "100%", marginBottom: "16px" }}
                      />
                    </div>
                    <button
                      type="button"
                      className="primary"
                      onClick={() => {
                        const link = document.createElement("a");
                        link.href = audioUrl;
                        link.download = `merged-audio-${Date.now()}.mp3`;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                      }}
                    >
                      Download Audio
                    </button>
                  </div>
                )}
              </div>
              <div className="export-note">
                <h3>Export notes</h3>
                <p>
                  The final file is generated with the selected ad slot and
                  brand script. You can re-run the render after editing the
                  script or selecting a new slot.
                </p>
                {audioUrl && !isRendering && (
                  <p style={{ marginTop: "12px", color: "#666" }}>
                    Preview the full audio above and download when ready.
                  </p>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {(error || status) && (
        <section className="status-bar">
          {status && <p className="status-ok">{status}</p>}
          {error && <p className="status-error">{error}</p>}
        </section>
      )}

      {(isCloningVoice || isAnalyzing) && (
        <div className="processing-screen" aria-live="polite" aria-label="Processing audio">
          <div className="processing-waveform">
            {WAVEFORM_BARS.map((h, i) => (
              <div
                key={i}
                className="processing-bar"
                style={
                  {
                    "--peak-h": `${h}px`,
                    animationDelay: `${(i * 0.038) % 0.95}s`,
                    animationDuration: `${0.55 + (i % 7) * 0.06}s`,
                  } as CSSProperties
                }
              />
            ))}
          </div>
          <div className="processing-text">
            <p className="processing-status">
              {isCloningVoice ? "Cloning voice signature" : "Finding insertion points"}
            </p>
            <p className="processing-substatus">
              {isCloningVoice
                ? "Analyzing vocal characteristics from your audio"
                : "Scanning the timeline for optimal placement windows"}
            </p>
            <div className="processing-dots">
              <div className="processing-dot" />
              <div className="processing-dot" />
              <div className="processing-dot" />
            </div>
            <button
              type="button"
              className="ghost small processing-cancel"
              onClick={cancelProcessing}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {showRightsModal && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal-card">
            <div className="modal-header">
              <span className="modal-title">
                <span className="modal-icon">⚠</span>
                Voice Rights Certification
              </span>
              <button
                type="button"
                className="icon-button"
                onClick={() => setShowRightsModal(false)}
                aria-label="Close"
              >
                ×
              </button>
            </div>
            <p className="modal-body">
              Placement analysis has already run without touching your voice.
              This step clones it to generate the sponsor read, so please
              confirm you have the necessary rights.
            </p>
            <label className="modal-check">
              <input
                type="checkbox"
                checked={rightsAccepted}
                onChange={(event) => setRightsAccepted(event.target.checked)}
              />
              <span>
                I certify that I own the rights to this audio and voice. I
                understand that <SlotifyLogo size="sm" variant="light" /> will generate sponsor audio using
                voice cloning technology, and I confirm that no unauthorized
                voice impersonation is involved.
              </span>
            </label>
            <div className="modal-actions">
              <button
                type="button"
                className="ghost"
                onClick={() => {
                  setShowRightsModal(false);
                  setPendingGeneration(null);
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                className="primary"
                disabled={!rightsAccepted}
                onClick={() => void handleConfirmRights()}
              >
                Continue to generation
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
