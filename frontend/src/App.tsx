import type { DragEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

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

type Slot = {
  id: string;
  time: number;
  confidence: number;
};

type InsertSuggestion = {
  time: number;
  confidence: number;
};

type Sponsor = {
  id: string;
  name: string;
  script: string;
};

const slotNotes = [
  [
    "Natural topic shift detected",
    "Clean pause boundary (0.8s)",
    "Slightly early in content",
  ],
  [
    "Extended silence detected (1.2s)",
    "Mid-episode engagement peak",
    "Minor audio level mismatch",
  ],
  [
    "Audio energy valley",
    "Speaker breath pause",
    "Near existing music transition",
  ],
];

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
              d="M12 3a1 1 0 0 1 1 1v8.6l2.3-2.3a1 1 0 1 1 1.4 1.4l-4.01 4a1 1 0 0 1-1.38 0l-4.01-4a1 1 0 1 1 1.42-1.4L11 12.6V4a1 1 0 0 1 1-1z"
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

function App() {
  const apiBase =
    import.meta.env.VITE_API_BASE_URL ?? "http://localhost:3001";
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const waveformRef = useRef<HTMLCanvasElement | null>(null);
  const waveformPeaksRef = useRef<Float32Array | null>(null);

  const [activePage, setActivePage] = useState<PageId>("landing");
  const [baseAudio, setBaseAudio] = useState<File | null>(null);
  const [voiceId, setVoiceId] = useState("");
  const [isCloningVoice, setIsCloningVoice] = useState(false);
  const [voiceCloneError, setVoiceCloneError] = useState("");
  const [insertAt, setInsertAt] = useState("0");
  const [analysisError, setAnalysisError] = useState("");
  const [insertSuggestions, setInsertSuggestions] = useState<InsertSuggestion[]>(
    [],
  );
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  const [slots, setSlots] = useState<Slot[]>([]);
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
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [isRendering, setIsRendering] = useState(false);
  const [showRightsModal, setShowRightsModal] = useState(false);
  const [rightsAccepted, setRightsAccepted] = useState(false);
  const [rightsCertified, setRightsCertified] = useState(false);

  const baseAudioName = useMemo(
    () => baseAudio?.name ?? "No file selected.",
    [baseAudio],
  );

  const selectedSlots = useMemo(
    () => slots.filter((slot) => selectedSlotIds.includes(slot.id)),
    [slots, selectedSlotIds],
  );
  const focusedSlot = useMemo(
    () => slots.find((slot) => slot.id === focusedSlotId) ?? null,
    [slots, focusedSlotId],
  );
  const isUploadReady = Boolean(
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

  const drawWaveform = () => {
    const canvas = waveformRef.current;
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

    const peaks = waveformPeaksRef.current;
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
        const x = (slot.time / audioDuration) * width;
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
      if (Number.isFinite(selectedInsert) && selectedInsert >= 0) {
        const x = (selectedInsert / audioDuration) * width;
        if (Number.isFinite(x)) {
          ctx.strokeStyle = "rgba(245, 157, 0, 0.95)";
          ctx.beginPath();
          ctx.moveTo(x, 4);
          ctx.lineTo(x, height - 4);
          ctx.stroke();
        }
      }
    }
  };

  useEffect(() => {
    if (!baseAudio) {
      waveformPeaksRef.current = null;
      setInsertSuggestions([]);
      setAudioDuration(null);
      drawWaveform();
      return;
    }

    let cancelled = false;
    const buildWaveform = async () => {
      try {
        const arrayBuffer = await baseAudio.arrayBuffer();
        if (cancelled) return;
        const AudioContextCtor =
          window.AudioContext ||
          (window as typeof window & {
            webkitAudioContext?: typeof AudioContext;
          }).webkitAudioContext;
        if (!AudioContextCtor) {
          waveformPeaksRef.current = null;
          drawWaveform();
          return;
        }
        const context = new AudioContextCtor();
        const audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0));
        const channel = audioBuffer.getChannelData(0);
        setAudioDuration(audioBuffer.duration);
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
        waveformPeaksRef.current = peaks;
        await context.close();
        if (!cancelled) {
          drawWaveform();
        }
      } catch (waveError) {
        waveformPeaksRef.current = null;
        setAudioDuration(null);
        drawWaveform();
      }
    };

    buildWaveform();
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
  }, []);

  useEffect(() => {
    drawWaveform();
  }, [insertAt, slots, audioDuration, selectedSlotIds]);

  useEffect(() => {
    const fallbackTimes =
      audioDuration && audioDuration > 0
        ? [0.22, 0.48, 0.72].map((ratio) => ratio * audioDuration)
        : [12, 24, 36];
    const fallbackSlots = fallbackTimes.map((time, index) => ({
      time,
      confidence: [92, 85, 78][index] ?? 72,
    }));
    const suggestions =
      insertSuggestions.length >= 3
        ? insertSuggestions.slice(0, 3)
        : fallbackSlots;
    const nextSlots = suggestions.map((entry, index) => ({
      id: `slot-${index + 1}`,
      time: entry.time,
      confidence: entry.confidence,
    }));
    setSlots(nextSlots);
    if (!selectedSlotIds.length && nextSlots.length) {
      setSelectedSlotIds([nextSlots[0].id]);
      setFocusedSlotId(nextSlots[0].id);
    }
  }, [insertSuggestions, audioDuration, selectedSlotIds.length]);

  useEffect(() => {
    if (!focusedSlot) return;
    setInsertAt(focusedSlot.time.toFixed(2));
  }, [focusedSlot]);

  useEffect(() => {
    setRightsCertified(false);
  }, [baseAudio, sponsors]);

  useEffect(() => {
    setVoiceId("");
    setVoiceCloneError("");
  }, [baseAudio]);

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
        .map((slot) => `Slot ${slot.id.replace("slot-", "")}`)
        .join(", ")
    : "Not selected";
  const selectedConfidence =
    selectedSlots.length === 1 ? `${selectedSlots[0].confidence}%` : "--";

  const updateSponsor = (id: string, patch: Partial<Sponsor>) => {
    setSponsors((prev) =>
      prev.map((entry) =>
        entry.id === id ? { ...entry, ...patch } : entry,
      ),
    );
  };

  const addSponsor = () => {
    setSponsors((prev) => {
      if (prev.length >= 3) return prev;
      const nextIndex = prev.length + 1;
      return [
        ...prev,
        { id: `sponsor-${nextIndex}`, name: "", script: "" },
      ];
    });
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

  const handleAnalyzeInsert = async () => {
    setAnalysisError("");
    if (!baseAudio) {
      setAnalysisError("Upload a base audio file first.");
      return;
    }

    setIsAnalyzing(true);
    try {
      const form = new FormData();
      form.append("audio", baseAudio);
      form.append("count", "3");

      const response = await fetch(`${apiBase}/api/insert-sections`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Insert analysis failed.");
      }

      const data = (await response.json()) as {
        points?: Array<number | { time?: number; confidence?: number }>;
        confidences?: number[];
      };
      const points = Array.isArray(data.points) ? data.points : [];
      const confidences = Array.isArray(data.confidences)
        ? data.confidences
        : [];
      const suggestions = points
        .map((value, index) => {
          if (value && typeof value === "object") {
            const time = Number(value.time);
            const confidence = Number(value.confidence);
            return {
              time,
              confidence: Number.isFinite(confidence) ? confidence : 72,
            };
          }
          const time = Number(value);
          const confidence = Number(confidences[index]);
          return {
            time,
            confidence: Number.isFinite(confidence) ? confidence : 72,
          };
        })
        .filter((entry) => Number.isFinite(entry.time));
      setInsertSuggestions(suggestions);
      if (suggestions.length === 0) {
        setAnalysisError("No insert points returned.");
      }
    } catch (analysisErr) {
      setAnalysisError(
        analysisErr instanceof Error
          ? analysisErr.message
          : "Insert analysis failed.",
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRequestAnalyze = () => {
    setAnalysisError("");
    if (!isUploadReady) {
      setAnalysisError(
        "Add the audio file and at least one sponsor name before continuing.",
      );
      return;
    }
    setRightsAccepted(false);
    setShowRightsModal(true);
  };

  const handleConfirmRights = async () => {
    setShowRightsModal(false);
    setRightsCertified(true);
    setVoiceCloneError("");
    if (!baseAudio) {
      setError("Upload a base audio file before cloning voice.");
      return;
    }

    setIsCloningVoice(true);
    try {
      const cloneForm = new FormData();
      cloneForm.append("files", baseAudio);
      cloneForm.append(
        "name",
        baseAudio.name ? `${baseAudio.name} Clone` : "Podcast Voice Clone",
      );
      const response = await fetch(`${apiBase}/api/clone`, {
        method: "POST",
        body: cloneForm,
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Voice clone failed.");
      }
      const data = (await response.json()) as { voiceId?: string };
      if (!data.voiceId) {
        throw new Error("Voice clone response missing voiceId.");
      }
      setVoiceId(data.voiceId);
    } catch (cloneErr) {
      const message =
        cloneErr instanceof Error ? cloneErr.message : "Voice clone failed.";
      setVoiceCloneError(message);
      setError(message);
      setIsCloningVoice(false);
      return;
    }

    setIsCloningVoice(false);
    await handleAnalyzeInsert();
    setActivePage("analyze");
  };

  const handleMerge = async (
    mode: "preview" | "render",
    slotIds?: string[],
  ) => {
    setError("");
    setStatus("");

    if (!voiceId) {
      setError("Voice clone not ready yet. Finish cloning from your upload.");
      return;
    }

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

    for (const slot of slotsToInsert) {
      const sponsorId = slotAssignments[slot.id];
      const sponsorEntry =
        sponsors.find((entry) => entry.id === sponsorId) ?? sponsors[0];
      const scriptText = sponsorEntry?.script ?? "";
      if (!scriptText.trim()) {
        setError(`Add a sponsor statement for ${slot.id}.`);
        return;
      }
    }

    if (mode === "preview") {
      setIsPreviewing(true);
    } else {
      setIsRendering(true);
    }

    try {
      const slotsInOrder = [...slotsToInsert].sort((a, b) => b.time - a.time);
      let currentAudio: Blob | File = baseAudio;

      for (const slot of slotsInOrder) {
        const sponsorId = slotAssignments[slot.id];
        const sponsorEntry =
          sponsors.find((entry) => entry.id === sponsorId) ?? sponsors[0];
        const scriptText = sponsorEntry?.script ?? "";

        const response = await fetch(`${apiBase}/api/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            voiceId,
            text: scriptText,
            modelId: "eleven_multilingual_v2",
            outputFormat: "mp3_44100_128",
          }),
        });

        if (!response.ok) {
          const message = await response.text();
          throw new Error(message || "TTS request failed.");
        }

        const ttsBlob = await response.blob();
        const mergeForm = new FormData();
        mergeForm.append("audio", currentAudio, "base.mp3");
        mergeForm.append("insert", ttsBlob, "insert.mp3");
        mergeForm.append("insertAt", slot.time.toString());
        mergeForm.append("pause", "0.12");

        const mergeResponse = await fetch(`${apiBase}/api/merge`, {
          method: "POST",
          body: mergeForm,
        });

        if (!mergeResponse.ok) {
          const message = await mergeResponse.text();
          throw new Error(message || "Merge request failed.");
        }

        currentAudio = await mergeResponse.blob();
      }

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
      setError(
        ttsError instanceof Error ? ttsError.message : "Merge failed.",
      );
    } finally {
      setIsPreviewing(false);
      setIsRendering(false);
    }
  };

  return (
    <div className={`app page-${activePage}`}>
      <header className="topbar">
        <a href='/' className="logo">Sl|lotify</a>
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
                (!isUploadReady || !rightsCertified)
              }
              onClick={() => {
                if (
                  step.id !== "upload" &&
                  step.id !== "landing" &&
                  (!isUploadReady || !rightsCertified)
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
      </header>

      {activePage === "landing" && (
        <section className="page landing">
          <div className="landing-hero">
            <div className="hero-pill">
              <span className="hero-dot" />
              AI-Powered Audio Insertion
            </div>
            <a className="hero-title">Sl|lotify</a>
            <p className="hero-subtitle">
              Seamless sponsor insertion for audio.
            </p>
            <p className="hero-body">
              Upload an episode, paste an ad read, and Sl|lotify finds the best
              slot and inserts it naturally — in the creator&apos;s own voice.
            </p>
            <div className="hero-actions">
              <button
                type="button"
                className="primary hero-primary"
                onClick={() => setActivePage("upload")}
              >
                Try Sl|lotify
              </button>
              <button
                type="button"
                className="ghost hero-ghost"
                onClick={() => setActivePage("analyze")}
              >
                How it works
                <span className="chevron">▾</span>
              </button>
            </div>
            <div className="hero-progress">
              <div className="progress-bar">
                <span className="progress-fill" />
                <span className="progress-spot" />
              </div>
              <div className="progress-labels">
                <span>Main Audio</span>
                <span>Sponsor Insert</span>
                <span>Main Audio</span>
              </div>
            </div>
            <div className="hero-bars" aria-hidden="true">
              {Array.from({ length: 9 }).map((_, index) => (
                <span key={`bar-${index}`} className="hero-bar" />
              ))}
            </div>
            <button
              type="button"
              className="hero-scroll"
              onClick={() => setActivePage("upload")}
            >
              ⌄
            </button>
          </div>
        </section>
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
                onFiles={(nextFiles) =>
                  setBaseAudio(nextFiles?.[0] ?? null)
                }
              />
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
                  onClick={handleRequestAnalyze}
                  disabled={isAnalyzing || !isUploadReady}
                >
                  {isAnalyzing ? "Analyzing..." : "Analyze & recommend slots"}
                </button>
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
                  Optimized for smooth transitions and listener retention.
                  Select slots to continue.
                </p>
              </div>
            </div>

            <div className="timeline-card">
              <div className="timeline-title">Timeline Visualization</div>
              <div className="timeline-waveform">
                <div className="waveform timeline-canvas">
                  <canvas ref={waveformRef} />
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
                          ? `${(slot.time / audioDuration) * 100}%`
                          : "50%",
                      }}
                      onClick={() => toggleSlotSelection(slot.id)}
                      aria-label={`Select ${slot.id}`}
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
            <div className="slot-grid">
              {slots.map((slot, index) => {
                const notes = slotNotes[index] ?? [];
                const badgeClass =
                  slot.confidence >= 90
                    ? "badge-high"
                    : slot.confidence >= 80
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
                          Slot {slot.id.replace("slot-", "")}
                        </div>
                        <div className="slot-preview-time">
                          {formatTime(slot.time)}
                        </div>
                      </div>
                      <span className={`slot-badge ${badgeClass}`}>
                        {slot.confidence}%
                      </span>
                    </div>
                    <div className="slot-select">
                      <label htmlFor={`${slot.id}-sponsor`}>
                        Insert brand statement
                      </label>
                      <select
                        id={`${slot.id}-sponsor`}
                        value={
                          slotAssignments[slot.id] ?? sponsors[0]?.id ?? ""
                        }
                        onChange={(event) =>
                          setSlotAssignments((prev) => ({
                            ...prev,
                            [slot.id]: event.target.value,
                          }))
                        }
                      >
                        {sponsors.map((entry) => (
                          <option key={entry.id} value={entry.id}>
                            {entry.name.trim()
                              ? entry.name
                              : `Sponsor ${entry.id.replace("sponsor-", "")}`}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="slot-preview-notes">
                      {notes.map((note, noteIndex) => (
                        <div key={`${slot.id}-${noteIndex}`} className="note">
                          <span
                            className={`note-icon ${
                              noteIndex === notes.length - 1 ? "down" : "up"
                            }`}
                          >
                            {noteIndex === notes.length - 1 ? "↓" : "✓"}
                          </span>
                          {note}
                        </div>
                      ))}
                    </div>
                    <div className="slot-preview-actions">
                      <button
                        type="button"
                        className="ghost"
                        onClick={() => {
                          setSelectedSlotIds((prev) =>
                            prev.includes(slot.id) ? prev : [...prev, slot.id],
                          );
                          setFocusedSlotId(slot.id);
                          handleMerge("preview", [slot.id]);
                        }}
                        disabled={isPreviewing}
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
                  Review confidence and export the final merged file.
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
                  <span>Confidence</span>
                  <strong>{selectedConfidence}</strong>
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
                  onClick={() => handleMerge("render", selectedSlotIds)}
                  disabled={isRendering}
                >
                  {isRendering ? "Rendering..." : "Render & export"}
                </button>
                {isRendering && <div className="loader" />}
                {status && <div className="helper">{status}</div>}
              </div>
              <div className="export-note">
                <h3>Export notes</h3>
                <p>
                  The final file is generated with the selected ad slot and
                  brand script. You can re-run the render after editing the
                  script or selecting a new slot.
                </p>
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
              Before proceeding, please confirm you have the necessary rights.
            </p>
            <label className="modal-check">
              <input
                type="checkbox"
                checked={rightsAccepted}
                onChange={(event) => setRightsAccepted(event.target.checked)}
              />
              <span>
                I certify that I own the rights to this audio and voice. I
                understand that Sl|lotify will generate sponsor audio using
                voice cloning technology, and I confirm that no unauthorized
                voice impersonation is involved.
              </span>
            </label>
            <div className="modal-actions">
              <button
                type="button"
                className="ghost"
                onClick={() => setShowRightsModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="primary"
                disabled={!rightsAccepted}
                onClick={handleConfirmRights}
              >
                Continue to Analysis
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
