import type { DragEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const sampleScripts = [
  {
    title: "Morning Roast",
    text: "",
  },
  {
    title: "City Sprint",
    text: "Run on City Sprint sneakers. Lightweight, quiet, and built for your every day commute.",
  },
  {
    title: "Glow Skin",
    text: "Glow Skin Serum is a daily reset for your skin barrier. Clean ingredients, real results.",
  },
];

const timelineSteps = [
  { id: "upload", label: "Upload" },
  { id: "analyze", label: "Analyze" },
  { id: "preview", label: "Preview" },
  { id: "export", label: "Export" },
];

type PageId = "landing" | "upload" | "analyze" | "preview" | "export";

type UploadDropzoneProps = {
  id: string;
  title: string;
  subtitle: string;
  helper: string;
  accept?: string;
  multiple?: boolean;
  onFiles: (files: FileList) => void;
};

type Slot = {
  id: string;
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
      className={`upload-card${isDragActive ? " drag-active" : ""}`}
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
  const [insertAt, setInsertAt] = useState("0");
  const [analysisError, setAnalysisError] = useState("");
  const [insertSuggestions, setInsertSuggestions] = useState<number[]>([]);
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  const [slots, setSlots] = useState<Slot[]>([]);
  const [selectedSlotId, setSelectedSlotId] = useState<string | null>(null);
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

  const selectedSlot = useMemo(
    () => slots.find((slot) => slot.id === selectedSlotId) ?? null,
    [slots, selectedSlotId],
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
          ctx.strokeStyle = "rgba(255, 193, 7, 0.75)";
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
  }, [insertAt, slots, audioDuration]);

  useEffect(() => {
    const fallbackTimes =
      audioDuration && audioDuration > 0
        ? [0.22, 0.48, 0.72].map((ratio) => ratio * audioDuration)
        : [12, 24, 36];
    const times =
      insertSuggestions.length >= 3
        ? insertSuggestions.slice(0, 3)
        : fallbackTimes;
    const confidences = [92, 85, 78];
    const nextSlots = times.map((time, index) => ({
      id: `slot-${index + 1}`,
      time,
      confidence: confidences[index] ?? 72,
    }));
    setSlots(nextSlots);
    if (!selectedSlotId && nextSlots.length) {
      setSelectedSlotId(nextSlots[0].id);
    }
  }, [insertSuggestions, audioDuration, selectedSlotId]);

  useEffect(() => {
    if (!selectedSlot) return;
    setInsertAt(selectedSlot.time.toFixed(2));
  }, [selectedSlot]);

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

  const selectedSponsorId = selectedSlot
    ? slotAssignments[selectedSlot.id]
    : sponsors[0]?.id;
  const selectedSponsor =
    sponsors.find((entry) => entry.id === selectedSponsorId) ?? sponsors[0];
  const selectedScript = selectedSponsor?.script ?? "";
  const selectedSponsorName = selectedSponsor?.name ?? "";

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

      const data = (await response.json()) as { points?: number[] };
      const points = Array.isArray(data.points) ? data.points : [];
      setInsertSuggestions(points);
      if (points.length === 0) {
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
    await handleAnalyzeInsert();
    setActivePage("analyze");
  };

  const handleMerge = async (mode: "preview" | "render", slotId?: string) => {
    setError("");
    setStatus("");

    if (!voiceId) {
      setError("Add a voice ID before generating audio.");
      return;
    }

    if (!baseAudio) {
      setError("Upload a base audio file for merging.");
      return;
    }

    const slot =
      (slotId ? slots.find((entry) => entry.id === slotId) : selectedSlot) ??
      null;
    if (!slot) {
      setError("Select an insertion slot before generating audio.");
      return;
    }

    const sponsorId = slotAssignments[slot.id];
    const sponsorEntry =
      sponsors.find((entry) => entry.id === sponsorId) ?? sponsors[0];
    const scriptText = sponsorEntry?.script ?? "";
    if (!scriptText.trim()) {
      setError("Add a sponsor statement for the selected slot.");
      return;
    }

    const insertAtSeconds = Number.parseFloat(slot.time.toString());
    if (!Number.isFinite(insertAtSeconds) || insertAtSeconds < 0) {
      setError("Insert time must be a positive number.");
      return;
    }

    if (mode === "preview") {
      setIsPreviewing(true);
    } else {
      setIsRendering(true);
    }

    try {
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
      mergeForm.append("audio", baseAudio);
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

      const mergedBlob = await mergeResponse.blob();
      const nextUrl = URL.createObjectURL(mergedBlob);
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
        <div className="logo">Sl|lotify</div>
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
            <h1 className="hero-title">Sl|lotify</h1>
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
                  Select a slot to continue.
                </p>
              </div>
              <div className="header-actions">
                <button
                  type="button"
                  className="secondary"
                  onClick={handleAnalyzeInsert}
                  disabled={isAnalyzing || !baseAudio}
                >
                  {isAnalyzing ? "Analyzing..." : "Analyze audio"}
                </button>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => setActivePage("preview")}
                >
                  Next: Preview
                </button>
              </div>
            </div>

            <div className="timeline-card">
              <div className="timeline-title">Timeline Visualization</div>
              <div className="timeline-track">
                {slots.map((slot) => (
                  <button
                    key={slot.id}
                    type="button"
                    className={`timeline-dot${
                      slot.id === selectedSlotId ? " active" : ""
                    }`}
                    style={{
                      left: audioDuration
                        ? `${(slot.time / audioDuration) * 100}%`
                        : "50%",
                    }}
                    onClick={() => setSelectedSlotId(slot.id)}
                    aria-label={`Select ${slot.id}`}
                  />
                ))}
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
                      slot.id === selectedSlotId ? " active" : ""
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
                          setSelectedSlotId(slot.id);
                          handleMerge("preview", slot.id);
                        }}
                      >
                        Preview
                      </button>
                      <button
                        type="button"
                        className={`primary select-slot${
                          slot.id === selectedSlotId ? " selected" : ""
                        }`}
                        onClick={() => setSelectedSlotId(slot.id)}
                      >
                        Select Slot
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
          </div>
        </section>
      )}

      {activePage === "preview" && (
        <section className="page">
          <div className="page-card">
            <div className="page-header">
              <div>
                <p className="eyebrow">Preview</p>
                <h2>Test the sponsor read before export.</h2>
                <p className="subtitle">
                  Select a script and listen to how it lands at the chosen slot.
                </p>
              </div>
              <button
                type="button"
                className="ghost"
                onClick={() => setActivePage("export")}
              >
                Next: Export
              </button>
            </div>

            <div className="preview-grid">
              <div className="script-panel">
                <div className="field">
                  <label htmlFor="voiceId">Voice ID</label>
                  <input
                    id="voiceId"
                    type="text"
                    placeholder="Paste your ElevenLabs voice ID"
                    value={voiceId}
                    onChange={(event) => setVoiceId(event.target.value)}
                  />
                </div>
                <div className="script-list">
                  {sampleScripts.map((script) => (
                    <button
                      key={script.title}
                      type="button"
                      className="script-card"
                      onClick={() => {
                        if (!selectedSponsor) return;
                        updateSponsor(selectedSponsor.id, { script: script.text });
                      }}
                    >
                      <div className="script-title">{script.title}</div>
                      <div className="script-text">{script.text}</div>
                    </button>
                  ))}
                </div>
                <div className="field">
                  <label htmlFor="ttsText">Brand script</label>
                  <textarea
                    id="ttsText"
                    rows={4}
                    value={selectedScript}
                    onChange={(event) => {
                      if (!selectedSponsor) return;
                      updateSponsor(selectedSponsor.id, {
                        script: event.target.value,
                      });
                    }}
                  />
                </div>
              </div>
              <div className="preview-panel">
                <div className="preview-summary">
                  <div className="summary-item">
                    <span>Selected slot</span>
                    <strong>
                      {selectedSlot
                        ? `${selectedSlot.time.toFixed(2)}s`
                        : "Not selected"}
                    </strong>
                  </div>
                  <div className="summary-item">
                    <span>Confidence</span>
                    <strong>
                      {selectedSlot ? `${selectedSlot.confidence}%` : "--"}
                    </strong>
                  </div>
                <div className="summary-item">
                  <span>Sponsor</span>
                  <strong>
                    {selectedSponsorName || "Add sponsor name"}
                  </strong>
                </div>
                </div>
                <button
                  type="button"
                  className="primary"
                  onClick={() =>
                    handleMerge("preview", selectedSlotId ?? undefined)
                  }
                  disabled={isPreviewing}
                >
                  {isPreviewing ? "Building preview..." : "Preview slot"}
                </button>
                <div className="audio-panel">
                  <audio ref={audioRef} controls src={audioUrl} />
                </div>
              </div>
            </div>
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
                  <span>Selected slot</span>
                  <strong>
                    {selectedSlot
                      ? `${selectedSlot.time.toFixed(2)}s`
                      : "Not selected"}
                  </strong>
                </div>
                <div className="summary-item">
                  <span>Confidence</span>
                  <strong>
                    {selectedSlot ? `${selectedSlot.confidence}%` : "--"}
                  </strong>
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
                  onClick={() =>
                    handleMerge("render", selectedSlotId ?? undefined)
                  }
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
