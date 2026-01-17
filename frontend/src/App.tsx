import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const defaultText =
  "Say something short here to verify your cloned voice sounds right.";

function App() {
  const apiBase =
    import.meta.env.VITE_API_BASE_URL ?? "http://localhost:3001";
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const [voiceName, setVoiceName] = useState("My Voice Clone");
  const [voiceId, setVoiceId] = useState("");
  const [text, setText] = useState(defaultText);
  const [baseAudio, setBaseAudio] = useState<File | null>(null);
  const [insertAt, setInsertAt] = useState("0");
  const [crossfade, setCrossfade] = useState("0.15");
  const [pause, setPause] = useState("0.12");
  const [cloneMessage, setCloneMessage] = useState("");
  const [cloneOk, setCloneOk] = useState<boolean | null>(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [isCloning, setIsCloning] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const fileCount = useMemo(() => files?.length ?? 0, [files]);

  useEffect(() => {
    if (!audioUrl) return;
    audioRef.current?.play().catch(() => undefined);
    return () => {
      URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const handleClone = async () => {
    setError("");
    setStatus("");
    setCloneMessage("");
    setCloneOk(null);

    if (!files || files.length === 0) {
      setError("Select at least one audio file.");
      return;
    }

    setIsCloning(true);
    try {
      const form = new FormData();
      form.append("name", voiceName);
      Array.from(files).forEach((file) => form.append("files", file));

      const response = await fetch(`${apiBase}/api/clone`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Clone request failed.");
      }

      const data = (await response.json()) as { voiceId?: string };
      if (!data.voiceId) {
        throw new Error("No voiceId returned.");
      }

      setVoiceId(data.voiceId);
      setStatus("Clone ready. Voice ID saved.");
      setCloneMessage("Voice cloned successfully.");
      setCloneOk(true);
    } catch (cloneError) {
      const message =
        cloneError instanceof Error ? cloneError.message : "Clone failed.";
      setError(message);
      setCloneMessage(message);
      setCloneOk(false);
    } finally {
      setIsCloning(false);
    }
  };

  const handleSpeak = async () => {
    setError("");
    setStatus("");

    if (!voiceId) {
      setError("Clone a voice before requesting speech.");
      return;
    }

    if (!baseAudio) {
      setError("Upload a base audio file for merging.");
      return;
    }

    const insertAtSeconds = Number.parseFloat(insertAt);
    const crossfadeSeconds = Number.parseFloat(crossfade);
    if (!Number.isFinite(insertAtSeconds) || insertAtSeconds < 0) {
      setError("Insert time must be a positive number.");
      return;
    }
    if (!Number.isFinite(crossfadeSeconds) || crossfadeSeconds < 0) {
      setError("Crossfade must be a positive number.");
      return;
    }

    setIsSpeaking(true);
    try {
      const response = await fetch(`${apiBase}/api/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          voiceId,
          text,
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
    mergeForm.append("insertAt", insertAt);
    mergeForm.append("crossfade", crossfade);
    mergeForm.append("pause", pause);

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
      setStatus("Speech generated and merged.");
    } catch (ttsError) {
      setError(
        ttsError instanceof Error ? ttsError.message : "TTS failed.",
      );
    } finally {
      setIsSpeaking(false);
    }
  };

  return (
    <div className="app">
      <header className="hero">
        <p className="eyebrow">Voice Lab</p>
        <h1>Clone, then speak.</h1>
        <p className="subtitle">
          Upload sample audio, create a custom voice, then test it instantly.
        </p>
      </header>

      <section className="panel">
        <h2>1. Upload audio</h2>
        <div className="field">
          <label htmlFor="voiceName">Voice name</label>
          <input
            id="voiceName"
            type="text"
            value={voiceName}
            onChange={(event) => setVoiceName(event.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="voiceFiles">Audio files</label>
          <input
            id="voiceFiles"
            type="file"
            accept="audio/*"
            multiple
            onChange={(event) => setFiles(event.target.files)}
          />
          <span className="helper">
            {fileCount === 0
              ? "Add at least one clip."
              : `${fileCount} file${fileCount === 1 ? "" : "s"} selected.`}
          </span>
        </div>
        <button
          type="button"
          className="primary"
          onClick={handleClone}
          disabled={isCloning}
        >
          {isCloning ? "Cloning..." : "Create voice"}
        </button>
        <p className="helper">
          The generated voice is kept in memory for this session.
        </p>
        {cloneOk !== null && (
          <div className={`result-card ${cloneOk ? "ok" : "error"}`}>
            {cloneMessage}
          </div>
        )}
      </section>

      <section className="panel">
        <h2>2. Generate speech</h2>
        <div className="field">
          <label htmlFor="baseAudio">Base audio</label>
          <input
            id="baseAudio"
            type="file"
            accept="audio/*"
            onChange={(event) => setBaseAudio(event.target.files?.[0] ?? null)}
          />
          <span className="helper">
            The MP3 that will receive the generated speech.
          </span>
        </div>
        <div className="field">
          <label htmlFor="insertAt">Insert time (seconds)</label>
          <input
            id="insertAt"
            type="number"
            min="0"
            step="0.01"
            value={insertAt}
            onChange={(event) => setInsertAt(event.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="crossfade">Crossfade (seconds)</label>
          <input
            id="crossfade"
            type="number"
            min="0"
            step="0.01"
            value={crossfade}
            onChange={(event) => setCrossfade(event.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="pause">Pause between segments (seconds)</label>
          <input
            id="pause"
            type="number"
            min="0"
            step="0.01"
            value={pause}
            onChange={(event) => setPause(event.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="ttsText">Text prompt</label>
          <textarea
            id="ttsText"
            rows={4}
            value={text}
            onChange={(event) => setText(event.target.value)}
          />
        </div>
        <button
          type="button"
          className="primary"
          onClick={handleSpeak}
          disabled={isSpeaking}
        >
          {isSpeaking ? "Generating..." : "Generate speech"}
        </button>
        <div className="audio-panel">
          <audio ref={audioRef} controls src={audioUrl} />
        </div>
      </section>

      {(status || error) && (
        <section className="status">
          {status && <p className="status-ok">{status}</p>}
          {error && <p className="status-error">{error}</p>}
        </section>
      )}
    </div>
  );
}

export default App;
