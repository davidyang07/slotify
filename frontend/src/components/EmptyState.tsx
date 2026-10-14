import type { DragEvent } from "react";
import { useState } from "react";
import SoundwaveBackground from "./SoundwaveBackground";
import { UploadIcon } from "./icons";
import "./EmptyState.css";

type EmptyStateProps = {
  /** Starts analysis at once: there is no Analyze button. */
  onFile: (file: File) => void;
  /** Problems the user should know before dropping a file, in words. */
  problems: string[];
};

/**
 * No file yet: one heading, one line, and the dropzone, over the dimmed
 * grayscale soundwave video. The dropzone is the action, so there is no amber
 * button here. Dropping anywhere on the screen works, so a near miss does not
 * make the browser navigate away to the file.
 */
const EmptyState = ({ onFile, problems }: EmptyStateProps) => {
  const [dragActive, setDragActive] = useState(false);

  const take = (files: FileList | null | undefined) => {
    const file = files?.[0];
    if (file) onFile(file);
  };

  const onDragOver = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    if (!dragActive) setDragActive(true);
  };
  const onDragLeave = (event: DragEvent<HTMLElement>) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
      setDragActive(false);
    }
  };
  const onDrop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    setDragActive(false);
    take(event.dataTransfer?.files);
  };

  return (
    <main
      className="empty-screen"
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <SoundwaveBackground isActive={dragActive} />
      <div className="empty-state">
        <h1 className="empty-state-title">Seamless sponsor inserts for audio.</h1>
        <p className="empty-state-lead">
          Drop an episode: Slotify ranks its natural ad breaks, then voices
          your sponsor read in the host&rsquo;s own voice.
        </p>
        <label className={`dropzone${dragActive ? " dropzone-active" : ""}`}>
          {/* Visually hidden but focusable, so Tab reaches the dropzone and
              Enter or Space opens the file picker; the ring shows on the label. */}
          <input
            type="file"
            accept="audio/*"
            className="dropzone-input"
            onChange={(event) => {
              take(event.target.files);
              event.target.value = "";
            }}
          />
          <UploadIcon size={24} className="dropzone-icon" />
          <span className="dropzone-title">
            Drop an audio file here, or <span className="dropzone-link">choose one</span>
          </span>
          <span className="dropzone-caption">
            MP3, WAV, M4A, FLAC or OGG · up to 200 MB · analysis needs no API keys
          </span>
        </label>
        {problems.map((problem) => (
          <p key={problem} className="status-error empty-state-problem" role="alert">
            {problem}
          </p>
        ))}
      </div>
    </main>
  );
};

export default EmptyState;
