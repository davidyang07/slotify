import { useRef } from "react";
import { AUDIO_ACCEPT } from "../lib/audio-formats";
import { describeFile } from "../lib/format";
import SlotifyLogo from "./SlotifyLogo";
import SoundwaveIcon from "./SoundwaveIcon";
import { GitHubIcon } from "./icons";
import "./AppHeader.css";

type FileBarProps = {
  file: File;
  durationSeconds: number | null;
  /** A new file starts over: the current file's work is abandoned. */
  onReplace: (file: File) => void;
};

/**
 * `name · format · size · duration` in mono, then Replace. It replaces the old
 * dropzone and the Format / Size / Duration / Status card.
 */
const FileBar = ({ file, durationSeconds, onReplace }: FileBarProps) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const info = describeFile(file, durationSeconds);
  return (
    <div className="file-bar">
      <p className="file-bar-text" title={`${info.name} · ${info.format} · ${info.size} · ${info.duration}`}>
        <span className="file-bar-name">{info.name}</span>
        <span className="file-bar-meta">
          <span aria-hidden="true"> · </span>
          {info.format}
          <span aria-hidden="true"> · </span>
          {info.size}
          <span aria-hidden="true"> · </span>
          <span aria-label={`duration ${info.duration}`}>{info.duration}</span>
        </span>
      </p>
      <input
        ref={inputRef}
        type="file"
        accept={AUDIO_ACCEPT}
        className="file-bar-input"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(event) => {
          const next = event.target.files?.[0];
          event.target.value = "";
          if (next) onReplace(next);
        }}
      />
      <button
        type="button"
        className="secondary"
        aria-label={`Replace ${info.name}`}
        onClick={() => inputRef.current?.click()}
      >
        Replace
      </button>
    </div>
  );
};

type AppHeaderProps = {
  file: File | null;
  durationSeconds: number | null;
  onReplace: (file: File) => void;
};

/** 56px bar on canvas: wordmark left; the file bar and GitHub link right. */
const AppHeader = ({ file, durationSeconds, onReplace }: AppHeaderProps) => (
  <header className={`app-header${file ? " has-file" : ""}`}>
    <a href="/" className="brand" aria-label="Slotify home">
      <SoundwaveIcon size="md" variant="light" />
      <SlotifyLogo size="md" variant="light" />
    </a>
    {file && (
      <FileBar file={file} durationSeconds={durationSeconds} onReplace={onReplace} />
    )}
    <a
      href="https://github.com/jweng121/uoft-winners"
      target="_blank"
      rel="noopener noreferrer"
      className="github-link"
      aria-label="Slotify on GitHub (opens in a new tab)"
    >
      <GitHubIcon size={20} />
    </a>
  </header>
);

export default AppHeader;
