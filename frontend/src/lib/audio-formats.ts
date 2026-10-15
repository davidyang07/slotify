/**
 * The audio formats the API accepts, so the file picker offers only those.
 *
 * The picker used to accept `audio/*`, which offered AIFF, CAF and WMA files
 * the API then refused with a 415. This mirrors the backend's allow-list (the
 * extension filter in backend/src/middleware/upload.ts over the containers in
 * backend/src/services/ffmpeg.ts); audio-formats.test.ts fails if they drift.
 */

export const SUPPORTED_AUDIO_FORMATS: ReadonlyArray<{ label: string; extensions: readonly string[] }> = [
  { label: "MP3", extensions: [".mp3", ".mpga"] },
  { label: "WAV", extensions: [".wav", ".wave"] },
  { label: "FLAC", extensions: [".flac"] },
  { label: "OGG/Opus", extensions: [".ogg", ".oga", ".opus"] },
  { label: "M4A/MP4", extensions: [".m4a", ".mp4", ".mov", ".3gp"] },
  { label: "AAC", extensions: [".aac"] },
  { label: "WebM", extensions: [".webm", ".mka"] },
];

export const SUPPORTED_AUDIO_EXTENSIONS: readonly string[] = SUPPORTED_AUDIO_FORMATS.flatMap(
  (format) => format.extensions,
);

/** The file inputs' `accept`: extensions only, since `audio/*` is wider. */
export const AUDIO_ACCEPT = SUPPORTED_AUDIO_EXTENSIONS.join(",");

/** "MP3, WAV, FLAC, OGG/Opus, M4A/MP4, AAC or WebM" */
export const SUPPORTED_AUDIO_LABEL = (() => {
  const labels = SUPPORTED_AUDIO_FORMATS.map((format) => format.label);
  return `${labels.slice(0, -1).join(", ")} or ${labels[labels.length - 1]}`;
})();

/**
 * False for a file whose extension the API refuses. A drop bypasses the
 * picker's filter, so the dropzone checks too. A name with no extension is
 * left to the server, which identifies uploads by content.
 */
export const isSupportedAudioFile = (file: { name: string }): boolean => {
  const dot = file.name.lastIndexOf(".");
  if (dot <= 0) return true;
  return SUPPORTED_AUDIO_EXTENSIONS.includes(file.name.slice(dot).toLowerCase());
};

export const unsupportedAudioMessage = (name: string): string =>
  `"${name}" is not a supported audio format. Use ${SUPPORTED_AUDIO_LABEL}.`;
