/**
 * The numbers an editor checks, formatted once.
 *
 * Timestamps, sizes and the file bar (`ryan-test1.mp3 · MP3 · 358 KB · 0:22`)
 * are all set in JetBrains Mono, so they have to be formatted the same way
 * everywhere they appear. No React here, so `format.test.ts` can run it.
 */

/**
 * `m:ss`, with minutes unbounded (`75:03`), matching the minutes-and-seconds
 * fields of "Add a break at mm:ss". Anything that is not a finite,
 * non-negative number reads as `0:00` rather than `NaN:NaN`.
 */
export const formatTimestamp = (seconds: number | null | undefined): string => {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "0:00";
  const total = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
};

/** 1024-based, as the browser and the OS file picker report sizes. */
export const formatBytes = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes < 0) return "0 B";
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};

/**
 * `MP3`, `WAV`, `M4A`... from the extension, then the MIME subtype, so a file
 * with no extension still says what it is.
 */
export const fileFormatLabel = (name: string, mimeType = ""): string => {
  const dot = name.lastIndexOf(".");
  const extension = dot > 0 && dot < name.length - 1 ? name.slice(dot + 1) : "";
  if (extension && /^[a-z0-9]{1,5}$/i.test(extension)) return extension.toUpperCase();
  const subtype = mimeType.split("/")[1]?.split(";")[0]?.trim() ?? "";
  if (subtype === "mpeg") return "MP3";
  if (subtype && /^[a-z0-9.+-]{1,12}$/i.test(subtype)) {
    return subtype.replace(/^x-/, "").toUpperCase();
  }
  return "AUDIO";
};

/** Shown while the browser is still decoding, or when it could not. */
export const UNKNOWN_DURATION = "-:--";

export interface FileBarText {
  name: string;
  format: string;
  size: string;
  duration: string;
}

/** The parts of the header's `file-bar`, in order. */
export const describeFile = (
  file: { name: string; size: number; type?: string },
  durationSeconds: number | null,
): FileBarText => ({
  name: file.name || "Untitled audio",
  format: fileFormatLabel(file.name, file.type ?? ""),
  size: formatBytes(file.size),
  duration:
    typeof durationSeconds === "number" && Number.isFinite(durationSeconds) && durationSeconds > 0
      ? formatTimestamp(durationSeconds)
      : UNKNOWN_DURATION,
});

/** `episode-with-ads.mp3` for `episode.wav`: the merge always returns MP3. */
export const downloadName = (fileName: string): string => {
  const dot = fileName.lastIndexOf(".");
  const stem = (dot > 0 ? fileName.slice(0, dot) : fileName).trim() || "slotify";
  return `${stem}-with-ads.mp3`;
};
