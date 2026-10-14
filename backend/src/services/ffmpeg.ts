import { ffmpegTimeoutMs } from "../config";
import { runProcess } from "../lib/subprocess";

export interface FfmpegOptions {
  /** Kills ffmpeg when aborted (the client went away). */
  signal?: AbortSignal;
  timeoutMs?: number;
}

/** Run ffmpeg with the given args, resolving on a clean exit. */
export const runFfmpeg = async (
  args: string[],
  { signal, timeoutMs = ffmpegTimeoutMs }: FfmpegOptions = {},
): Promise<void> => {
  await runProcess("ffmpeg", args, {
    label: "ffmpeg",
    timeoutMs,
    timeoutEnvVar: "SLOTIFY_FFMPEG_TIMEOUT_MS",
    signal,
  });
};

/** Probe a media file's duration in seconds, or null if unavailable. */
export const getAudioDuration = async (
  filePath: string,
  { signal, timeoutMs = ffmpegTimeoutMs }: FfmpegOptions = {},
): Promise<number | null> => {
  const { stdout } = await runProcess(
    "ffprobe",
    [
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      filePath,
    ],
    { label: "ffprobe", timeoutMs, timeoutEnvVar: "SLOTIFY_FFMPEG_TIMEOUT_MS", signal },
  );
  const duration = Number.parseFloat(stdout.trim());
  return Number.isFinite(duration) && duration > 0 ? duration : null;
};
