import { ffmpegTimeoutMs } from "../config";
import { runProcess, SubprocessError } from "../lib/subprocess";

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

// Episodes open with music and intros; a voice clone wants the host talking.
const CLONE_INTRO_SECONDS = 30;
// Skip the intro only when at least this much audio follows it.
const CLONE_MIN_AFTER_INTRO_SECONDS = 60;

/** Where a clone excerpt starts and how long it may run. */
export const cloneSampleWindow = (
  durationSeconds: number | null,
  capSeconds: number,
  { skipIntro = true }: { skipIntro?: boolean } = {},
): { startSeconds: number; seconds: number } => {
  const longEnough =
    durationSeconds !== null &&
    durationSeconds >= CLONE_INTRO_SECONDS + CLONE_MIN_AFTER_INTRO_SECONDS;
  return { startSeconds: skipIntro && longEnough ? CLONE_INTRO_SECONDS : 0, seconds: capSeconds };
};

/**
 * Cut the excerpt of `input` that is sent to ElevenLabs: at most
 * `capSeconds`, past the intro when the file is long enough, as mono
 * 44.1 kHz MP3. Uploading a whole hour-long episode cost bandwidth and upload
 * time for no better a voice.
 */
export const trimCloneSample = async (
  input: string,
  output: string,
  {
    durationSeconds,
    capSeconds,
    skipIntro = true,
    signal,
  }: { durationSeconds: number | null; capSeconds: number; skipIntro?: boolean; signal?: AbortSignal },
): Promise<void> => {
  const { startSeconds, seconds } = cloneSampleWindow(durationSeconds, capSeconds, { skipIntro });
  await runFfmpeg(
    [
      "-y",
      "-v",
      "error",
      // Input options: seek and read only the excerpt, not the whole file.
      "-ss",
      String(startSeconds),
      "-t",
      String(seconds),
      "-i",
      input,
      "-vn",
      "-ac",
      "1",
      "-ar",
      "44100",
      "-c:a",
      "libmp3lame",
      "-b:a",
      "192k",
      output,
    ],
    { signal },
  );
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

/**
 * Demuxers an upload may be opened with (ffprobe `format_name` components).
 * ffmpeg picks a demuxer by content, so without a list an upload named
 * `x.mp3` could be opened as an HLS playlist (`hls`) whose entries are read
 * from the server's own filesystem, or as any of ffmpeg's other formats.
 */
export const ALLOWED_AUDIO_FORMATS = [
  "mp3",
  "wav",
  "flac",
  "ogg",
  "mov",
  "mp4",
  "m4a",
  "3gp",
  "aac",
  "matroska",
  "webm",
] as const;

// The file extension each format is stored under, so tools that look at the
// name (pydub, Whisper) see the truth rather than the client's filename.
const EXTENSION_FOR_FORMAT: Record<(typeof ALLOWED_AUDIO_FORMATS)[number], string> = {
  mp3: "mp3",
  wav: "wav",
  flac: "flac",
  ogg: "ogg",
  mov: "m4a",
  mp4: "m4a",
  m4a: "m4a",
  "3gp": "m4a",
  aac: "aac",
  matroska: "webm",
  webm: "webm",
};

export interface AudioProbe {
  /** ffprobe's format_name, e.g. "mp3" or "mov,mp4,m4a,3gp,3g2,mj2". */
  formatName: string;
  /** Extension (without the dot) matching the detected container. */
  extension: string;
  /** Null when the container does not say (e.g. a MediaRecorder webm). */
  durationSeconds: number | null;
}

/**
 * Why an upload is not usable audio. "unsupported": not a container on the
 * allow-list, or not media at all; "no-audio": a valid container with no audio
 * stream in it.
 */
export class UnreadableAudioError extends Error {
  constructor(
    readonly kind: "unsupported" | "no-audio",
    message: string,
    readonly detail = "",
  ) {
    super(message);
    this.name = "UnreadableAudioError";
  }
}

/**
 * Identify an uploaded file by its content. Only the local file protocol and
 * the allow-listed demuxers are enabled, so probing cannot itself open a
 * playlist or a URL.
 */
export const probeAudio = async (
  filePath: string,
  { signal, timeoutMs = ffmpegTimeoutMs }: FfmpegOptions = {},
): Promise<AudioProbe> => {
  let stdout: string;
  try {
    ({ stdout } = await runProcess(
      "ffprobe",
      [
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        ALLOWED_AUDIO_FORMATS.join(","),
        "-show_entries",
        "format=format_name,duration:stream=codec_type",
        "-of",
        "json",
        filePath,
      ],
      { label: "ffprobe", timeoutMs, timeoutEnvVar: "SLOTIFY_FFMPEG_TIMEOUT_MS", signal },
    ));
  } catch (error) {
    // ffprobe ran and could not make sense of the file: the upload's fault.
    // A timeout, an abort or a missing ffprobe is ours, and stays an error.
    if (error instanceof SubprocessError && error.reason === "exit") {
      throw new UnreadableAudioError("unsupported", "not a supported audio file", error.stderr);
    }
    throw error;
  }

  let parsed: any;
  try {
    parsed = JSON.parse(stdout);
  } catch {
    throw new UnreadableAudioError("unsupported", "not a supported audio file", stdout.slice(0, 500));
  }
  const formatName: string =
    typeof parsed?.format?.format_name === "string" ? parsed.format.format_name : "";
  // Checked again here, not left to -format_whitelist alone.
  const known = formatName
    .split(",")
    .find((name): name is keyof typeof EXTENSION_FOR_FORMAT =>
      (ALLOWED_AUDIO_FORMATS as readonly string[]).includes(name),
    );
  if (!known) {
    throw new UnreadableAudioError("unsupported", `unsupported container (${formatName || "unknown"})`);
  }
  const streams: any[] = Array.isArray(parsed?.streams) ? parsed.streams : [];
  if (!streams.some((stream) => stream?.codec_type === "audio")) {
    throw new UnreadableAudioError("no-audio", "the file has no audio stream");
  }
  const duration = Number.parseFloat(parsed?.format?.duration);
  return {
    formatName,
    extension: EXTENSION_FOR_FORMAT[known],
    durationSeconds: Number.isFinite(duration) && duration > 0 ? duration : null,
  };
};
