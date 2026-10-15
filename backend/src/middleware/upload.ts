import fs from "node:fs";
import path from "node:path";
import type { Request, RequestHandler } from "express";
import multer from "multer";
import {
  maxAudioSeconds,
  maxCloneSampleBytes,
  maxCloneSamples,
  maxUploadBytes,
} from "../config";
import { HttpError } from "../lib/http";
import { makeTempDir, removeTempDir } from "../lib/temp";
import { probeAudio, UnreadableAudioError, type AudioProbe } from "../services/ffmpeg";

/** Voice-clone samples per request (SLOTIFY_MAX_CLONE_SAMPLES). */
export const MAX_CLONE_SAMPLES = maxCloneSamples;

// Browsers and Python's multipart helper often send audio as
// application/octet-stream (a Blob with no type), so the extension decides.
// This is only a cheap first filter: `inspectUploads` decides by content. One
// extension per container on ALLOWED_AUDIO_FORMATS; the UI's file picker
// offers exactly these (frontend/src/lib/audio-formats.ts, whose test checks).
const AUDIO_EXTENSIONS = new Set([
  ".mp3", ".mpga", ".wav", ".wave", ".flac", ".ogg", ".oga", ".opus", ".m4a",
  ".mp4", ".mov", ".3gp", ".aac", ".webm", ".mka",
]);
// Containers ffmpeg decodes that recorders label as video.
const AUDIO_CONTAINER_TYPES = new Set([
  "video/mp4", "video/webm", "video/ogg", "video/quicktime", "video/3gpp",
]);
// Playlists are "audio/*" by MIME type and are never audio.
const PLAYLIST_TYPES = new Set(["audio/x-mpegurl", "audio/mpegurl", "audio/x-scpls"]);
const PLAYLIST_EXTENSIONS = new Set([".m3u", ".m3u8", ".pls"]);

const isAudioUpload = (file: Express.Multer.File): boolean => {
  const type = (file.mimetype || "").toLowerCase();
  const extension = path.extname(file.originalname || "").toLowerCase();
  if (PLAYLIST_TYPES.has(type) || PLAYLIST_EXTENSIONS.has(extension)) return false;
  if (type.startsWith("audio/") || AUDIO_CONTAINER_TYPES.has(type)) return true;
  return (type === "" || type === "application/octet-stream") && AUDIO_EXTENSIONS.has(extension);
};

// ---------------------------------------------------------------------------
// Per-request upload directory
//
// Uploads used to be buffered in memory -- /api/clone alone accepted 10 files
// of 200 MB each -- and then copied to disk under the client's filename. They
// now stream to disk, into a temp dir that belongs to the request, under fixed
// names. The route that handles the request takes the dir over as its work dir
// (`claimWorkDir`) and removes it in its `finally`; a request that ends before
// any route claims it (a 400, a multer limit) has it removed when the response
// closes.

interface UploadState {
  dirPrefix: string;
  dir: Promise<string> | null;
  claimed: boolean;
  count: number;
}

const STATE = Symbol("slotifyUpload");
const stateOf = (req: Request): UploadState | undefined =>
  (req as unknown as Record<symbol, UploadState | undefined>)[STATE];

const ensureDir = (state: UploadState): Promise<string> =>
  (state.dir ??= makeTempDir(state.dirPrefix));

const storage = multer.diskStorage({
  destination: (req, _file, callback) => {
    const state = stateOf(req);
    if (!state) {
      callback(new Error("Upload state missing."), "");
      return;
    }
    ensureDir(state).then(
      (dir) => callback(null, dir),
      (error: Error) => callback(error, ""),
    );
  },
  // Never the client's name: it reached ffmpeg's argv, could be too long for
  // the filesystem, and an extension such as .m3u8 changes how ffmpeg reads it.
  filename: (req, _file, callback) => {
    const state = stateOf(req);
    callback(null, `upload-${state ? state.count++ : 0}`);
  },
});

/**
 * The request's work dir: the dir its uploads were written to, or a new one
 * when it has none. The caller owns it from here and must `removeTempDir` it.
 */
export const claimWorkDir = async (req: Request, prefix: string): Promise<string> => {
  const state = stateOf(req);
  if (state?.dir) {
    state.claimed = true;
    return state.dir;
  }
  return makeTempDir(prefix);
};

export interface UploadOptions {
  /** Temp dir prefix, e.g. "insert-sections-". */
  dirPrefix?: string;
  maxFileBytes?: number;
  /** The env var that sets `maxFileBytes`, named in the 413. */
  sizeEnvVar?: string;
}

const multerFor = (maxFiles: number, maxFileBytes: number) =>
  multer({
    storage,
    limits: {
      fileSize: maxFileBytes,
      files: maxFiles,
      fields: 30,
      fieldSize: 1024 * 1024,
      parts: maxFiles + 30,
    },
    fileFilter: (_req, file, callback) => {
      if (isAudioUpload(file)) {
        callback(null, true);
        return;
      }
      callback(
        new HttpError(
          415,
          `Unsupported upload type for "${file.fieldname}"; upload an audio file (mp3, wav, m4a, ...).`,
        ),
      );
    },
  });

/**
 * multer's parse errors (truncated or malformed multipart) are busboy errors
 * with no status; they are the client's fault, so they become a 400 here and
 * the JSON error middleware maps MulterError limits to 413/400.
 */
const withUploadDir = (
  middleware: RequestHandler,
  { dirPrefix = "slotify-upload-", maxFileBytes = maxUploadBytes, sizeEnvVar = "SLOTIFY_MAX_UPLOAD_MB" }: UploadOptions,
): RequestHandler =>
  (req, res, next) => {
    const state: UploadState = { dirPrefix, dir: null, claimed: false, count: 0 };
    (req as unknown as Record<symbol, UploadState>)[STATE] = state;
    res.once("close", () => {
      if (!state.claimed) void state.dir?.then(removeTempDir, () => undefined);
    });
    middleware(req, res, (error?: unknown) => {
      if (!error || error instanceof HttpError) {
        next(error);
        return;
      }
      if (error instanceof multer.MulterError) {
        if (error.code === "LIMIT_FILE_SIZE") {
          const megabytes = Math.round((maxFileBytes / (1024 * 1024)) * 1000) / 1000;
          next(
            new HttpError(
              413,
              `Uploaded file "${error.field ?? "file"}" is too large: the limit is ${megabytes} MB (see ${sizeEnvVar}).`,
            ),
          );
          return;
        }
        next(error);
        return;
      }
      next(new HttpError(400, "Malformed multipart upload."));
    });
  };

export const upload = {
  single: (field: string, options: UploadOptions = {}) =>
    withUploadDir(multerFor(1, options.maxFileBytes ?? maxUploadBytes).single(field), options),
  array: (field: string, maxCount: number, options: UploadOptions = {}) =>
    withUploadDir(
      multerFor(maxCount, options.maxFileBytes ?? maxUploadBytes).array(field, maxCount),
      options,
    ),
  fields: (fields: Array<{ name: string; maxCount: number }>, options: UploadOptions = {}) =>
    withUploadDir(
      multerFor(
        fields.reduce((total, entry) => total + entry.maxCount, 0),
        options.maxFileBytes ?? maxUploadBytes,
      ).fields(fields),
      options,
    ),
};

/** The voice-clone sample limits (SLOTIFY_MAX_CLONE_SAMPLES / _SAMPLE_MB). */
export const cloneUploadOptions: UploadOptions = {
  dirPrefix: "voice-clone-",
  maxFileBytes: maxCloneSampleBytes,
  sizeEnvVar: "SLOTIFY_MAX_CLONE_SAMPLE_MB",
};

// ---------------------------------------------------------------------------
// Content checks

export type InspectedUpload = Express.Multer.File & { probe: AudioProbe };

const uploadedFiles = (req: Request): Express.Multer.File[] => {
  if (req.file) return [req.file];
  if (Array.isArray(req.files)) return req.files;
  if (req.files && typeof req.files === "object") return Object.values(req.files).flat();
  return [];
};

const minutes = (seconds: number) => Math.round((seconds / 60) * 10) / 10;

/**
 * Identify every upload by its content with ffprobe, before anything else
 * reads it: a container outside the audio allow-list (an HLS playlist named
 * .mp3, a text file) or a file that is not media is a 415, a container with no
 * audio stream a 422, and audio longer than SLOTIFY_MAX_AUDIO_MINUTES a 413.
 * Each file is then renamed to a fixed name with the extension of what it
 * actually is. Call after the request's own fields are validated, so a
 * malformed request costs no subprocess.
 */
export const inspectUploads = async (
  req: Request,
  { signal, maxSeconds = maxAudioSeconds }: { signal?: AbortSignal; maxSeconds?: number } = {},
): Promise<InspectedUpload[]> => {
  const inspected: InspectedUpload[] = [];
  for (const [index, file] of uploadedFiles(req).entries()) {
    let probe: AudioProbe;
    try {
      probe = await probeAudio(file.path, { signal });
    } catch (error) {
      if (!(error instanceof UnreadableAudioError)) throw error;
      console.warn(`Rejected upload "${file.fieldname}": ${error.message}.`, error.detail.slice(-500));
      if (error.kind === "no-audio") {
        throw new HttpError(422, `The upload "${file.fieldname}" contains no audio stream.`);
      }
      throw new HttpError(
        415,
        `The upload "${file.fieldname}" is not a supported audio file; ` +
          "upload mp3, wav, flac, ogg, m4a/mp4, aac or webm.",
      );
    }
    if (probe.durationSeconds !== null && probe.durationSeconds > maxSeconds) {
      throw new HttpError(
        413,
        `The upload "${file.fieldname}" is ${minutes(probe.durationSeconds)} minutes long; ` +
          `the limit is ${minutes(maxSeconds)} minutes (see SLOTIFY_MAX_AUDIO_MINUTES).`,
      );
    }
    const renamed = path.join(path.dirname(file.path), `audio-${index}.${probe.extension}`);
    await fs.promises.rename(file.path, renamed);
    file.path = renamed;
    file.filename = path.basename(renamed);
    inspected.push(Object.assign(file, { probe }));
  }
  return inspected;
};
