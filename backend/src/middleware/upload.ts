import path from "node:path";
import type { RequestHandler } from "express";
import multer from "multer";
import { maxUploadBytes } from "../config";
import { HttpError } from "../lib/http";

/** Voice-clone samples per request; ElevenLabs IVC takes a handful anyway. */
export const MAX_CLONE_SAMPLES = 10;

// Browsers and Python's multipart helper often send audio as
// application/octet-stream (a Blob with no type), so the extension decides.
const AUDIO_EXTENSIONS = new Set([
  ".mp3", ".mpga", ".mpeg", ".wav", ".wave", ".m4a", ".mp4", ".aac", ".ogg",
  ".oga", ".opus", ".flac", ".webm", ".aif", ".aiff", ".caf", ".wma",
]);
// Containers ffmpeg decodes that recorders label as video.
const AUDIO_CONTAINER_TYPES = new Set(["video/mp4", "video/webm", "video/ogg", "video/quicktime"]);

const isAudioUpload = (file: Express.Multer.File): boolean => {
  const type = (file.mimetype || "").toLowerCase();
  if (type.startsWith("audio/") || AUDIO_CONTAINER_TYPES.has(type)) return true;
  const extension = path.extname(file.originalname || "").toLowerCase();
  return (type === "" || type === "application/octet-stream") && AUDIO_EXTENSIONS.has(extension);
};

// Files are held in memory; routes persist them to temp dirs as needed. Every
// limit is explicit because memory storage with none let one request buffer
// as much as a client cared to send.
const multerUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: maxUploadBytes,
    files: MAX_CLONE_SAMPLES,
    fields: 30,
    fieldSize: 1024 * 1024,
    parts: MAX_CLONE_SAMPLES + 30,
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
const withClientErrors =
  (middleware: RequestHandler): RequestHandler =>
  (req, res, next) => {
    middleware(req, res, (error?: unknown) => {
      if (!error || error instanceof HttpError || error instanceof multer.MulterError) {
        next(error);
        return;
      }
      next(new HttpError(400, "Malformed multipart upload."));
    });
  };

export const upload = {
  single: (field: string) => withClientErrors(multerUpload.single(field)),
  array: (field: string, maxCount: number) =>
    withClientErrors(multerUpload.array(field, maxCount)),
  fields: (fields: Array<{ name: string; maxCount: number }>) =>
    withClientErrors(multerUpload.fields(fields)),
};
