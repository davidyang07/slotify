import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { Router } from "express";
import { cloneSampleSeconds } from "../config";
import { isInternalRequest } from "../middleware/auth";
import {
  claimWorkDir,
  cloneUploadOptions,
  inspectUploads,
  MAX_CLONE_SAMPLES,
  upload,
} from "../middleware/upload";
import { getElevenLabs, isElevenLabsConfigured, voiceRegistry } from "../services/elevenlabs";
import { trimCloneSample } from "../services/ffmpeg";
import { sendFailure } from "../lib/errors";
import {
  asyncHandler,
  clientAbortSignal,
  HttpError,
  sendJsonConfirmed,
  throwIfClientGone,
} from "../lib/http";
import { optionalString } from "../lib/request-fields";
import { removeTempDir } from "../lib/temp";

export const cloneRouter = Router();

const sha256File = (filePath: string): Promise<string> =>
  new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    fs.createReadStream(filePath)
      .on("error", reject)
      .on("data", (chunk) => hash.update(chunk))
      .on("end", () => resolve(hash.digest("hex")));
  });

/** One digest for the sample set: the same upload maps to the same voice. */
const samplesHash = async (paths: string[]): Promise<string> => {
  const outer = createHash("sha256");
  for (const filePath of paths) outer.update(await sha256File(filePath));
  return outer.digest("hex");
};

// ElevenLabs voice ids are short alphanumeric tokens.
const VOICE_ID_PATTERN = /^[A-Za-z0-9_-]{1,100}$/;

cloneRouter.post(
  "/api/clone",
  upload.array("files", MAX_CLONE_SAMPLES, cloneUploadOptions),
  asyncHandler(async (req, res) => {
    const files = (req.files as Express.Multer.File[]) ?? [];
    // Only a label in the ElevenLabs dashboard, and the UI derives it from the
    // filename: a long one is shortened rather than refused.
    const name = (optionalString(req.body, "name", 1_000) ?? "").slice(0, 100).trim() || "My Voice Clone";
    if (!Array.isArray(files) || files.length === 0) {
      res.status(400).json({ error: "No audio files uploaded." });
      return;
    }

    if (!isElevenLabsConfigured()) {
      res.status(500).json({
        error:
          "ELEVENLABS_API_KEY not configured. Please set it in your environment variables.",
      });
      return;
    }

    const signal = clientAbortSignal(req, res);
    const internal = isInternalRequest(req);
    let tempDir: string | null = null;
    let voiceId: string | null = null;
    try {
      const workDir = await claimWorkDir(req, "voice-clone-");
      tempDir = workDir;
      const samples = await inspectUploads(req, { signal });
      // Hashed as uploaded, not as trimmed: the same episode maps to the same
      // voice, and a different one never does just because its excerpt matches.
      // A pipeline's own clone (insert_ad's per-speaker samples) is deleted by
      // that pipeline, so it is never handed to, or reused for, anyone else.
      const audioHash = internal ? null : await samplesHash(samples.map((sample) => sample.path));
      throwIfClientGone(signal, "Voice cloning");

      const acquired = await voiceRegistry.acquire(audioHash, async () => {
        // Only what IVC needs is sent. A pipeline's sample is already one
        // speaker's speech, so it has no intro to skip.
        const excerpts: string[] = [];
        for (const [index, sample] of samples.entries()) {
          const excerpt = path.join(workDir, `clone-sample-${index}.mp3`);
          await trimCloneSample(sample.path, excerpt, {
            durationSeconds: sample.probe.durationSeconds,
            capSeconds: cloneSampleSeconds,
            skipIntro: !internal,
            signal,
          });
          excerpts.push(excerpt);
        }
        throwIfClientGone(signal, "Voice cloning");
        // Not cancelled if the client leaves: an interrupted create can leave
        // a voice whose id we never learn. It finishes, and is deleted below.
        const voice = await getElevenLabs().voices.ivc.create({
          name,
          files: excerpts.map((excerpt) => fs.createReadStream(excerpt)),
        });
        return voice.voiceId;
      });
      voiceId = acquired.voiceId;

      const delivered = await sendJsonConfirmed(res, { voiceId, reused: acquired.reused });
      const settled = voiceId;
      voiceId = null;
      if (delivered) {
        voiceRegistry.delivered(settled);
      } else {
        console.warn("Clone client disconnected before receiving its voice; releasing it.");
        await voiceRegistry.abandoned(settled);
      }
    } catch (error) {
      if (voiceId) await voiceRegistry.abandoned(voiceId);
      sendFailure(res, "Clone failed.", error, { signal });
    } finally {
      await removeTempDir(tempDir);
    }
  }),
);

/**
 * Delete a voice this process cloned, now rather than at its TTL. The UI calls
 * it when its file is replaced and when the page is closed. Idempotent; a
 * voice this process did not create is a 404 and is never deleted.
 */
cloneRouter.post(
  "/api/clone/:voiceId/release",
  asyncHandler(async (req, res) => {
    const { voiceId } = req.params;
    if (!VOICE_ID_PATTERN.test(voiceId)) throw new HttpError(400, "Invalid voice id.");
    const outcome = await voiceRegistry.release(voiceId);
    if (outcome === "unknown") {
      throw new HttpError(404, "This server has no cloned voice with that id.");
    }
    res.json({ voiceId, released: outcome === "released" });
  }),
);
