import fs from "node:fs";
import path from "node:path";
import { Router } from "express";
import { upload } from "../middleware/upload";
import { runFfmpeg, getAudioDuration } from "../services/ffmpeg";
import { buildMergeFilter } from "../lib/merge-filter";
import { failureBody } from "../lib/errors";
import { asyncHandler, clientAbortSignal } from "../lib/http";
import { optionalNumber, optionalString } from "../lib/request-fields";
import { sendAudioFile } from "../lib/stream";
import { makeTempDir, removeTempDir } from "../lib/temp";

export const mergeRouter = Router();

// Bounds keep a typo from asking ffmpeg for hours of silence.
const MAX_CROSSFADE_SECONDS = 5;
const MAX_PAUSE_SECONDS = 10;

mergeRouter.post(
  "/api/merge",
  upload.fields([
    { name: "audio", maxCount: 1 },
    { name: "insert", maxCount: 1 },
  ]),
  asyncHandler(async (req, res) => {
    const filesMap = req.files as
      | Record<string, Express.Multer.File[]>
      | undefined;
    const audioFile = filesMap?.audio?.[0];
    const insertFile = filesMap?.insert?.[0];
    const insertAt = optionalNumber(req.body, "insertAt") ?? Number.NaN;
    const crossfade = optionalNumber(req.body, "crossfade") ?? 0.08;
    const pause = optionalNumber(req.body, "pause") ?? 0.2;
    const previewFlag =
      optionalString(req.body, "preview", 20) ?? optionalString(req.body, "mode", 20);
    const previewSeconds = optionalNumber(req.body, "previewSeconds") ?? 3;
    const isPreview =
      previewFlag === "1" ||
      previewFlag === "true" ||
      previewFlag === "preview";

    if (!audioFile || !insertFile) {
      res.status(400).json({ error: "audio and insert files are required." });
      return;
    }

    if (!Number.isFinite(insertAt) || insertAt < 0) {
      res.status(400).json({ error: "insertAt must be a positive number." });
      return;
    }

    if (crossfade < 0 || crossfade > MAX_CROSSFADE_SECONDS) {
      res.status(400).json({
        error: `crossfade must be between 0 and ${MAX_CROSSFADE_SECONDS} seconds.`,
      });
      return;
    }

    if (pause < 0 || pause > MAX_PAUSE_SECONDS) {
      res.status(400).json({ error: `pause must be between 0 and ${MAX_PAUSE_SECONDS} seconds.` });
      return;
    }

    if (previewSeconds < 0) {
      res.status(400).json({ error: "previewSeconds must be a positive number." });
      return;
    }

    const signal = clientAbortSignal(req, res);
    let tempDir: string | null = null;
    try {
      tempDir = await makeTempDir("voice-merge-");
      const basePath = path.join(tempDir, "base.mp3");
      const insertPath = path.join(tempDir, "insert.mp3");
      const outPath = path.join(tempDir, "merged.mp3");
      await fs.promises.writeFile(basePath, audioFile.buffer);
      await fs.promises.writeFile(insertPath, insertFile.buffer);

      // Get main audio duration to validate and clamp insertAt
      const mainDuration = await getAudioDuration(basePath, { signal });
      const adDuration = await getAudioDuration(insertPath, { signal });

      // Validate and clamp insertAt
      let clampedInsertAt = insertAt;
      if (mainDuration !== null) {
        if (clampedInsertAt < 0) {
          clampedInsertAt = 0;
        } else if (clampedInsertAt > mainDuration) {
          clampedInsertAt = mainDuration;
        }
      }

      // Debug logging
      let expectedDuration: number | null = null;
      if (adDuration !== null) {
        const breakDuration = adDuration + 2 * pause;
        if (isPreview) {
          const start = Math.max(0, clampedInsertAt - previewSeconds);
          const end =
            mainDuration !== null
              ? Math.min(mainDuration, clampedInsertAt + previewSeconds)
              : clampedInsertAt + previewSeconds;
          expectedDuration = Math.max(0, end - start) + breakDuration;
        } else if (mainDuration !== null) {
          expectedDuration = mainDuration + breakDuration;
        }
      }
      console.log("Merge debug:", {
        mainDuration: mainDuration?.toFixed(3),
        insertDuration: adDuration?.toFixed(3),
        insertAt: insertAt.toFixed(3),
        crossfade: crossfade.toFixed(3),
        pause: pause.toFixed(3),
        expectedDuration: expectedDuration?.toFixed(3),
      });

      const { filter, previewStart, previewEnd } = buildMergeFilter({
        insertAt: clampedInsertAt,
        previewSeconds: isPreview ? previewSeconds : 0,
        mainDuration,
        crossfade,
        pause,
      });

      if (isPreview) {
        console.log("Preview window:", {
          start: previewStart?.toFixed(3),
          end: previewEnd?.toFixed(3),
        });
      }

      // Input 0: main audio
      // Input 1: insert audio
      const args = [
        "-y",
        "-i",
        basePath,
        "-i",
        insertPath,
        "-filter_complex",
        filter,
        "-map",
        "[out]",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",
        outPath,
      ];

      await runFfmpeg(args, { signal });

      // Log final duration for verification
      const finalDuration = await getAudioDuration(outPath, { signal }).catch(() => null);
      if (finalDuration !== null) {
        console.log("Merge result:", {
          finalDuration: finalDuration.toFixed(3),
          expectedDuration: expectedDuration?.toFixed(3),
          match:
            expectedDuration !== null
              ? Math.abs(finalDuration - expectedDuration) < 0.1
              : "unknown",
        });
      }

      await sendAudioFile(res, outPath);
    } catch (error) {
      const body = failureBody("Merge failed.", error);
      if (!res.headersSent) {
        res.status(500).json(body);
      } else {
        res.destroy();
      }
    } finally {
      await removeTempDir(tempDir);
    }
  }),
);
