import path from "node:path";
import { Router } from "express";
import { claimWorkDir, inspectUploads, upload, type InspectedUpload } from "../middleware/upload";
import { runFfmpeg, getAudioDuration } from "../services/ffmpeg";
import { leaseHeavyJob, refuseWhenBusy } from "../lib/concurrency";
import { buildMergeFilter } from "../lib/merge-filter";
import { sendFailure } from "../lib/errors";
import { asyncHandler, clientAbortSignal } from "../lib/http";
import { optionalNumber, optionalString } from "../lib/request-fields";
import { sendAudioFile } from "../lib/stream";
import { removeTempDir } from "../lib/temp";

export const mergeRouter = Router();

// Bounds keep a typo from asking ffmpeg for hours of silence.
const MAX_CROSSFADE_SECONDS = 5;
const MAX_PAUSE_SECONDS = 10;

mergeRouter.post(
  "/api/merge",
  refuseWhenBusy(),
  upload.fields(
    [
      { name: "audio", maxCount: 1 },
      { name: "insert", maxCount: 1 },
    ],
    { dirPrefix: "voice-merge-" },
  ),
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
    let releaseJob: (() => void) | null = null;
    try {
      tempDir = await claimWorkDir(req, "voice-merge-");
      await inspectUploads(req, { signal });
      // Renamed by inspectUploads to fixed names with their real extensions.
      const basePath = audioFile.path;
      const insertPath = insertFile.path;
      const outPath = path.join(tempDir, "merged.mp3");
      releaseJob = await leaseHeavyJob(req, { signal });

      // Durations come from the inspection's ffprobe; used to clamp insertAt.
      const mainDuration = (audioFile as InspectedUpload).probe.durationSeconds;
      const adDuration = (insertFile as InspectedUpload).probe.durationSeconds;

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

      // The slot is for the ffmpeg work, not for a slow client's download.
      releaseJob();
      await sendAudioFile(res, outPath);
    } catch (error) {
      sendFailure(res, "Merge failed.", error, { signal });
    } finally {
      releaseJob?.();
      await removeTempDir(tempDir);
    }
  }),
);
