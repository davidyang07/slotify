import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Router } from "express";
import { upload } from "../middleware/upload";
import { runFfmpeg, getAudioDuration } from "../services/ffmpeg";
import { buildMergeFilter } from "../lib/merge-filter";

export const mergeRouter = Router();

mergeRouter.post(
  "/api/merge",
  upload.fields([
    { name: "audio", maxCount: 1 },
    { name: "insert", maxCount: 1 },
  ]),
  async (req, res) => {
    const filesMap = req.files as
      | Record<string, Express.Multer.File[]>
      | undefined;
    const audioFile = filesMap?.audio?.[0];
    const insertFile = filesMap?.insert?.[0];
    const insertAt = Number.parseFloat(req.body?.insertAt ?? "");
    const crossfade = Number.parseFloat(req.body?.crossfade ?? "0.08");
    const pause = Number.parseFloat(req.body?.pause ?? "0.2");
    const previewFlag = req.body?.preview ?? req.body?.mode;
    const previewSecondsValue = Number.parseFloat(
      req.body?.previewSeconds ?? "3",
    );
    const previewSeconds = Number.isFinite(previewSecondsValue)
      ? previewSecondsValue
      : 3;
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

    if (!Number.isFinite(crossfade) || crossfade < 0) {
      res.status(400).json({ error: "crossfade must be a positive number." });
      return;
    }

    if (!Number.isFinite(pause) || pause < 0) {
      res.status(400).json({ error: "pause must be a positive number." });
      return;
    }

    const tempDir = await fs.promises.mkdtemp(
      path.join(os.tmpdir(), "voice-merge-"),
    );
    const basePath = path.join(tempDir, "base.mp3");
    const insertPath = path.join(tempDir, "insert.mp3");
    const outPath = path.join(tempDir, "merged.mp3");
    const cleanup = async () => {
      await Promise.all(
        [basePath, insertPath, outPath].map((filePath) =>
          fs.promises.unlink(filePath).catch(() => undefined),
        ),
      );
      await fs.promises.rmdir(tempDir).catch(() => undefined);
    };

    try {
      await fs.promises.writeFile(basePath, audioFile.buffer);
      await fs.promises.writeFile(insertPath, insertFile.buffer);

      // Get main audio duration to validate and clamp insertAt
      const mainDuration = await getAudioDuration(basePath);
      const adDuration = await getAudioDuration(insertPath);

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
        if (isPreview) {
          const start = Math.max(0, clampedInsertAt - previewSeconds);
          const end =
            mainDuration !== null
              ? Math.min(mainDuration, clampedInsertAt + previewSeconds)
              : clampedInsertAt + previewSeconds;
          expectedDuration = Math.max(0, end - start) + adDuration;
        } else if (mainDuration !== null) {
          expectedDuration = mainDuration + adDuration;
        }
      }
      console.log("Merge debug:", {
        mainDuration: mainDuration?.toFixed(3),
        insertDuration: adDuration?.toFixed(3),
        insertAt: insertAt.toFixed(3),
        expectedDuration: expectedDuration?.toFixed(3),
      });

      const { filter, previewStart, previewEnd } = buildMergeFilter({
        insertAt: clampedInsertAt,
        previewSeconds: isPreview ? previewSeconds : 0,
        mainDuration,
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

      await runFfmpeg(args);
      await fs.promises.access(outPath);

      // Log final duration for verification
      const finalDuration = await getAudioDuration(outPath).catch(() => null);
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

      res.setHeader("Content-Type", "audio/mpeg");

      const stream = fs.createReadStream(outPath);
      stream.on("error", (streamError) => {
        if (!res.headersSent) {
          res.status(500).json({
            error:
              streamError instanceof Error
                ? streamError.message
                : "Failed to stream merged audio.",
          });
        }
      });
      res.on("close", cleanup);
      res.on("finish", cleanup);
      stream.pipe(res);
    } catch (error) {
      res.status(500).json({
        error: error instanceof Error ? error.message : "Merge failed.",
      });
      await cleanup();
    }
  },
);
