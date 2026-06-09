import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Router } from "express";
import { upload } from "../middleware/upload";
import { runPythonModule } from "../services/python";
import { apiBaseUrl } from "../config";

export const generateRouter = Router();

generateRouter.post("/api/generate", upload.single("audio"), async (req, res) => {
  const audioFile = req.file;
  const voiceId = req.body?.voiceId?.toString()?.trim();
  const voiceIds = req.body?.voiceIds?.toString()?.trim();
  const brand = req.body?.brand?.toString()?.trim();
  const productDesc = req.body?.productDesc?.toString()?.trim();

  if (!audioFile) {
    res.status(400).json({ error: "audio file is required." });
    return;
  }

  if (!voiceId && !voiceIds) {
    res.status(400).json({ error: "voiceId or voiceIds is required." });
    return;
  }

  if (!brand) {
    res.status(400).json({ error: "brand is required." });
    return;
  }

  const resolvedDesc = productDesc || `A short audio ad spot for ${brand}.`;

  const tempDir = await fs.promises.mkdtemp(
    path.join(os.tmpdir(), "ad-generate-"),
  );
  const basePath = path.join(tempDir, "base.mp3");
  const outPath = path.join(tempDir, "out.mp3");
  const cleanup = async () => {
    await Promise.all(
      [basePath, outPath].map((filePath) =>
        fs.promises.unlink(filePath).catch(() => undefined),
      ),
    );
    await fs.promises.rmdir(tempDir).catch(() => undefined);
  };

  try {
    await fs.promises.writeFile(basePath, audioFile.buffer);

    const args = [
      "-m",
      "ad_inserter.cli",
      "--main",
      basePath,
      ...(voiceIds ? ["--voice-ids", voiceIds] : ["--voice-id", voiceId]),
      "--product-name",
      brand,
      "--product-desc",
      resolvedDesc,
      "--out",
      outPath,
      "--tts-url",
      `${apiBaseUrl}/api/tts`,
      "--merge-url",
      `${apiBaseUrl}/api/merge`,
    ];

    await runPythonModule(args, "ad_inserter");
    await fs.promises.access(outPath);
    res.setHeader("Content-Type", "audio/mpeg");
    const stream = fs.createReadStream(outPath);
    stream.on("error", (streamError) => {
      if (!res.headersSent) {
        res.status(500).json({
          error:
            streamError instanceof Error
              ? streamError.message
              : "Failed to stream output.",
        });
      }
    });
    res.on("close", cleanup);
    res.on("finish", cleanup);
    stream.pipe(res);
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Generation failed.",
    });
    await cleanup();
  }
});
