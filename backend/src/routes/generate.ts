import fs from "node:fs";
import path from "node:path";
import { Router } from "express";
import { upload } from "../middleware/upload";
import { runPythonModule } from "../services/python";
import { apiBaseUrl } from "../config";
import { failureBody } from "../lib/errors";
import { asyncHandler, clientAbortSignal } from "../lib/http";
import { optionalString } from "../lib/request-fields";
import { sendAudioFile } from "../lib/stream";
import { makeTempDir, removeTempDir } from "../lib/temp";

export const generateRouter = Router();

generateRouter.post("/api/generate", upload.single("audio"), asyncHandler(async (req, res) => {
  const audioFile = req.file;
  const voiceId = optionalString(req.body, "voiceId", 200);
  const voiceIds = optionalString(req.body, "voiceIds", 2_000);
  const brand = optionalString(req.body, "brand", 200);
  const productDesc = optionalString(req.body, "productDesc");

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
  const signal = clientAbortSignal(req, res);

  let tempDir: string | null = null;
  try {
    tempDir = await makeTempDir("ad-generate-");
    const basePath = path.join(tempDir, "base.mp3");
    const outPath = path.join(tempDir, "out.mp3");
    await fs.promises.writeFile(basePath, audioFile.buffer);

    // `--flag=value`: a brand such as "-Sale" is otherwise read by argparse as
    // an option and the pipeline exits 2.
    const args = [
      "-m",
      "ad_inserter.cli",
      `--main=${basePath}`,
      voiceIds ? `--voice-ids=${voiceIds}` : `--voice-id=${voiceId}`,
      `--product-name=${brand}`,
      `--product-desc=${resolvedDesc}`,
      `--out=${outPath}`,
      `--tts-url=${apiBaseUrl}/api/tts`,
      `--merge-url=${apiBaseUrl}/api/merge`,
    ];

    await runPythonModule(args, "ad_inserter", { signal });
    await sendAudioFile(res, outPath);
  } catch (error) {
    const body = failureBody("Generation failed.", error);
    if (!res.headersSent) {
      res.status(500).json(body);
    } else {
      res.destroy();
    }
  } finally {
    await removeTempDir(tempDir);
  }
}));
