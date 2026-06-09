import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Router } from "express";
import { upload } from "../middleware/upload";
import { runPythonModule } from "../services/python";
import { apiBaseUrl } from "../config";

export const adInsertRouter = Router();

adInsertRouter.post("/ad/insert", upload.single("audio"), async (req, res) => {
  const audioFile = req.file;
  const productName = req.body?.productName?.toString()?.trim();
  const productBlurb = req.body?.productBlurb?.toString()?.trim();
  const adStyle = req.body?.adStyle?.toString()?.trim();
  const adMode = req.body?.adMode?.toString()?.trim();
  const voiceIdA = req.body?.voiceIdA?.toString()?.trim();
  const voiceIdB = req.body?.voiceIdB?.toString()?.trim();
  const llmProvider = req.body?.llmProvider?.toString()?.trim();
  const llmModel = req.body?.llmModel?.toString()?.trim();
  const cloneVoices =
    req.body?.cloneVoices?.toString()?.trim().toLowerCase() === "true";

  if (!audioFile) {
    res.status(400).json({ error: "audio file is required." });
    return;
  }

  if (!productName) {
    res.status(400).json({ error: "productName is required." });
    return;
  }

  if (!productBlurb) {
    res.status(400).json({ error: "productBlurb is required." });
    return;
  }

  if (!adStyle || !["casual", "serious", "funny"].includes(adStyle)) {
    res
      .status(400)
      .json({ error: "adStyle must be casual, serious, or funny." });
    return;
  }

  if (!adMode || !["A_ONLY", "B_ONLY", "DUO"].includes(adMode)) {
    res
      .status(400)
      .json({ error: "adMode must be A_ONLY, B_ONLY, or DUO." });
    return;
  }

  const tempDir = await fs.promises.mkdtemp(
    path.join(os.tmpdir(), "two-speaker-ad-"),
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
      "ad_inserter.insert_ad",
      "--input",
      basePath,
      "--product-name",
      productName,
      "--product-blurb",
      productBlurb,
      "--ad-style",
      adStyle,
      "--ad-mode",
      adMode,
      "--out",
      outPath,
      "--tts-url",
      `${apiBaseUrl}/api/tts`,
      "--clone-url",
      `${apiBaseUrl}/api/clone`,
    ];

    if (voiceIdA) {
      args.push("--voice-id-a", voiceIdA);
    }
    if (voiceIdB) {
      args.push("--voice-id-b", voiceIdB);
    }
    if (llmProvider) {
      args.push("--llm-provider", llmProvider);
    }
    if (llmModel) {
      args.push("--llm-model", llmModel);
    }
    if (cloneVoices) {
      args.push("--clone-voices");
    }

    await runPythonModule(args, "ad_inserter.insert_ad");
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
      error: error instanceof Error ? error.message : "Insertion failed.",
    });
    await cleanup();
  }
});
