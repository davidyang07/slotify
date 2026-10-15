import path from "node:path";
import { Router } from "express";
import { claimWorkDir, inspectUploads, upload } from "../middleware/upload";
import { runPythonModule } from "../services/python";
import { apiBaseUrl } from "../config";
import { leaseHeavyJob, refuseWhenBusy } from "../lib/concurrency";
import { sendFailure } from "../lib/errors";
import { asyncHandler, clientAbortSignal } from "../lib/http";
import { optionalString } from "../lib/request-fields";
import { sendAudioFile } from "../lib/stream";
import { removeTempDir } from "../lib/temp";

export const generateRouter = Router();

generateRouter.post("/api/generate", refuseWhenBusy(), upload.single("audio", { dirPrefix: "ad-generate-" }), asyncHandler(async (req, res) => {
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
  let releaseJob: (() => void) | null = null;
  try {
    // The upload's own dir is the pipeline's --work-dir; removed in `finally`.
    tempDir = await claimWorkDir(req, "ad-generate-");
    const [base] = await inspectUploads(req, { signal });
    const basePath = base.path;
    const outPath = path.join(tempDir, "out.mp3");
    releaseJob = await leaseHeavyJob(req, { signal });

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

    await runPythonModule(args, "ad_inserter", { signal, workDir: tempDir, kind: "generate" });
    releaseJob();
    await sendAudioFile(res, outPath);
  } catch (error) {
    sendFailure(res, "Generation failed.", error, { signal });
  } finally {
    releaseJob?.();
    await removeTempDir(tempDir);
  }
}));
