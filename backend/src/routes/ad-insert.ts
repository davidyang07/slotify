import path from "node:path";
import { Router } from "express";
import { claimWorkDir, inspectUploads, upload } from "../middleware/upload";
import { runPythonModule } from "../services/python";
import { apiBaseUrl, allowedOpenAiModels } from "../config";
import { leaseHeavyJob, refuseWhenBusy } from "../lib/concurrency";
import { sendFailure } from "../lib/errors";
import { asyncHandler, clientAbortSignal } from "../lib/http";
import { optionalBoolean, optionalEnum, optionalString } from "../lib/request-fields";
import { sendAudioFile } from "../lib/stream";
import { removeTempDir } from "../lib/temp";

export const adInsertRouter = Router();

const AD_STYLES = ["casual", "serious", "funny"] as const;
const AD_MODES = ["A_ONLY", "B_ONLY", "DUO"] as const;
const LLM_PROVIDERS = ["openai", "none"] as const;

adInsertRouter.post("/ad/insert", refuseWhenBusy(), upload.single("audio", { dirPrefix: "two-speaker-ad-" }), asyncHandler(async (req, res) => {
  const audioFile = req.file;
  const productName = optionalString(req.body, "productName", 200);
  const productBlurb = optionalString(req.body, "productBlurb");
  const adStyle = optionalString(req.body, "adStyle", 100);
  const adMode = optionalString(req.body, "adMode", 100);
  const voiceIdA = optionalString(req.body, "voiceIdA", 200);
  const voiceIdB = optionalString(req.body, "voiceIdB", 200);
  const llmProvider = optionalEnum(req.body, "llmProvider", LLM_PROVIDERS);
  const llmModel = optionalString(req.body, "llmModel", 200);
  const cloneVoices = optionalBoolean(req.body, "cloneVoices") ?? false;

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

  if (!adStyle || !(AD_STYLES as readonly string[]).includes(adStyle)) {
    res
      .status(400)
      .json({ error: "adStyle must be casual, serious, or funny." });
    return;
  }

  if (!adMode || !(AD_MODES as readonly string[]).includes(adMode)) {
    res
      .status(400)
      .json({ error: "adMode must be A_ONLY, B_ONLY, or DUO." });
    return;
  }

  // The client picks among the models the operator pays for, not any model
  // OpenAI sells. ad_inserter checks the same SLOTIFY_OPENAI_MODELS list.
  const allowedModels = allowedOpenAiModels();
  if (llmModel && !allowedModels.includes(llmModel)) {
    res.status(400).json({
      error: `llmModel must be one of: ${allowedModels.join(", ")} (see SLOTIFY_OPENAI_MODELS).`,
    });
    return;
  }

  const signal = clientAbortSignal(req, res);
  let tempDir: string | null = null;
  let releaseJob: (() => void) | null = null;
  try {
    // The upload's own dir is the pipeline's --work-dir; removed in `finally`.
    tempDir = await claimWorkDir(req, "two-speaker-ad-");
    const [base] = await inspectUploads(req, { signal });
    const basePath = base.path;
    const outPath = path.join(tempDir, "out.mp3");
    releaseJob = await leaseHeavyJob(req, { signal });

    // `--flag=value`: a product name such as "-Sale" is otherwise read by
    // argparse as an option and the pipeline exits 2.
    const args = [
      "-m",
      "ad_inserter.insert_ad",
      `--input=${basePath}`,
      `--product-name=${productName}`,
      `--product-blurb=${productBlurb}`,
      `--ad-style=${adStyle}`,
      `--ad-mode=${adMode}`,
      `--out=${outPath}`,
      `--tts-url=${apiBaseUrl}/api/tts`,
      `--clone-url=${apiBaseUrl}/api/clone`,
    ];

    if (voiceIdA) {
      args.push(`--voice-id-a=${voiceIdA}`);
    }
    if (voiceIdB) {
      args.push(`--voice-id-b=${voiceIdB}`);
    }
    if (llmProvider) {
      args.push(`--llm-provider=${llmProvider}`);
    }
    if (llmModel) {
      args.push(`--llm-model=${llmModel}`);
    }
    if (cloneVoices) {
      args.push("--clone-voices");
    }

    await runPythonModule(args, "ad_inserter.insert_ad", {
      signal,
      workDir: tempDir,
      kind: "ad-insert",
    });
    releaseJob();
    await sendAudioFile(res, outPath);
  } catch (error) {
    sendFailure(res, "Insertion failed.", error, { signal });
  } finally {
    releaseJob?.();
    await removeTempDir(tempDir);
  }
}));
