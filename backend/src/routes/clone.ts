import fs from "node:fs";
import path from "node:path";
import { Router } from "express";
import { upload, MAX_CLONE_SAMPLES } from "../middleware/upload";
import { getElevenLabs } from "../services/elevenlabs";
import { failureBody } from "../lib/errors";
import { asyncHandler } from "../lib/http";
import { optionalString } from "../lib/request-fields";
import { makeTempDir, removeTempDir } from "../lib/temp";

export const cloneRouter = Router();

cloneRouter.post(
  "/api/clone",
  upload.array("files", MAX_CLONE_SAMPLES),
  asyncHandler(async (req, res) => {
    const files = (req.files as Express.Multer.File[]) ?? [];
    const name = optionalString(req.body, "name", 100) || "My Voice Clone";
    if (!Array.isArray(files) || files.length === 0) {
      res.status(400).json({ error: "No audio files uploaded." });
      return;
    }

    if (!process.env.ELEVENLABS_API_KEY) {
      res.status(500).json({
        error:
          "ELEVENLABS_API_KEY not configured. Please set it in your environment variables.",
      });
      return;
    }

    let tempDir: string | null = null;
    try {
      tempDir = await makeTempDir("voice-clone-");
      const tempPaths: string[] = [];
      for (const file of files) {
        const safeName = path.basename(file.originalname || "sample.wav");
        const tempPath = path.join(
          tempDir,
          `${Date.now()}-${Math.random().toString(16).slice(2)}-${safeName}`,
        );
        await fs.promises.writeFile(tempPath, file.buffer);
        tempPaths.push(tempPath);
      }

      const streams = tempPaths.map((tempPath) => fs.createReadStream(tempPath));
      const voice = await getElevenLabs().voices.ivc.create({
        name,
        files: streams,
      });

      res.json({ voiceId: voice.voiceId });
    } catch (error) {
      res.status(500).json(failureBody("Clone failed.", error));
    } finally {
      await removeTempDir(tempDir);
    }
  }),
);
