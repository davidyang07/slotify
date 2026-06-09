import fs from "node:fs";
import { Readable } from "node:stream";
import { Router } from "express";
import { elevenlabs, buildSponsorBlock } from "../services/elevenlabs";
import { generateBrandStatement } from "../services/openai";
import { normalizeStatements } from "../lib/text";

export const ttsRouter = Router();

ttsRouter.post("/api/tts", async (req, res) => {
  const { voiceId, modelId, outputFormat, pauseMs } = req.body ?? {};
  let statements = normalizeStatements(
    req.body?.statements ?? req.body?.texts ?? req.body?.text,
  );
  if (!voiceId) {
    res.status(400).json({ error: "voiceId is required." });
    return;
  }

  if (statements.length === 0) {
    const sponsor = req.body?.sponsor ?? {};
    const name = String(
      sponsor?.name ?? req.body?.name ?? req.body?.brand ?? "",
    ).trim();
    const productDesc = String(
      sponsor?.productDesc ?? req.body?.productDesc ?? "",
    ).trim();
    const generated = await generateBrandStatement({ name, productDesc });
    if (generated) {
      statements = [generated];
    }
  }

  if (statements.length === 0) {
    res.status(400).json({ error: "text is required." });
    return;
  }

  try {
    console.log("TTS statements:", { count: statements.length, statements });
    if (statements.length === 1) {
      const audio = await elevenlabs.textToSpeech.convert(voiceId, {
        text: statements[0],
        modelId: modelId ?? "eleven_multilingual_v2",
        outputFormat: outputFormat ?? "mp3_44100_128",
      });
      res.setHeader("Content-Type", "audio/mpeg");
      const stream = Readable.fromWeb(audio as any);
      stream.pipe(res);
      return;
    }

    const pauseMsValue = Number.parseFloat(pauseMs ?? "150");
    const normalizedPauseMs = Number.isFinite(pauseMsValue) ? pauseMsValue : 150;

    const sponsorBlock = await buildSponsorBlock({
      voiceId,
      statements,
      modelId,
      outputFormat,
      pauseMs: normalizedPauseMs,
    });

    res.setHeader("Content-Type", "audio/mpeg");
    const stream = fs.createReadStream(sponsorBlock.outputPath);
    stream.on("error", (streamError) => {
      if (!res.headersSent) {
        res.status(500).json({
          error:
            streamError instanceof Error
              ? streamError.message
              : "Failed to stream TTS audio.",
        });
      }
    });
    res.on("close", sponsorBlock.cleanup);
    res.on("finish", sponsorBlock.cleanup);
    stream.pipe(res);
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "TTS failed.",
    });
  }
});

ttsRouter.post("/api/speech", async (req, res) => {
  res.redirect(307, "/api/tts");
});
