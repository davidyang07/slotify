import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { Router } from "express";
import {
  buildSponsorBlock,
  checkTtsBudget,
  getElevenLabs,
  MAX_TTS_PAUSE_MS,
  ttsModelFor,
  voiceRegistry,
} from "../services/elevenlabs";
import { generateBrandStatement } from "../services/openai";
import { sendFailure } from "../lib/errors";
import { asyncHandler, clientAbortSignal, throwIfClientGone } from "../lib/http";
import {
  optionalEnum,
  optionalNumber,
  optionalObject,
  optionalStatements,
  optionalString,
} from "../lib/request-fields";
import { SPONSOR_LANGUAGES, SPONSOR_TONES } from "../lib/sponsor-options";
import { sendAudioFile } from "../lib/stream";

export const ttsRouter = Router();

ttsRouter.post("/api/tts", asyncHandler(async (req, res) => {
  const body = req.body;
  const voiceId = optionalString(body, "voiceId", 200);
  const modelId = optionalString(body, "modelId", 100);
  const outputFormat = optionalString(body, "outputFormat", 100);
  const pauseMs = optionalNumber(body, "pauseMs");
  const tone = optionalEnum(body, "tone", SPONSOR_TONES);
  const language = optionalEnum(body, "language", SPONSOR_LANGUAGES);
  const sponsor = optionalObject(body, "sponsor") ?? {};
  // Read every alias so a malformed one is a 400 even when another is set.
  const statementFields = [
    optionalStatements(body, "statements"),
    optionalStatements(body, "texts"),
    optionalStatements(body, "text"),
  ];
  const name =
    optionalString(sponsor, "name") ?? optionalString(body, "name") ?? optionalString(body, "brand") ?? "";
  const productDesc =
    optionalString(sponsor, "productDesc") ?? optionalString(body, "productDesc") ?? "";
  let statements = statementFields.find((entry) => entry !== undefined) ?? [];

  if (!voiceId) {
    res.status(400).json({ error: "voiceId is required." });
    return;
  }

  if (pauseMs !== undefined && (pauseMs < 0 || pauseMs > MAX_TTS_PAUSE_MS)) {
    res.status(400).json({ error: `pauseMs must be between 0 and ${MAX_TTS_PAUSE_MS}.` });
    return;
  }

  // Every statement is a paid ElevenLabs call: the budget is checked before
  // the first one, and before OpenAI is asked to write anything.
  checkTtsBudget(statements, { pauseMs: pauseMs ?? 150 });

  if (voiceRegistry.wasReleased(voiceId)) {
    res.status(410).json({
      error: "This voice clone has expired and was deleted; clone the voice again.",
    });
    return;
  }

  const signal = clientAbortSignal(req, res);
  let cleanup: (() => Promise<void>) | null = null;
  try {
    if (statements.length === 0) {
      // Only copy the server writes takes the tone and language; a statement the
      // user supplied is read verbatim.
      const generated = await generateBrandStatement({ name, productDesc, tone, language });
      if (generated) {
        statements = [generated];
      }
    }

    if (statements.length === 0) {
      res.status(400).json({ error: "text is required." });
      return;
    }

    // Keeps a registry voice from expiring while it is in use.
    voiceRegistry.touch(voiceId);
    // Counts and lengths only: statement text is the customer's ad copy.
    console.log("TTS request:", {
      count: statements.length,
      lengths: statements.map((statement) => statement.length),
    });
    if (statements.length === 1) {
      throwIfClientGone(signal, "Text-to-speech");
      const audio = await getElevenLabs().textToSpeech.convert(
        voiceId,
        {
          text: statements[0],
          ...ttsModelFor(modelId, language),
          outputFormat: (outputFormat ?? "mp3_44100_128") as any,
        },
        { abortSignal: signal },
      );
      res.setHeader("Content-Type", "audio/mpeg");
      // pipeline, not pipe: a mid-stream ElevenLabs failure rejects here
      // instead of surfacing as an unhandled 'error' event.
      await pipeline(Readable.fromWeb(audio as any), res);
      return;
    }

    const sponsorBlock = await buildSponsorBlock({
      voiceId,
      statements,
      modelId,
      outputFormat,
      pauseMs: pauseMs ?? 150,
      language,
      signal,
    });
    cleanup = sponsorBlock.cleanup;
    await sendAudioFile(res, sponsorBlock.outputPath);
  } catch (error) {
    sendFailure(res, "TTS failed.", error, { signal });
  } finally {
    await cleanup?.();
  }
}));

ttsRouter.post("/api/speech", (_req, res) => {
  res.redirect(307, "/api/tts");
});
