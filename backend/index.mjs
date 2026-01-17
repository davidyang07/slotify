import "dotenv/config";
import cors from "cors";
import express from "express";
import multer from "multer";
import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { spawn } from "node:child_process";

const app = express();
const port = Number.parseInt(process.env.PORT ?? "3001", 10);
const elevenlabs = new ElevenLabsClient();
const upload = multer({ storage: multer.memoryStorage() });

const allowedOrigins = (process.env.CORS_ORIGIN ?? "http://localhost:5173")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

app.use(
  cors({
    origin(origin, callback) {
      if (!origin || allowedOrigins.includes(origin)) {
        callback(null, true);
        return;
      }
      callback(new Error("CORS blocked"));
    },
  }),
);
app.use(express.json({ limit: "2mb" }));

app.post("/api/clone", upload.array("files"), async (req, res) => {
  const files = req.files ?? [];
  if (!Array.isArray(files) || files.length === 0) {
    res.status(400).json({ error: "No audio files uploaded." });
    return;
  }

  const name = req.body?.name?.toString()?.trim() || "My Voice Clone";
  const tempDir = await fs.promises.mkdtemp(
    path.join(os.tmpdir(), "voice-clone-"),
  );
  const tempPaths = [];

  try {
    for (const file of files) {
      const safeName = path.basename(file.originalname || "sample.wav");
      const tempPath = path.join(
        tempDir,
        `${Date.now()}-${Math.random().toString(16).slice(2)}-${safeName}`,
      );
      await fs.promises.writeFile(tempPath, file.buffer);
      tempPaths.push(tempPath);
    }

    const streams = tempPaths.map((tempPath) =>
      fs.createReadStream(tempPath),
    );
    const voice = await elevenlabs.voices.ivc.create({
      name,
      files: streams,
    });

    res.json({ voiceId: voice.voiceId });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Clone failed.",
    });
  } finally {
    await Promise.all(
      tempPaths.map((tempPath) =>
        fs.promises.unlink(tempPath).catch(() => undefined),
      ),
    );
    await fs.promises.rmdir(tempDir).catch(() => undefined);
  }
});

const runFfmpeg = (args) =>
  new Promise((resolve, reject) => {
    const process = spawn("ffmpeg", args, { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    process.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    process.on("error", reject);
    process.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(stderr || `ffmpeg exited with code ${code}`));
      }
    });
  });

app.post(
  "/api/merge",
  upload.fields([
    { name: "audio", maxCount: 1 },
    { name: "insert", maxCount: 1 },
  ]),
  async (req, res) => {
    const audioFile = req.files?.audio?.[0];
    const insertFile = req.files?.insert?.[0];
    const insertAt = Number.parseFloat(req.body?.insertAt ?? "");
    const crossfade = Number.parseFloat(req.body?.crossfade ?? "0.15");
    const pause = Number.parseFloat(req.body?.pause ?? "0.12");

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

    try {
      await fs.promises.writeFile(basePath, audioFile.buffer);
      await fs.promises.writeFile(insertPath, insertFile.buffer);

      let filter;
      const baseHead = `[0:a]aresample=44100,aformat=channel_layouts=mono,atrim=0:${insertAt},asetpts=PTS-STARTPTS[a0]`;
      const baseTail = `[0:a]aresample=44100,aformat=channel_layouts=mono,atrim=${insertAt},asetpts=PTS-STARTPTS[a1]`;
      const insert = `[1:a]aresample=44100,aformat=channel_layouts=mono,asetpts=PTS-STARTPTS[ins]`;

      if (pause > 0) {
        filter = [
          baseHead,
          baseTail,
          insert,
          `anullsrc=r=44100:cl=mono,atrim=0:${pause}[sil]`,
          `[a0][sil][ins][sil][a1]concat=n=5:v=0:a=1[aout]`,
        ].join(";");
      } else if (crossfade > 0) {
        filter = [
          baseHead,
          baseTail,
          insert,
          `[a0][ins]acrossfade=d=${crossfade}:curve1=tri:curve2=tri[a01]`,
          `[a01][a1]acrossfade=d=${crossfade}:curve1=tri:curve2=tri[aout]`,
        ].join(";");
      } else {
        filter = [
          baseHead,
          baseTail,
          insert,
          `[a0][ins][a1]concat=n=3:v=0:a=1[aout]`,
        ].join(";");
      }

      const args = [
        "-y",
        "-i",
        basePath,
        "-i",
        insertPath,
        "-filter_complex",
        filter,
        "-map",
        "[aout]",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",
        outPath,
      ];

      await runFfmpeg(args);
      res.setHeader("Content-Type", "audio/mpeg");
      fs.createReadStream(outPath).pipe(res);
    } catch (error) {
      res.status(500).json({
        error: error instanceof Error ? error.message : "Merge failed.",
      });
    } finally {
      await Promise.all(
        [basePath, insertPath, outPath].map((filePath) =>
          fs.promises.unlink(filePath).catch(() => undefined),
        ),
      );
      await fs.promises.rmdir(tempDir).catch(() => undefined);
    }
  },
);

app.post("/api/tts", async (req, res) => {
  const { voiceId, text, modelId, outputFormat } = req.body ?? {};
  if (!voiceId || !text) {
    res.status(400).json({ error: "voiceId and text are required." });
    return;
  }

  try {
    const audio = await elevenlabs.textToSpeech.convert(voiceId, {
      text,
      modelId: modelId ?? "eleven_multilingual_v2",
      outputFormat: outputFormat ?? "mp3_44100_128",
    });

    res.setHeader("Content-Type", "audio/mpeg");
    const stream = Readable.fromWeb(audio);
    stream.pipe(res);
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "TTS failed.",
    });
  }
});

app.post("/api/speech", async (req, res) => {
  res.redirect(307, "/api/tts");
});

app.listen(port, () => {
  console.log(`API listening on http://localhost:${port}`);
});
