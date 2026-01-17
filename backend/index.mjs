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
import { fileURLToPath } from "node:url";

const app = express();
const port = Number.parseInt(process.env.PORT ?? "3001", 10);

// Check for ElevenLabs API key
if (!process.env.ELEVENLABS_API_KEY) {
  console.warn(
    "Warning: ELEVENLABS_API_KEY not set. Voice cloning and TTS will fail.",
  );
}

const elevenlabs = new ElevenLabsClient({
  apiKey: process.env.ELEVENLABS_API_KEY,
});
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

// Health check endpoint
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    timestamp: new Date().toISOString(),
    elevenlabsConfigured: !!process.env.ELEVENLABS_API_KEY,
  });
});

app.post("/api/clone", upload.array("files"), async (req, res) => {
  const files = req.files ?? [];
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
    console.error("Clone error:", error);
    const errorMessage =
      error instanceof Error ? error.message : "Clone failed.";
    res.status(500).json({
      error: errorMessage,
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

const getAudioDuration = async (filePath) =>
  new Promise((resolve, reject) => {
    const process = spawn(
      "ffprobe",
      [
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        filePath,
      ],
      { stdio: ["ignore", "pipe", "pipe"] },
    );
    let stdout = "";
    let stderr = "";
    process.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    process.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    process.on("error", reject);
    process.on("close", (code) => {
      if (code === 0) {
        const duration = Number.parseFloat(stdout.trim());
        resolve(Number.isFinite(duration) && duration > 0 ? duration : null);
      } else {
        reject(new Error(stderr || `ffprobe exited with code ${code}`));
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
    const crossfade = Number.parseFloat(req.body?.crossfade ?? "0.08");
    const pause = Number.parseFloat(req.body?.pause ?? "0.2");

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
      const expectedDuration =
        mainDuration !== null && adDuration !== null
          ? mainDuration + adDuration
          : null;
      console.log("Merge debug:", {
        mainDuration: mainDuration?.toFixed(3),
        insertDuration: adDuration?.toFixed(3),
        insertAt: insertAt.toFixed(3),
        expectedDuration: expectedDuration?.toFixed(3),
      });

      const filter = [
        `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
          `atrim=0:${clampedInsertAt},asetpts=PTS-STARTPTS[a0]`,
        `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
          `atrim=${clampedInsertAt},asetpts=PTS-STARTPTS[a1]`,
        `[1:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
          `asetpts=PTS-STARTPTS[ad]`,
        `[a0][ad][a1]concat=n=3:v=0:a=1[out]`,
      ].join(";");

      // Input 0: main audio
      // Input 1: insert audio
      const args = [
        "-y",
        "-i", basePath,
        "-i", insertPath,
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

app.post("/api/insert-sections", upload.single("audio"), async (req, res) => {
  const audioFile = req.file;
  const count = Number.parseInt(req.body?.count ?? "5", 10);

  if (!audioFile) {
    res.status(400).json({ error: "audio file is required." });
    return;
  }

  if (!process.env.OPENAI_API_KEY) {
    res.status(500).json({
      error: "OPENAI_API_KEY not configured. Please set it in your environment.",
    });
    return;
  }

  const tempDir = await fs.promises.mkdtemp(
    path.join(os.tmpdir(), "insert-sections-"),
  );
  const audioPath = path.join(
    tempDir,
    `${Date.now()}-${audioFile.originalname || "audio.mp3"}`,
  );
  const cleanup = async () => {
    await fs.promises.unlink(audioPath).catch(() => undefined);
    await fs.promises.rmdir(tempDir).catch(() => undefined);
  };

  try {
    await fs.promises.writeFile(audioPath, audioFile.buffer);
    const duration = await getAudioDuration(audioPath).catch(() => null);

    const transcriptForm = new FormData();
    transcriptForm.append("model", "whisper-1");
    transcriptForm.append("response_format", "verbose_json");
    transcriptForm.append(
      "file",
      new Blob([audioFile.buffer]),
      audioFile.originalname || "audio.mp3",
    );

    const transcriptResponse = await fetch(
      "https://api.openai.com/v1/audio/transcriptions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        },
        body: transcriptForm,
      },
    );

    if (!transcriptResponse.ok) {
      const detail = await transcriptResponse.text();
      throw new Error(
        `Transcription failed: ${transcriptResponse.status} ${detail}`,
      );
    }

    const transcript = await transcriptResponse.json();
    const segments = Array.isArray(transcript?.segments)
      ? transcript.segments
      : [];

    const candidates = [];
    let prevEnd = 0;
    for (const segment of segments) {
      const start = Number(segment.start ?? 0);
      const end = Number(segment.end ?? 0);
      if (Number.isFinite(start) && start - prevEnd >= 0.4) {
        candidates.push((start + prevEnd) / 2);
      }
      if (Number.isFinite(end)) {
        prevEnd = Math.max(prevEnd, end);
      }
    }
    if (duration !== null && duration - prevEnd >= 0.4) {
      candidates.push((duration + prevEnd) / 2);
    }

    const prompt = [
      "Pick natural ad insertion points for the audio.",
      "Return JSON with a points array of seconds (numbers).",
      `Max points: ${Number.isFinite(count) ? count : 5}.`,
      duration !== null
        ? `Audio duration: ${duration.toFixed(2)} seconds.`
        : "Audio duration unknown.",
      candidates.length > 0
        ? `Candidate silent midpoints (seconds): ${candidates
            .map((value) => value.toFixed(2))
            .join(", ")}.`
        : "No candidate silences detected; pick evenly spaced points.",
      `Transcript segments: ${segments
        .map((segment) => {
          const start = Number(segment.start ?? 0);
          const end = Number(segment.end ?? 0);
          const text = String(segment.text ?? "").trim();
          return `${start.toFixed(2)}-${end.toFixed(2)}: ${text}`;
        })
        .join(" | ")}`,
    ].join(" ");

    let points = [];
    const completionResponse = await fetch(
      "https://api.openai.com/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          temperature: 0.2,
          messages: [
            {
              role: "system",
              content:
                "Choose insertion points based on when the person talking in the audio clip completley finishes a sentence. Look for pauses between sentences. Choose points where the waveform is lowest. Never choose in between sentences.",
            },
            { role: "user", content: prompt },
          ],
          response_format: { type: "json_object" },
        }),
      },
    );

    if (!completionResponse.ok) {
      const detail = await completionResponse.text();
      throw new Error(
        `OpenAI selection failed: ${completionResponse.status} ${detail}`,
      );
    }

    const completion = await completionResponse.json();
    const rawContent =
      completion?.choices?.[0]?.message?.content?.trim() ?? "{}";
    try {
      const parsed = JSON.parse(rawContent);
      if (Array.isArray(parsed.points)) {
        points = parsed.points;
      }
    } catch (parseError) {
      points = [];
    }

    if (points.length === 0 && candidates.length > 0) {
      points = candidates;
    }

    const maxPoints = Number.isFinite(count) ? count : 5;
    const normalized = points
      .filter((value) => Number.isFinite(value))
      .map((value) => Number(value))
      .filter((value) => (duration ? value >= 0 && value <= duration : value >= 0))
      .sort((a, b) => a - b)
      .slice(0, Math.max(1, maxPoints));

    res.json({
      points: normalized,
      duration,
      source: normalized.length === points.length ? "openai" : "fallback",
    });
  } catch (error) {
    res.status(500).json({
      error:
        error instanceof Error ? error.message : "Insert analysis failed.",
    });
  } finally {
    await cleanup();
  }
});

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

app.post("/api/generate", upload.single("audio"), async (req, res) => {
  const audioFile = req.file;
  const voiceId = req.body?.voiceId?.toString()?.trim();
  const brand = req.body?.brand?.toString()?.trim();
  const productDesc = req.body?.productDesc?.toString()?.trim();

  if (!audioFile) {
    res.status(400).json({ error: "audio file is required." });
    return;
  }

  if (!voiceId) {
    res.status(400).json({ error: "voiceId is required." });
    return;
  }

  if (!brand) {
    res.status(400).json({ error: "brand is required." });
    return;
  }

  const resolvedDesc =
    productDesc || `A short audio ad spot for ${brand}.`;

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

    const pythonBin = process.env.PYTHON_BIN ?? "python";
    const baseUrl =
      process.env.API_BASE_URL ?? `http://localhost:${port}`;

    const args = [
      "-m",
      "ad_inserter.cli",
      "--main",
      basePath,
      "--voice-id",
      voiceId,
      "--product-name",
      brand,
      "--product-desc",
      resolvedDesc,
      "--out",
      outPath,
      "--tts-url",
      `${baseUrl}/api/tts`,
      "--merge-url",
      `${baseUrl}/api/merge`,
    ];

    await new Promise((resolve, reject) => {
      const child = spawn(pythonBin, args, {
        cwd: path.dirname(fileURLToPath(import.meta.url)),
        env: process.env,
      });
      let stderr = "";
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
      child.on("error", reject);
      child.on("close", (code) => {
        if (code === 0) {
          resolve(null);
        } else {
          reject(
            new Error(
              stderr || `ad_inserter exited with code ${code}`,
            ),
          );
        }
      });
    });
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

app.listen(port, () => {
  console.log(`API listening on http://localhost:${port}`);
  console.log(`CORS allowed origins: ${allowedOrigins.join(", ")}`);
  if (!process.env.ELEVENLABS_API_KEY) {
    console.warn(
      "⚠️  ELEVENLABS_API_KEY not set. Voice cloning and TTS endpoints will fail.",
    );
  } else {
    console.log("✓ ElevenLabs API key configured");
  }
});
