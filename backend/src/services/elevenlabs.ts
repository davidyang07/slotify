import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";
import { runFfmpeg, getAudioDuration } from "./ffmpeg";
import { streamToBuffer } from "../lib/stream";

export const elevenlabs = new ElevenLabsClient({
  apiKey: process.env.ELEVENLABS_API_KEY,
});

export interface SponsorBlock {
  tempDir: string;
  outputPath: string;
  duration: number | null;
  cleanup: () => Promise<void>;
}

/**
 * Render one or more spoken statements into a single normalized sponsor MP3,
 * separated by short silences. Caller is responsible for invoking `cleanup`.
 */
export const buildSponsorBlock = async ({
  voiceId,
  statements,
  modelId,
  outputFormat,
  pauseMs,
}: {
  voiceId: string;
  statements: string[];
  modelId?: string;
  outputFormat?: string;
  pauseMs?: number;
}): Promise<SponsorBlock> => {
  const tempDir = await fs.promises.mkdtemp(
    path.join(os.tmpdir(), "sponsor-block-"),
  );
  const cleanup = async (paths: string[]) => {
    await Promise.all(
      paths.map((filePath) =>
        fs.promises.unlink(filePath).catch(() => undefined),
      ),
    );
    await fs.promises.rmdir(tempDir).catch(() => undefined);
  };

  const statementPaths: string[] = [];
  const allPaths: string[] = [];
  try {
    for (let index = 0; index < statements.length; index += 1) {
      const statement = statements[index];
      const audio = await elevenlabs.textToSpeech.convert(voiceId, {
        text: statement,
        modelId: modelId ?? "eleven_multilingual_v2",
        outputFormat: (outputFormat ?? "mp3_44100_128") as any,
      });
      const buffer = await streamToBuffer(audio);
      const statementPath = path.join(tempDir, `statement-${index}.mp3`);
      await fs.promises.writeFile(statementPath, buffer);
      statementPaths.push(statementPath);
      allPaths.push(statementPath);
    }

    const pauseSeconds = Math.max(0, (pauseMs ?? 150) / 1000);
    const pausePath = path.join(tempDir, "pause.mp3");
    if (statements.length > 1 && pauseSeconds > 0) {
      await runFfmpeg([
        "-y",
        "-f",
        "lavfi",
        "-i",
        `anullsrc=channel_layout=stereo:sample_rate=44100`,
        "-t",
        pauseSeconds.toString(),
        pausePath,
      ]);
      allPaths.push(pausePath);
    }

    const concatInputs: string[] = [];
    const inputPaths: string[] = [];
    statementPaths.forEach((statementPath, index) => {
      inputPaths.push(statementPath);
      concatInputs.push(statementPath);
      if (index < statementPaths.length - 1 && pauseSeconds > 0) {
        inputPaths.push(pausePath);
        concatInputs.push(pausePath);
      }
    });

    const outputPath = path.join(tempDir, "sponsor_block.mp3");
    if (concatInputs.length === 1) {
      await fs.promises.copyFile(concatInputs[0], outputPath);
    } else {
      const filterParts = inputPaths.map(
        (_, idx) =>
          `[${idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
          `asetpts=PTS-STARTPTS[a${idx}]`,
      );
      const concatLabels = inputPaths.map((_, idx) => `[a${idx}]`).join("");
      const filter = [
        ...filterParts,
        `${concatLabels}concat=n=${inputPaths.length}:v=0:a=1[concat]`,
        `[concat]loudnorm=I=-16:TP=-1.5:LRA=11[out]`,
      ].join(";");
      const args = [
        "-y",
        ...inputPaths.flatMap((inputPath) => ["-i", inputPath]),
        "-filter_complex",
        filter,
        "-map",
        "[out]",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",
        outputPath,
      ];
      await runFfmpeg(args);
    }

    const duration = await getAudioDuration(outputPath);
    if (duration !== null && duration > 20) {
      throw new Error("Sponsor block exceeds 20s limit.");
    }

    return {
      tempDir,
      outputPath,
      duration,
      cleanup: () => cleanup([...allPaths, outputPath]),
    };
  } catch (error) {
    await cleanup(allPaths);
    throw error;
  }
};
