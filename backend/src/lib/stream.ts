import fs from "node:fs";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { Response } from "express";

/** Collect a web ReadableStream into a single Buffer. */
export const streamToBuffer = async (
  webStream: ReadableStream,
): Promise<Buffer> => {
  const chunks: Buffer[] = [];
  const nodeStream = Readable.fromWeb(webStream as any);
  for await (const chunk of nodeStream) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
};

/**
 * Send a finished audio file as the response. Resolves once the client has it,
 * so a caller's `finally` can delete the file; rejects (with the response
 * destroyed) if the client leaves or the read fails mid-stream. Opening first
 * means a missing file is still reported as JSON, before any header is sent.
 */
export const sendAudioFile = async (res: Response, filePath: string): Promise<void> => {
  const handle = await fs.promises.open(filePath, "r");
  let size: number;
  try {
    ({ size } = await handle.stat());
  } catch (error) {
    await handle.close();
    throw error;
  }
  res.setHeader("Content-Type", "audio/mpeg");
  res.setHeader("Content-Length", String(size));
  await pipeline(handle.createReadStream(), res);
};
