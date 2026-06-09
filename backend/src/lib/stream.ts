import { Readable } from "node:stream";

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
