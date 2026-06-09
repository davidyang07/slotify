import path from "node:path";
import { fileURLToPath } from "node:url";

// Backend project root (the directory that contains the `ad_inserter` Python
// package). All Python subprocesses are spawned with this as their cwd so that
// `python -m ad_inserter.*` resolves.
export const BACKEND_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);

export const port = Number.parseInt(process.env.PORT ?? "3001", 10);

export const allowedOrigins = (
  process.env.CORS_ORIGIN ?? "http://localhost:5173"
)
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

export const pythonBin = process.env.PYTHON_BIN ?? "python";

// Base URL the Python pipeline uses to call back into this API (for TTS/merge).
export const apiBaseUrl =
  process.env.API_BASE_URL ?? `http://localhost:${port}`;
