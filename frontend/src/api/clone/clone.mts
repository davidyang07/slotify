import "dotenv/config";
import fs from "node:fs";
import path from "node:path";

const apiBase = process.env.VITE_API_BASE_URL ?? "http://localhost:3001";
const voiceName = process.env.VOICE_NAME ?? "My Voice Clone";
const filePaths = process.argv.slice(2);

if (filePaths.length === 0) {
  throw new Error("Provide one or more audio file paths.");
}

const form = new FormData();
form.append("name", voiceName);

for (const filePath of filePaths) {
  const buffer = await fs.promises.readFile(filePath);
  const fileName = path.basename(filePath);
  form.append("files", new Blob([buffer]), fileName);
}

const response = await fetch(`${apiBase}/api/speech`, {
  method: "POST",
  body: form,
});

if (!response.ok) {
  const text = await response.text();
  throw new Error(`Clone failed: ${response.status} ${text}`);
}

const data = await response.json();
console.log(data.voiceId);
