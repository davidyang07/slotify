import type { InsertionMode, ScoredCandidate, Slot } from "../types";

const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions";
const OPENAI_TRANSCRIPTION_URL =
  "https://api.openai.com/v1/audio/transcriptions";

const generateStatementFallback = (name: string, productDesc: string): string => {
  const brand = name || "Our sponsor";
  const desc = productDesc || "a thoughtful companion for your day";
  return `${brand} supports this episode with ${desc}, offering a simple way to stay focused and refreshed.`;
};

/** Generate a single sponsor-read sentence, falling back to a template. */
export const generateBrandStatement = async ({
  name,
  productDesc,
}: {
  name: string;
  productDesc: string;
}): Promise<string> => {
  if (!process.env.OPENAI_API_KEY) {
    return generateStatementFallback(name, productDesc);
  }

  const prompt = [
    "Write one sponsor read sentence (8-12 seconds when spoken).",
    "Sound native to the episode, calm and conversational.",
    "Avoid hypey marketing language and emojis.",
    `Brand name: ${name || "Sponsor"}.`,
    productDesc ? `Product description: ${productDesc}.` : "",
  ]
    .filter(Boolean)
    .join(" ");

  try {
    const response = await fetch(OPENAI_CHAT_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        temperature: 0,
        messages: [
          { role: "system", content: "You write concise sponsor reads." },
          { role: "user", content: prompt },
        ],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "sponsor_statement",
            strict: true,
            schema: {
              type: "object",
              properties: {
                statement: { type: "string" },
              },
              required: ["statement"],
              additionalProperties: false,
            },
          },
        },
      }),
    });
    if (!response.ok) {
      throw new Error(`OpenAI statement failed: ${response.status}`);
    }
    const data = (await response.json()) as any;
    const raw = data?.choices?.[0]?.message?.content ?? "{}";
    const parsed = JSON.parse(raw);
    const statement = String(parsed.statement || "").trim();
    return statement || generateStatementFallback(name, productDesc);
  } catch (error) {
    console.warn("Statement generation failed, using fallback.", error);
    return generateStatementFallback(name, productDesc);
  }
};

/**
 * Ask OpenAI to fill in pros/cons/rationale for the provided slots. Returns the
 * raw slot detail array, or null when unavailable.
 */
export const enhanceSlotsWithOpenAI = async ({
  slots,
  candidates,
  mode,
  durationSeconds,
}: {
  slots: Slot[];
  candidates: ScoredCandidate[];
  mode: InsertionMode;
  durationSeconds: number | null;
}): Promise<any[] | null> => {
  if (!process.env.OPENAI_API_KEY) return null;
  const payload = {
    mode,
    duration_seconds: durationSeconds,
    slots: slots.map((slot) => ({
      insertion_ms: slot.insertion_ms,
      insertion_time_seconds: slot.insertion_time_seconds,
      silence_ms: slot.silence_ms ?? 0,
      snippet: slot.snippet ?? "",
    })),
    candidates: candidates.map((cand) => ({
      insertion_ms: cand.ms,
      silence_ms: cand.silenceMs ?? 0,
      snippet: cand.snippet ?? "",
    })),
    rules: {
      pros_count: 3,
      cons_count: 2,
      max_words_per_bullet: 7,
    },
  };

  const prompt = [
    "Generate pros/cons and rationale for each slot.",
    "Use the provided slots; do not invent new times.",
    "Pros: exactly 3 bullets, short and specific.",
    "Cons: exactly 2 bullets, short and specific.",
    "Rationale: one sentence.",
  ].join(" ");

  const response = await fetch(OPENAI_CHAT_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      temperature: 0,
      messages: [
        { role: "system", content: "You are a precise audio editor." },
        { role: "user", content: `${prompt}\n\n${JSON.stringify(payload)}` },
      ],
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "slot_details",
          strict: true,
          schema: {
            type: "object",
            properties: {
              slots: {
                type: "array",
                minItems: slots.length,
                maxItems: slots.length,
                items: {
                  type: "object",
                  properties: {
                    insertion_ms: { type: "integer" },
                    pros: {
                      type: "array",
                      minItems: 3,
                      maxItems: 3,
                      items: { type: "string" },
                    },
                    cons: {
                      type: "array",
                      minItems: 2,
                      maxItems: 2,
                      items: { type: "string" },
                    },
                    rationale: { type: "string" },
                  },
                  required: ["insertion_ms", "pros", "cons", "rationale"],
                  additionalProperties: false,
                },
              },
            },
            required: ["slots"],
            additionalProperties: false,
          },
        },
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI slot details failed: ${response.status}`);
  }

  const data = (await response.json()) as any;
  const raw = data?.choices?.[0]?.message?.content ?? "{}";
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed.slots)) return null;
  return parsed.slots;
};

/**
 * POST audio to the Whisper transcription endpoint. Returns the raw fetch
 * Response so callers can decide how to handle non-OK statuses.
 */
export const requestTranscription = (
  buffer: Buffer,
  filename: string,
): Promise<Response> => {
  const form = new FormData();
  form.append("model", "whisper-1");
  form.append("response_format", "verbose_json");
  form.append("file", new Blob([buffer]), filename);

  return fetch(OPENAI_TRANSCRIPTION_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: form,
  });
};
