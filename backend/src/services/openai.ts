import type { InsertionMode, ScoredCandidate, Slot } from "../types";
import {
  LANGUAGE_NAMES,
  TONE_DIRECTIONS,
  type SponsorCopyOptions,
  type SponsorLanguage,
} from "../lib/sponsor-options";

const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions";
const OPENAI_TRANSCRIPTION_URL =
  "https://api.openai.com/v1/audio/transcriptions";

// Without OpenAI the server still writes the read in the requested language;
// tone needs the model.
const FALLBACK_TEMPLATES: Record<
  SponsorLanguage,
  { brand: string; desc: string; render: (brand: string, desc: string) => string }
> = {
  en: {
    brand: "Our sponsor",
    desc: "a thoughtful companion for your day",
    render: (brand, desc) =>
      `${brand} supports this episode with ${desc}, offering a simple way to stay focused and refreshed.`,
  },
  es: {
    brand: "nuestro patrocinador",
    desc: "un compañero atento para tu día",
    render: (brand, desc) => `Este episodio cuenta con el apoyo de ${brand}: ${desc}.`,
  },
  fr: {
    brand: "notre partenaire",
    desc: "un compagnon attentionné pour votre journée",
    render: (brand, desc) => `Cet épisode vous est proposé avec le soutien de ${brand} : ${desc}.`,
  },
  de: {
    brand: "unserem Sponsor",
    desc: "ein aufmerksamer Begleiter für deinen Tag",
    render: (brand, desc) => `Diese Folge wird unterstützt von ${brand}: ${desc}.`,
  },
  it: {
    brand: "il nostro sponsor",
    desc: "un compagno attento per la tua giornata",
    render: (brand, desc) => `Questo episodio è offerto da ${brand}: ${desc}.`,
  },
  pt: {
    brand: "nosso patrocinador",
    desc: "um companheiro atencioso para o seu dia",
    render: (brand, desc) => `Este episódio tem o apoio de ${brand}: ${desc}.`,
  },
  ja: {
    brand: "スポンサー",
    desc: "毎日に寄り添うパートナー",
    render: (brand, desc) => `このエピソードは${brand}の提供でお送りします。${desc}。`,
  },
  ko: {
    brand: "저희 후원사",
    desc: "하루를 함께하는 든든한 동반자",
    render: (brand, desc) => `이 에피소드는 ${brand}의 후원으로 제공됩니다. ${desc}.`,
  },
  zh: {
    brand: "我们的赞助商",
    desc: "陪伴你每一天的贴心伙伴",
    render: (brand, desc) => `本期节目由${brand}赞助播出。${desc}。`,
  },
};

export const generateStatementFallback = (
  name: string,
  productDesc: string,
  language: SponsorLanguage = "en",
): string => {
  const template = FALLBACK_TEMPLATES[language];
  return template.render(name || template.brand, productDesc || template.desc);
};

/** The copywriting prompt; exported so the tone/language wiring is testable. */
export const buildStatementPrompt = ({
  name,
  productDesc,
  tone,
  language,
}: { name: string; productDesc: string } & SponsorCopyOptions): string =>
  [
    "Write one sponsor read sentence (8-12 seconds when spoken).",
    tone
      ? `Sound native to the episode; the tone is ${TONE_DIRECTIONS[tone]}.`
      : "Sound native to the episode, calm and conversational.",
    "Avoid hypey marketing language and emojis.",
    language && language !== "en"
      ? `Write the sentence entirely in ${LANGUAGE_NAMES[language]}; keep the brand name as given.`
      : "",
    `Brand name: ${name || "Sponsor"}.`,
    productDesc ? `Product description: ${productDesc}.` : "",
  ]
    .filter(Boolean)
    .join(" ");

/** Generate a single sponsor-read sentence, falling back to a template. */
export const generateBrandStatement = async ({
  name,
  productDesc,
  tone,
  language,
}: {
  name: string;
  productDesc: string;
} & SponsorCopyOptions): Promise<string> => {
  if (!process.env.OPENAI_API_KEY) {
    return generateStatementFallback(name, productDesc, language);
  }

  const prompt = buildStatementPrompt({ name, productDesc, tone, language });

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
    const statement = typeof parsed?.statement === "string" ? parsed.statement.trim() : "";
    return statement || generateStatementFallback(name, productDesc, language);
  } catch (error) {
    console.warn("Statement generation failed, using fallback.", error);
    return generateStatementFallback(name, productDesc, language);
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
