import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Router } from "express";
import { upload } from "../middleware/upload";
import { requestTranscription } from "../services/openai";

export const relatedProductsRouter = Router();

relatedProductsRouter.post(
  "/api/related-products",
  upload.single("audio"),
  async (req, res) => {
    const audioFile = req.file;
    if (!audioFile) {
      res.status(400).json({ error: "audio file is required." });
      return;
    }

    if (!process.env.OPENAI_API_KEY) {
      res.status(500).json({
        error:
          "OPENAI_API_KEY not configured. Please set it in your environment.",
      });
      return;
    }

    if (!process.env.SERPAPI_KEY) {
      res.status(500).json({
        error:
          "SERPAPI_KEY not configured. Please set it in your environment.",
      });
      return;
    }

    const tempDir = await fs.promises.mkdtemp(
      path.join(os.tmpdir(), "related-products-"),
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

      const transcriptResponse = await requestTranscription(
        audioFile.buffer,
        audioFile.originalname || "audio.mp3",
      );

      if (!transcriptResponse.ok) {
        const detail = await transcriptResponse.text();
        throw new Error(
          `Transcription failed: ${transcriptResponse.status} ${detail}`,
        );
      }

      const transcript = (await transcriptResponse.json()) as any;
      const segments = Array.isArray(transcript?.segments)
        ? transcript.segments
        : [];
      const transcriptText = segments
        .map((segment: any) => String(segment.text ?? "").trim())
        .filter(Boolean)
        .join(" ")
        .slice(0, 3000);

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
                  "Extract up to 3 product search terms from the transcript. " +
                  "Focus on concrete product mentions or categories suitable for shopping searches. " +
                  "Return JSON only.",
              },
              {
                role: "user",
                content: `Transcript: ${transcriptText}`,
              },
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

      const completion = (await completionResponse.json()) as any;
      const rawContent =
        completion?.choices?.[0]?.message?.content?.trim() ?? "{}";
      let terms: string[] = [];
      try {
        const parsed = JSON.parse(rawContent);
        if (Array.isArray(parsed.products)) {
          terms = parsed.products
            .map((item: any) => (item?.term ? String(item.term) : ""))
            .filter(Boolean);
        } else if (Array.isArray(parsed.terms)) {
          terms = parsed.terms.map((item: any) => String(item)).filter(Boolean);
        }
      } catch {
        terms = [];
      }

      const uniqueTerms = [...new Set(terms)].slice(0, 3);
      const results: Array<{ term: string; items: any[] }> = [];

      for (const term of uniqueTerms) {
        const params = new URLSearchParams({
          engine: "google_shopping",
          q: term,
          api_key: process.env.SERPAPI_KEY as string,
          num: "3",
        });
        const serpResponse = await fetch(
          `https://serpapi.com/search.json?${params.toString()}`,
        );
        if (!serpResponse.ok) {
          const detail = await serpResponse.text();
          throw new Error(
            `SerpAPI search failed: ${serpResponse.status} ${detail}`,
          );
        }
        const serpJson = (await serpResponse.json()) as any;
        const items = Array.isArray(serpJson.shopping_results)
          ? serpJson.shopping_results.slice(0, 3).map((item: any) => ({
              title: item.title ?? "",
              link: item.link ?? "",
              price: item.price ?? "",
              source: item.source ?? "",
              thumbnail: item.thumbnail ?? "",
            }))
          : [];
        results.push({ term, items });
      }

      res.json({
        terms: uniqueTerms,
        results,
      });
    } catch (error) {
      res.status(500).json({
        error:
          error instanceof Error ? error.message : "Related products failed.",
      });
    } finally {
      await cleanup();
    }
  },
);
