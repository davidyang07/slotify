import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Router } from "express";
import { upload } from "../middleware/upload";
import { getAudioDuration } from "../services/ffmpeg";
import { runPythonAnalyze } from "../services/python";
import {
  generateBrandStatement,
  enhanceSlotsWithOpenAI,
  requestTranscription,
} from "../services/openai";
import {
  mergeCandidates,
  scoreCandidate,
  selectTopSlots,
  buildFallbackProsCons,
} from "../lib/candidates";
import {
  clamp,
  endsWithSentenceBoundary,
  parseJsonField,
  normalizeStatements,
} from "../lib/text";
import type {
  Candidate,
  InsertionMode,
  ScoredCandidate,
  Slot,
  SponsorStatement,
} from "../types";

export const insertSectionsRouter = Router();

insertSectionsRouter.post(
  "/api/insert-sections",
  upload.single("audio"),
  async (req, res) => {
    const audioFile = req.file;
    const count = Number.parseInt(req.body?.count ?? "5", 10);

    if (!audioFile) {
      res.status(400).json({ error: "audio file is required." });
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
      const mode: InsertionMode =
        String(req.body?.mode ?? "podcast").trim().toLowerCase() === "song"
          ? "song"
          : "podcast";
      let analysisResult: any = null;
      try {
        analysisResult = await runPythonAnalyze(audioPath, mode);
      } catch (error) {
        console.warn("Analyze CLI failed, using fallback.", error);
      }

      const durationMs = Number(analysisResult?.duration_ms ?? 0) || null;
      const durationSeconds =
        duration ?? (durationMs ? durationMs / 1000 : null);

      let transcriptCandidates: Candidate[] = [];
      if (process.env.OPENAI_API_KEY) {
        try {
          const transcriptResponse = await requestTranscription(
            audioFile.buffer,
            audioFile.originalname || "audio.mp3",
          );

          if (transcriptResponse.ok) {
            const transcript = (await transcriptResponse.json()) as any;
            const segments = Array.isArray(transcript?.segments)
              ? transcript.segments
              : [];
            transcriptCandidates = segments
              .map((segment: any, index: number) => {
                const text = String(segment.text ?? "").trim();
                if (!endsWithSentenceBoundary(text)) return null;
                const end = Number(segment.end ?? 0);
                if (!Number.isFinite(end)) return null;
                if (durationSeconds && end > durationSeconds) return null;
                const nextStart = Number(segments[index + 1]?.start ?? end);
                const gapMs = Math.max(0, (nextStart - end) * 1000);
                return {
                  ms: Math.round(end * 1000),
                  silenceMs: Math.round(gapMs),
                  snippet: text,
                };
              })
              .filter(Boolean) as Candidate[];
          }
        } catch (error) {
          console.warn("OpenAI transcript candidates failed.", error);
        }
      }

      const snippetsRaw = analysisResult?.snippets ?? {};
      const snippets: Record<number, string> = Object.fromEntries(
        Object.entries(snippetsRaw).map(([key, value]) => [
          Number.parseInt(key, 10),
          String(value ?? "").trim(),
        ]),
      );

      const candidates: Candidate[] = Array.isArray(analysisResult?.candidates)
        ? (analysisResult.candidates
            .map((entry: any) => {
              const ms = Number(
                entry.mid_ms ?? entry.ms ?? entry.time_ms ?? 0,
              );
              if (!Number.isFinite(ms) || ms < 0) return null;
              const silenceMs = Number(entry.silence_ms ?? 0);
              return {
                ms: Math.round(ms),
                silenceMs: Number.isFinite(silenceMs) ? silenceMs : 0,
                snippet: snippets[Math.round(ms)] ?? "",
              };
            })
            .filter(Boolean) as Candidate[])
        : [];
      const maxMs = durationSeconds ? durationSeconds * 1000 : null;
      const boundedCandidates =
        maxMs !== null
          ? candidates.filter((entry) => entry.ms <= maxMs)
          : candidates;
      const combinedCandidates = mergeCandidates(
        boundedCandidates,
        transcriptCandidates,
      );

      const fallbackCandidates: Candidate[] =
        durationSeconds && durationSeconds > 0
          ? [0.25, 0.5, 0.75].map((ratio) => ({
              ms: Math.round(durationSeconds * ratio * 1000),
              silenceMs: 0,
              snippet: "",
            }))
          : [
              { ms: 12000, silenceMs: 0, snippet: "" },
              { ms: 24000, silenceMs: 0, snippet: "" },
              { ms: 36000, silenceMs: 0, snippet: "" },
            ];

      const usableCandidates = combinedCandidates.length
        ? combinedCandidates
        : fallbackCandidates;
      const scoredCandidates: ScoredCandidate[] = usableCandidates.map(
        (candidate) => ({
          ...candidate,
          score: scoreCandidate(candidate, durationSeconds, mode),
        }),
      );

      console.log("Analyze candidates:", {
        count: scoredCandidates.length,
        mode,
      });

      const selected = selectTopSlots(
        scoredCandidates,
        durationSeconds,
        6,
        Number.isFinite(count) ? Math.max(3, count) : 3,
      ).slice(0, 3);

      const maxSlotMs =
        durationSeconds !== null && durationSeconds !== undefined
          ? Math.max(0, Math.round(durationSeconds * 1000) - 200)
          : null;
      const normalizedSelected = selected.map((candidate) => {
        if (maxSlotMs === null) return candidate;
        return {
          ...candidate,
          ms: Math.min(candidate.ms, maxSlotMs),
        };
      });
      const dedupedSelected: ScoredCandidate[] = [];
      const seenMs = new Set<number>();
      for (const candidate of normalizedSelected) {
        if (seenMs.has(candidate.ms)) continue;
        seenMs.add(candidate.ms);
        dedupedSelected.push(candidate);
      }

      let slots: Slot[] = dedupedSelected.map((candidate) => {
        const timeSeconds = candidate.ms / 1000;
        const clampedTimeSeconds =
          durationSeconds !== null && durationSeconds !== undefined
            ? Math.min(Math.max(0, timeSeconds), durationSeconds)
            : Math.max(0, timeSeconds);
        const clampedMs = Math.round(clampedTimeSeconds * 1000);
        const confidence = Math.round(clamp(70 + candidate.score * 25, 70, 95));
        const fallbackText = buildFallbackProsCons({
          mode,
          silenceMs: candidate.silenceMs,
          timeSeconds: clampedTimeSeconds,
          durationSeconds,
        });
        return {
          insertion_ms: clampedMs,
          insertion_time_seconds: Number(clampedTimeSeconds.toFixed(3)),
          confidence_percent: confidence,
          pros: fallbackText.pros,
          cons: fallbackText.cons,
          rationale: fallbackText.rationale,
          silence_ms: candidate.silenceMs,
          snippet: candidate.snippet ?? "",
        };
      });

      try {
        const openAiDetails = await enhanceSlotsWithOpenAI({
          slots,
          candidates: scoredCandidates,
          mode,
          durationSeconds,
        });
        if (openAiDetails) {
          slots = slots.map((slot) => {
            const match = openAiDetails.find(
              (entry) => Number(entry.insertion_ms) === slot.insertion_ms,
            );
            if (!match) return slot;
            return {
              ...slot,
              pros:
                Array.isArray(match.pros) && match.pros.length === 3
                  ? match.pros
                  : slot.pros,
              cons:
                Array.isArray(match.cons) && match.cons.length === 2
                  ? match.cons
                  : slot.cons,
              rationale:
                typeof match.rationale === "string" && match.rationale.trim()
                  ? match.rationale.trim()
                  : slot.rationale,
            };
          });
        }
      } catch (error) {
        console.warn("OpenAI slot details failed, using fallback.", error);
      }

      slots.sort((a, b) => b.confidence_percent - a.confidence_percent);

      console.log(
        "Analyze top slots:",
        slots.map((slot) => ({
          insertion_ms: slot.insertion_ms,
          confidence_percent: slot.confidence_percent,
        })),
      );

      const sponsorsField = parseJsonField(req.body?.sponsors);
      const statementsField = req.body?.statements ?? req.body?.statement;
      let sponsorStatements: SponsorStatement[] = [];
      if (Array.isArray(sponsorsField)) {
        sponsorStatements = await Promise.all(
          sponsorsField.map(async (entry: any, index: number) => {
            const name = String(entry?.name ?? entry?.brand ?? "").trim();
            const productDesc = String(entry?.productDesc ?? "").trim();
            const rawStatement = String(entry?.statement ?? "").trim();
            const statement =
              rawStatement ||
              (await generateBrandStatement({ name, productDesc }));
            return {
              id: entry?.id ?? `sponsor-${index + 1}`,
              name,
              statement,
              generated: !rawStatement,
            };
          }),
        );
      } else {
        const statements = normalizeStatements(statementsField);
        sponsorStatements = statements.map((statement, index) => ({
          id: `sponsor-${index + 1}`,
          name: "",
          statement,
          generated: false,
        }));
      }

      const points = slots.map((slot) => slot.insertion_time_seconds);
      const confidences = slots.map((slot) => slot.confidence_percent);

      res.json({
        points,
        confidences,
        duration: durationSeconds,
        source: analysisResult ? "heuristic" : "fallback",
        slots,
        sponsorStatements,
      });
    } catch (error) {
      res.status(500).json({
        error:
          error instanceof Error ? error.message : "Insert analysis failed.",
      });
    } finally {
      await cleanup();
    }
  },
);
