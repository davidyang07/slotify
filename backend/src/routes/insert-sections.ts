/**
 * POST /api/insert-sections -- find and rank ad-break candidates in an upload.
 *
 * Three outcomes are reported as three different things, because they mean
 * three different things to a user:
 *
 *   200 { placementStatus: "ok" }             analysis ran, points were found
 *   200 { placementStatus: "no_candidates" }  analysis ran, nothing qualified
 *   502 { placementStatus: "unavailable" }    analysis itself failed
 *
 * The route previously collapsed all three: when the Python analyser threw, it
 * logged a warning and substituted candidates at 25 %, 50 % and 75 % of the
 * duration, so a total analysis failure was indistinguishable in the response
 * from a successful run. Those fallbacks are gone. Nothing in this file
 * invents a timestamp.
 */

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
import { mergeCandidates } from "../lib/candidates";
import { rankWithHeuristic } from "../lib/ranking";
import { HEURISTIC_V1 } from "../lib/heuristic-config";
import {
  buildPlacementSignals,
  buildRationale,
  clampSlotMs,
  dedupeByMs,
  toPlacementScores,
} from "../lib/placement";
import {
  endsWithSentenceBoundary,
  parseJsonField,
  normalizeStatements,
} from "../lib/text";
import type {
  Candidate,
  InsertionMode,
  RankerProvenance,
  Slot,
  SponsorStatement,
} from "../types";

export const insertSectionsRouter = Router();

const CANDIDATE_GENERATION = "signal_offline_v1";
const MAX_SLOTS = HEURISTIC_V1.selection.max_returned;
const MIN_SEPARATION_SECONDS = HEURISTIC_V1.selection.min_separation_seconds;

/** Build the sponsor statements the client asked for, if any. */
const buildSponsorStatements = async (body: any): Promise<SponsorStatement[]> => {
  const sponsorsField = parseJsonField(body?.sponsors);
  if (Array.isArray(sponsorsField)) {
    return Promise.all(
      sponsorsField.map(async (entry: any, index: number) => {
        const name = String(entry?.name ?? entry?.brand ?? "").trim();
        const productDesc = String(entry?.productDesc ?? "").trim();
        const rawStatement = String(entry?.statement ?? "").trim();
        const statement =
          rawStatement || (await generateBrandStatement({ name, productDesc }));
        return {
          id: entry?.id ?? `sponsor-${index + 1}`,
          name,
          statement,
          generated: !rawStatement,
        };
      }),
    );
  }
  const statements = normalizeStatements(body?.statements ?? body?.statement);
  return statements.map((statement, index) => ({
    id: `sponsor-${index + 1}`,
    name: "",
    statement,
    generated: false,
  }));
};

/**
 * Optional extra candidates from OpenAI's transcription. Purely additive: a
 * failure here narrows the candidate pool, it never fabricates one.
 */
const transcriptCandidates = async (
  buffer: Buffer,
  filename: string,
  durationSeconds: number | null,
): Promise<Candidate[]> => {
  if (!process.env.OPENAI_API_KEY) return [];
  try {
    const response = await requestTranscription(buffer, filename);
    if (!response.ok) return [];
    const transcript = (await response.json()) as any;
    const segments = Array.isArray(transcript?.segments) ? transcript.segments : [];
    return segments
      .map((segment: any, index: number) => {
        const text = String(segment.text ?? "").trim();
        if (!endsWithSentenceBoundary(text)) return null;
        const end = Number(segment.end ?? 0);
        if (!Number.isFinite(end)) return null;
        if (durationSeconds && end > durationSeconds) return null;
        const nextStart = Number(segments[index + 1]?.start ?? end);
        return {
          ms: Math.round(end * 1000),
          silenceMs: Math.round(Math.max(0, (nextStart - end) * 1000)),
          snippet: text,
        };
      })
      .filter(Boolean) as Candidate[];
  } catch (error) {
    console.warn("OpenAI transcript candidates unavailable.", error);
    return [];
  }
};

insertSectionsRouter.post(
  "/api/insert-sections",
  upload.single("audio"),
  async (req, res) => {
    const audioFile = req.file;
    if (!audioFile) {
      res.status(400).json({
        placementStatus: "unavailable",
        error: "audio file is required.",
        slots: [],
        points: [],
      });
      return;
    }

    const requestedCount = Number.parseInt(req.body?.count ?? String(MAX_SLOTS), 10);
    const count = Number.isFinite(requestedCount)
      ? Math.min(MAX_SLOTS, Math.max(1, requestedCount))
      : MAX_SLOTS;
    const mode: InsertionMode =
      String(req.body?.mode ?? "podcast").trim().toLowerCase() === "song"
        ? "song"
        : "podcast";

    const tempDir = await fs.promises.mkdtemp(
      path.join(os.tmpdir(), "insert-sections-"),
    );
    const audioPath = path.join(
      tempDir,
      `${Date.now()}-${path.basename(audioFile.originalname || "audio.mp3")}`,
    );
    const cleanup = async () => {
      await fs.promises.rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
    };

    try {
      await fs.promises.writeFile(audioPath, audioFile.buffer);
      const probedDuration = await getAudioDuration(audioPath).catch(() => null);

      let analysis: any;
      try {
        analysis = await runPythonAnalyze(audioPath, mode);
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        console.error("Audio analysis failed.", detail);
        res.status(502).json({
          placementStatus: "unavailable",
          slots: [],
          points: [],
          duration: probedDuration,
          provenance: {
            candidateGeneration: CANDIDATE_GENERATION,
            source: null,
            mode: "heuristic",
            modelVariant: null,
            modelRunId: null,
            scoreScale: null,
            isCalibratedProbability: false,
          },
          error: "Audio analysis failed; no insertion points can be reported.",
          detail,
        });
        return;
      }

      const analysedDurationMs = Number(analysis?.duration_ms ?? 0) || null;
      const durationSeconds =
        probedDuration ?? (analysedDurationMs ? analysedDurationMs / 1000 : null);

      const snippets: Record<number, string> = Object.fromEntries(
        Object.entries(analysis?.snippets ?? {}).map(([key, value]) => [
          Number.parseInt(key, 10),
          String(value ?? "").trim(),
        ]),
      );

      const signalCandidates: Candidate[] = Array.isArray(analysis?.candidates)
        ? (analysis.candidates
            .map((entry: any) => {
              const ms = Number(entry.mid_ms ?? entry.ms ?? entry.time_ms ?? 0);
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
      const bounded =
        maxMs !== null
          ? signalCandidates.filter((entry) => entry.ms <= maxMs)
          : signalCandidates;
      const extra = await transcriptCandidates(
        audioFile.buffer,
        audioFile.originalname || "audio.mp3",
        durationSeconds,
      );
      const candidates = mergeCandidates(bounded, extra);

      const sponsorStatements = await buildSponsorStatements(req.body);

      if (candidates.length === 0) {
        res.json({
          placementStatus: "no_candidates",
          slots: [],
          points: [],
          duration: durationSeconds,
          candidateCount: 0,
          provenance: {
            candidateGeneration: CANDIDATE_GENERATION,
            source: null,
            mode: "heuristic",
            modelVariant: null,
            modelRunId: null,
            scoreScale: null,
            isCalibratedProbability: false,
          },
          sponsorStatements,
          warning:
            "Analysis completed but found no eligible insertion point in this audio.",
        });
        return;
      }

      const ranking = rankWithHeuristic({
        candidates,
        episodeId: "upload",
        durationSeconds,
        mode,
        minSeparationSeconds: MIN_SEPARATION_SECONDS,
        count,
      });

      const clamped = ranking.ranked.map((entry) => ({
        ...entry,
        ms: clampSlotMs(entry.ms, durationSeconds),
      }));
      const unique = dedupeByMs(clamped);
      const placementScores = toPlacementScores(
        unique.map((entry) => entry.rawScore),
        ranking.scoreScale,
      );

      let slots: Slot[] = unique.map((entry, index) => {
        const timeSeconds = entry.ms / 1000;
        const signals = buildPlacementSignals({
          mode,
          silenceMs: entry.silenceMs,
          snippet: entry.snippet,
          timeSeconds,
          durationSeconds,
        });
        return {
          candidate_id: entry.candidateId,
          insertion_ms: entry.ms,
          insertion_time_seconds: Number(timeSeconds.toFixed(3)),
          placement_score: placementScores[index],
          raw_score: entry.rawScore,
          rank: index + 1,
          source: ranking.source,
          signals,
          rationale: buildRationale(signals, timeSeconds),
          rationale_source: "measured_signals" as const,
          silence_ms: entry.silenceMs,
          snippet: entry.snippet ?? "",
        };
      });

      // Optional narrative enrichment. It may rewrite the rationale text; it may
      // never add, move or remove a slot, and the response says when it spoke.
      try {
        const details = await enhanceSlotsWithOpenAI({
          slots,
          candidates: candidates.map((candidate) => ({ ...candidate, score: 0 })),
          mode,
          durationSeconds,
        });
        if (details) {
          slots = slots.map((slot) => {
            const match = details.find(
              (entry) => Number(entry.insertion_ms) === slot.insertion_ms,
            );
            const rationale =
              typeof match?.rationale === "string" && match.rationale.trim()
                ? match.rationale.trim()
                : null;
            if (!rationale) return slot;
            return { ...slot, rationale, rationale_source: "openai" as const };
          });
        }
      } catch (error) {
        console.warn("OpenAI slot narration unavailable; keeping measured rationale.", error);
      }

      const provenance: RankerProvenance = {
        candidateGeneration: CANDIDATE_GENERATION,
        source: ranking.source,
        mode: ranking.mode,
        modelVariant: ranking.modelVariant,
        modelRunId: ranking.modelRunId,
        scoreScale: ranking.scoreScale,
        isCalibratedProbability: false,
      };

      res.json({
        placementStatus: "ok",
        slots,
        points: slots.map((slot) => slot.insertion_time_seconds),
        placementScores: slots.map((slot) => slot.placement_score),
        duration: durationSeconds,
        candidateCount: candidates.length,
        provenance,
        sponsorStatements,
        warning: ranking.warnings.length ? ranking.warnings.join(" ") : null,
      });
    } catch (error) {
      res.status(500).json({
        placementStatus: "unavailable",
        slots: [],
        points: [],
        error: error instanceof Error ? error.message : "Insert analysis failed.",
      });
    } finally {
      await cleanup();
    }
  },
);
