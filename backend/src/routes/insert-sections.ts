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
import { Router } from "express";
import { claimWorkDir, inspectUploads, upload } from "../middleware/upload";
import { maxSongSeconds } from "../config";
import { runPythonAnalyze } from "../services/python";
import {
  generateBrandStatement,
  enhanceSlotsWithOpenAI,
  requestTranscription,
} from "../services/openai";
import { mergeCandidates } from "../lib/candidates";
import { rankCandidates } from "../services/ranker";
import { resolveRankerMode } from "../lib/ranker-mode";
import { HEURISTIC_V1 } from "../lib/heuristic-config";
import {
  buildPlacementSignals,
  buildRationale,
  clampSlotMs,
  dedupeByMs,
} from "../lib/placement";
import { asText, endsWithSentenceBoundary, parseJsonField } from "../lib/text";
import { leaseHeavyJob, refuseWhenBusy } from "../lib/concurrency";
import { logFailure, sendFailure } from "../lib/errors";
import { asyncHandler, clientAbortSignal, HttpError } from "../lib/http";
import {
  fieldOf,
  optionalEnum,
  optionalNumber,
  optionalStatements,
  optionalString,
} from "../lib/request-fields";
import {
  SPONSOR_LANGUAGES,
  SPONSOR_TONES,
  type SponsorCopyOptions,
} from "../lib/sponsor-options";
import { removeTempDir } from "../lib/temp";
import { SubprocessError } from "../lib/subprocess";
import type {
  Candidate,
  InsertionMode,
  RankerProvenance,
  Slot,
  SponsorStatement,
} from "../types";

export const insertSectionsRouter = Router();

const CANDIDATE_GENERATION = "signal_offline_v1";

/**
 * The configured ranker, for the degraded responses. Reported even when nothing
 * was ranked, so a client can tell a heuristic deployment's failure from a
 * learned one's.
 */
const currentRankerMode = (): string => {
  try {
    return resolveRankerMode().mode;
  } catch {
    return "unavailable";
  }
};
const MAX_SLOTS = HEURISTIC_V1.selection.max_returned;
const MIN_SEPARATION_SECONDS = HEURISTIC_V1.selection.min_separation_seconds;
/** The UI's maximum. Each sponsor without a statement is one OpenAI call. */
const MAX_SPONSORS = 3;

interface SponsorRequest {
  id: string;
  name: string;
  productDesc: string;
  /** Empty when the server is to write it. */
  statement: string;
}

/**
 * Validate the sponsors the client asked for, before any analysis runs. The
 * `sponsors` field is a JSON array (multipart carries it as a string);
 * otherwise `statements`/`statement` are taken as given.
 */
const parseSponsorRequests = (body: unknown): SponsorRequest[] => {
  const raw = fieldOf(body, "sponsors");
  if (raw !== undefined && raw !== null && raw !== "") {
    const sponsors = typeof raw === "string" ? parseJsonField(raw) : raw;
    if (!Array.isArray(sponsors)) {
      throw new HttpError(400, "sponsors must be a JSON array.");
    }
    if (sponsors.length > MAX_SPONSORS) {
      throw new HttpError(400, `At most ${MAX_SPONSORS} sponsors per request.`);
    }
    return sponsors.map((entry: unknown, index: number) => {
      if (entry === null || typeof entry !== "object" || Array.isArray(entry)) {
        throw new HttpError(400, `sponsors[${index}] must be an object.`);
      }
      return {
        id: optionalString(entry, "id", 100) ?? `sponsor-${index + 1}`,
        name: optionalString(entry, "name", 200) ?? optionalString(entry, "brand", 200) ?? "",
        productDesc: optionalString(entry, "productDesc") ?? "",
        statement: optionalString(entry, "statement") ?? "",
      };
    });
  }
  const statements =
    optionalStatements(body, "statements") ?? optionalStatements(body, "statement") ?? [];
  if (statements.length > MAX_SPONSORS) {
    throw new HttpError(400, `At most ${MAX_SPONSORS} sponsor statements per request.`);
  }
  return statements.map((statement, index) => ({
    id: `sponsor-${index + 1}`,
    name: "",
    productDesc: "",
    statement,
  }));
};

/**
 * Fill in the statements the server writes. Tone and language shape only
 * those; a statement the user supplied is used verbatim.
 */
const buildSponsorStatements = (
  sponsors: SponsorRequest[],
  options: SponsorCopyOptions,
): Promise<SponsorStatement[]> =>
  Promise.all(
    sponsors.map(async (sponsor) => ({
      id: sponsor.id,
      name: sponsor.name,
      statement:
        sponsor.statement ||
        (await generateBrandStatement({
          name: sponsor.name,
          productDesc: sponsor.productDesc,
          ...options,
        })),
      generated: !sponsor.statement,
    })),
  );

/** Provenance for a response in which nothing was ranked. */
const unrankedProvenance = () => ({
  candidateGeneration: CANDIDATE_GENERATION,
  source: null,
  mode: currentRankerMode(),
  modelVariant: null,
  modelRunId: null,
  trainingLabelSource: null,
  scoreScale: null,
  isCalibratedProbability: false,
});

/**
 * Optional extra candidates from OpenAI's transcription. Purely additive: a
 * failure here narrows the candidate pool, it never fabricates one.
 */
const transcriptCandidates = async (
  audioPath: string,
  filename: string,
  durationSeconds: number | null,
): Promise<Candidate[]> => {
  if (!process.env.OPENAI_API_KEY) return [];
  try {
    // A file-backed Blob: the upload is streamed to OpenAI, never buffered.
    const response = await requestTranscription(await fs.openAsBlob(audioPath), filename);
    if (!response.ok) return [];
    const transcript = (await response.json()) as any;
    const segments = Array.isArray(transcript?.segments) ? transcript.segments : [];
    return segments
      .map((segment: any, index: number) => {
        const text = asText(segment?.text).trim();
        if (!endsWithSentenceBoundary(text)) return null;
        const end = Number(segment.end ?? 0);
        if (!Number.isFinite(end)) return null;
        if (durationSeconds && end > durationSeconds) return null;
        const nextStart = Number(segments[index + 1]?.start ?? end);
        return {
          ms: Math.round(end * 1000),
          silenceMs: Math.round(Math.max(0, (nextStart - end) * 1000)),
          snippet: text,
          generatedBy: "transcript" as const,
        };
      })
      .filter(Boolean) as Candidate[];
  } catch (error) {
    console.warn("OpenAI transcript candidates unavailable.", error);
    return [];
  }
};

const UNAVAILABLE = { placementStatus: "unavailable", slots: [], points: [] } as const;

/**
 * Song mode's own cap (SLOTIFY_MAX_SONG_MINUTES): beat tracking holds the
 * decoded song in memory. A song whose length cannot be read cannot be
 * checked, so it is refused rather than analysed blind.
 */
const checkSongLength = (seconds: number | null): void => {
  const limitMinutes = Math.round((maxSongSeconds / 60) * 10) / 10;
  if (seconds === null) {
    throw new HttpError(
      422,
      "Song analysis needs a file whose duration can be read; convert it to mp3 or wav and try again.",
    );
  }
  if (seconds > maxSongSeconds) {
    throw new HttpError(
      413,
      `Song analysis is limited to ${limitMinutes} minutes (see SLOTIFY_MAX_SONG_MINUTES); ` +
        `this upload is ${Math.round((seconds / 60) * 10) / 10} minutes.`,
    );
  }
};

insertSectionsRouter.post(
  "/api/insert-sections",
  refuseWhenBusy({ extra: UNAVAILABLE }),
  upload.single("audio", { dirPrefix: "insert-sections-" }),
  asyncHandler(async (req, res) => {
    const rejected = (status: number, error: string) =>
      res.status(status).json({ ...UNAVAILABLE, error });

    const audioFile = req.file;
    if (!audioFile) {
      rejected(400, "audio file is required.");
      return;
    }

    let count: number;
    let mode: InsertionMode;
    let sponsors: SponsorRequest[];
    let copyOptions: SponsorCopyOptions;
    try {
      const requestedCount = optionalNumber(req.body, "count");
      count =
        requestedCount !== undefined
          ? Math.min(MAX_SLOTS, Math.max(1, Math.trunc(requestedCount)))
          : MAX_SLOTS;
      mode = optionalEnum(req.body, "mode", ["podcast", "song"] as const) ?? "podcast";
      copyOptions = {
        tone: optionalEnum(req.body, "tone", SPONSOR_TONES),
        language: optionalEnum(req.body, "language", SPONSOR_LANGUAGES),
      };
      sponsors = parseSponsorRequests(req.body);
    } catch (error) {
      if (!(error instanceof HttpError)) throw error;
      rejected(error.status, error.message);
      return;
    }

    const signal = clientAbortSignal(req, res);
    let tempDir: string | null = null;
    let releaseJob: (() => void) | null = null;
    try {
      // The upload is already in this request's own dir, under a fixed name;
      // the analyser works there too and the whole dir goes in `finally`.
      tempDir = await claimWorkDir(req, "insert-sections-");
      const [audio] = await inspectUploads(req, { signal });
      const audioPath = audio.path;
      const probedDuration = audio.probe.durationSeconds;
      if (mode === "song") checkSongLength(probedDuration);

      // One slot from analysis through ranking: both decode the whole upload.
      releaseJob = await leaseHeavyJob(req, { signal });

      let analysis: any;
      try {
        analysis = await runPythonAnalyze(audioPath, mode, { signal, workDir: tempDir });
      } catch (error) {
        // Stopped for a shutdown: a 503 to retry, not a failed analysis.
        if (error instanceof SubprocessError && error.reason === "shutdown") throw error;
        // The analyser's stderr is a Python traceback. The whole thing goes to
        // the server log; the client gets the final line, which is the actual
        // error, without the repository's internal paths.
        const detail = logFailure("Audio analysis failed.", error);
        if (signal.aborted) return;
        res.status(502).json({
          placementStatus: "unavailable",
          slots: [],
          points: [],
          duration: probedDuration,
          provenance: unrankedProvenance(),
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
          asText(value).trim(),
        ]),
      );
      // Song candidates are beat-snapped only when the analyser found beats.
      const beatsDetected =
        Array.isArray(analysis?.beat_times_ms) && analysis.beat_times_ms.length > 0;

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
                generatedBy: "analyser" as const,
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
        audioPath,
        `audio.${audio.probe.extension}`,
        durationSeconds,
      );
      const candidates = mergeCandidates(bounded, extra);

      const sponsorStatements = await buildSponsorStatements(sponsors, copyOptions);

      if (candidates.length === 0) {
        res.json({
          placementStatus: "no_candidates",
          slots: [],
          points: [],
          duration: durationSeconds,
          candidateCount: 0,
          provenance: unrankedProvenance(),
          sponsorStatements,
          warning:
            "Analysis completed but found no eligible insertion point in this audio.",
        });
        return;
      }

      const ranking = await rankCandidates({
        candidates,
        episodeId: "upload",
        audioPath,
        durationSeconds,
        mode,
        minSeparationSeconds: MIN_SEPARATION_SECONDS,
        count,
        signal,
      });

      releaseJob();

      const clamped = ranking.ranked.map((entry) => ({
        ...entry,
        ms: clampSlotMs(entry.ms, durationSeconds),
      }));
      const unique = dedupeByMs(clamped);

      let slots: Slot[] = unique.map((entry, index) => {
        const timeSeconds = entry.ms / 1000;
        const signals = buildPlacementSignals({
          mode,
          silenceMs: entry.silenceMs,
          snippet: entry.snippet,
          timeSeconds,
          durationSeconds,
          transcriptAvailable: entry.textAvailable,
          generatedBy: entry.generatedBy,
          beatsDetected,
        });
        return {
          candidate_id: entry.candidateId,
          insertion_ms: entry.ms,
          insertion_time_seconds: Number(timeSeconds.toFixed(3)),
          placement_score: entry.normalizedScore,
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
        trainingLabelSource: ranking.trainingLabelSource,
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
      sendFailure(res, "Insert analysis failed.", error, { signal, extra: UNAVAILABLE });
    } finally {
      releaseJob?.();
      await removeTempDir(tempDir);
    }
  }),
);
