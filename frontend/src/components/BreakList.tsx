import type { FormEvent } from "react";
import { useId, useState } from "react";
import {
  describeScoreScale,
  describeSource,
  describeTrainingLabels,
  type PlacementResult,
  type Slot,
} from "../lib/recommendations";
import { parseBreakTime, type GenerationNote, type Sponsor } from "../lib/workspace";
import BarsAnimation from "./BarsAnimation";
import BreakRow from "./BreakRow";
import { AlertIcon } from "./icons";
import "./BreakList.css";

export type AnalysisNotice = {
  /** `empty`: analysis ran and found nothing; `error`: it did not finish. */
  kind: "error" | "cancelled" | "empty";
  text: string;
};

type BreakListProps = {
  placement: PlacementResult | null;
  slots: Slot[];
  selectedIds: string[];
  isAnalyzing: boolean;
  analysisNotice: AnalysisNotice | null;
  authHint: string | null;
  onCancelAnalysis: () => void;
  onAnalyzeAgain: () => void;
  onToggle: (slotId: string) => void;
  sponsors: Sponsor[];
  assignments: Record<string, string>;
  onAssign: (slotId: string, sponsorId: string) => void;
  onPreview: (slotId: string) => void;
  previewBlocked: string | null;
  generationBusy: boolean;
  previewingId: string | null;
  preview: { slotId: string; url: string } | null;
  note: GenerationNote | null;
  durationSeconds: number | null;
  onAddBreak: (seconds: number) => void;
};

/** "Add a break at mm:ss": two small mono inputs and Add. */
const AddBreakRow = ({
  durationSeconds,
  onAdd,
}: {
  durationSeconds: number | null;
  onAdd: (seconds: number) => void;
}) => {
  const [minutes, setMinutes] = useState("");
  const [seconds, setSeconds] = useState("");
  const [error, setError] = useState("");
  const uid = useId();

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const result = parseBreakTime(minutes, seconds, durationSeconds);
    if ("error" in result) {
      setError(result.error);
      return;
    }
    onAdd(result.seconds);
    setMinutes("");
    setSeconds("");
    setError("");
  };

  const digits = (value: string) => value.replace(/[^\d]/g, "");

  return (
    <form className="break-add" onSubmit={submit} aria-labelledby={`${uid}-label`}>
      <span className="break-add-label" id={`${uid}-label`}>
        Add a break at
      </span>
      <span className="break-add-time">
        <input
          type="text"
          inputMode="numeric"
          className="break-add-input"
          placeholder="mm"
          maxLength={4}
          value={minutes}
          onChange={(event) => setMinutes(digits(event.target.value))}
          aria-label="Minutes"
          aria-describedby={error ? `${uid}-error` : undefined}
        />
        <span className="break-add-colon" aria-hidden="true">
          :
        </span>
        <input
          type="text"
          inputMode="numeric"
          className="break-add-input"
          placeholder="ss"
          maxLength={2}
          value={seconds}
          onChange={(event) => setSeconds(digits(event.target.value))}
          aria-label="Seconds"
          aria-describedby={error ? `${uid}-error` : undefined}
        />
      </span>
      <button type="submit" className="secondary" disabled={!minutes && !seconds}>
        Add
      </button>
      {error && (
        <p className="status-error break-add-error" id={`${uid}-error`} role="alert">
          {error}
        </p>
      )}
    </form>
  );
};

/**
 * DESIGN.md `break-list`: a level-1 panel. The header says once what ranked
 * the breaks (the provenance chip) and, when it applies, that the ranker was
 * trained on weak labels -- that notice is never hidden. The body is the
 * inline progress while analysis runs, then one row per break in rank order,
 * then "Add a break at mm:ss".
 */
const BreakList = ({
  placement,
  slots,
  selectedIds,
  isAnalyzing,
  analysisNotice,
  authHint,
  onCancelAnalysis,
  onAnalyzeAgain,
  onToggle,
  sponsors,
  assignments,
  onAssign,
  onPreview,
  previewBlocked,
  generationBusy,
  previewingId,
  preview,
  note,
  durationSeconds,
  onAddBreak,
}: BreakListProps) => {
  const provenance = placement?.provenance ?? null;
  const trainingNote = describeTrainingLabels(provenance);
  const rankedCount = placement?.slots.length ?? 0;
  const scale = describeScoreScale(provenance);
  const meta =
    !isAnalyzing && placement && placement.candidateCount !== null
      ? `${placement.candidateCount} candidate${placement.candidateCount === 1 ? "" : "s"} considered, ${rankedCount} shown.${scale ? ` ${scale}` : ""}`
      : !isAnalyzing && scale
        ? scale
        : null;

  return (
    <section className="break-list" aria-labelledby="break-list-title" aria-busy={isAnalyzing}>
      <header className="break-list-header">
        <h2 className="break-list-title" id="break-list-title">
          Break points
        </h2>
        {meta && <p className="break-list-meta">{meta}</p>}
        {provenance && !isAnalyzing && (
          <div className="break-list-provenance">
            <span className="provenance-chip">{describeSource(provenance)}</span>
            {trainingNote && (
              <p className="weak-label-notice">
                <AlertIcon size={14} />
                <span>{trainingNote}</span>
              </p>
            )}
          </div>
        )}
      </header>

      {isAnalyzing ? (
        <div className="inline-progress" role="status">
          <BarsAnimation variant="inline" />
          <p className="inline-progress-text">
            Finding break points. This runs on the server and needs no API keys.
          </p>
          <button type="button" className="secondary" onClick={onCancelAnalysis}>
            Cancel
          </button>
        </div>
      ) : (
        <>
          {rankedCount === 0 && (
            <div className="break-list-message" role={analysisNotice?.kind === "error" ? "alert" : undefined}>
              {analysisNotice?.kind === "error" ? (
                <>
                  <p className="status-error">{analysisNotice.text}</p>
                  {authHint && <p className="status-error">{authHint}</p>}
                </>
              ) : analysisNotice?.kind === "cancelled" ? (
                <p className="break-list-message-text">{analysisNotice.text}</p>
              ) : (
                <>
                  <p className="break-list-message-title">No break points found.</p>
                  <p className="break-list-message-text">
                    {analysisNotice?.text ??
                      "Analysis has not run on this file yet."}{" "}
                    Nothing is filled in: you can add a break by hand below.
                  </p>
                </>
              )}
              <button type="button" className="secondary" onClick={onAnalyzeAgain}>
                Analyze again
              </button>
            </div>
          )}
          {slots.length > 0 && (
            <ul className="break-rows" aria-label="Break points, in rank order">
              {slots.map((slot) => (
                <BreakRow
                  key={slot.id}
                  slot={slot}
                  selected={selectedIds.includes(slot.id)}
                  sponsors={sponsors}
                  sponsorId={assignments[slot.id] ?? sponsors[0]?.id ?? ""}
                  onAssign={onAssign}
                  onToggle={onToggle}
                  onPreview={onPreview}
                  previewBlocked={previewBlocked}
                  previewBusy={generationBusy}
                  previewing={previewingId === slot.id}
                  previewUrl={preview?.slotId === slot.id ? preview.url : null}
                  note={note?.mode === "preview" && note.slotId === slot.id ? note : null}
                />
              ))}
            </ul>
          )}
          <AddBreakRow durationSeconds={durationSeconds} onAdd={onAddBreak} />
        </>
      )}
    </section>
  );
};

export default BreakList;
