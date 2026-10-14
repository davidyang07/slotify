import type { MouseEvent } from "react";
import { useId, useState } from "react";
import { formatTimestamp } from "../lib/format";
import type { Slot } from "../lib/recommendations";
import { reasonTags, type GenerationNote, type Sponsor } from "../lib/workspace";
import { AlertIcon, CheckIcon, ChevronIcon, PlayIcon } from "./icons";

export type BreakRowProps = {
  slot: Slot;
  selected: boolean;
  sponsors: Sponsor[];
  /** The sponsor this break reads; only offered when there is more than one. */
  sponsorId: string;
  onAssign: (slotId: string, sponsorId: string) => void;
  onToggle: (slotId: string) => void;
  onPreview: (slotId: string) => void;
  /** Why previews cannot run at all on this server, or null. */
  previewBlocked: string | null;
  /** Another preview or a render is running. */
  previewBusy: boolean;
  /** This row's preview is being generated. */
  previewing: boolean;
  /** This row's last preview, played inline. */
  previewUrl: string | null;
  /** What this row's last preview attempt said. */
  note: GenerationNote | null;
};

const sponsorLabel = (sponsor: Sponsor, index: number) =>
  sponsor.name.trim() || `Sponsor ${index + 1}`;

/**
 * DESIGN.md `break-row`: checkbox · timestamp · score + bar · reason tags ·
 * preview, with a chevron for the full reasoning. Clicking anywhere on the row
 * that is not a control toggles it; the checkbox is the keyboard path.
 */
const BreakRow = ({
  slot,
  selected,
  sponsors,
  sponsorId,
  onAssign,
  onToggle,
  onPreview,
  previewBlocked,
  previewBusy,
  previewing,
  previewUrl,
  note,
}: BreakRowProps) => {
  const [expanded, setExpanded] = useState(false);
  const [showBlocked, setShowBlocked] = useState(false);
  const uid = useId();
  const detailsId = `${uid}-details`;
  const tagsId = `${uid}-tags`;
  const time = formatTimestamp(slot.time);
  const tags = reasonTags(slot);
  const perRowSponsor = selected && sponsors.length > 1;
  // With a sponsor select in the row, the tags' tail gives way to it.
  const visibleTags = perRowSponsor ? tags.slice(0, 1) : tags;

  const onRowClick = (event: MouseEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    if (target.closest("button, input, select, a, audio, label")) return;
    onToggle(slot.id);
  };

  const preview = () => {
    if (previewBlocked) {
      setShowBlocked(true);
      return;
    }
    setShowBlocked(false);
    onPreview(slot.id);
  };

  const showOutput = previewing || Boolean(previewUrl) || Boolean(note) || showBlocked;

  return (
    <li
      id={`break-${slot.id}`}
      className={`break-row${selected ? " is-selected" : ""}${expanded ? " is-expanded" : ""}`}
    >
      <div className="break-row-main" onClick={onRowClick}>
        <input
          type="checkbox"
          className="checkbox break-check"
          checked={selected}
          onChange={() => onToggle(slot.id)}
          aria-label={`Break at ${time}`}
          aria-describedby={tagsId}
        />
        <span className="break-time">{time}</span>
        <span className="break-score">
          {slot.placementScore === null ? (
            <span className="break-score-num is-empty">
              <span aria-hidden="true">—</span>
              <span className="visually-hidden">no score</span>
            </span>
          ) : (
            <>
              <span className="break-score-num">
                {slot.placementScore}
                <span className="visually-hidden"> of 100</span>
              </span>
              <span className="score-bar" aria-hidden="true">
                <span
                  className="score-bar-fill"
                  style={{ width: `${Math.max(0, Math.min(100, slot.placementScore))}%` }}
                />
              </span>
            </>
          )}
        </span>
        <span className="break-reasons">
          <span className="break-tags" id={tagsId}>
            {visibleTags.length === 0 ? (
              <span className="reason-tag">no signal measured</span>
            ) : (
              visibleTags.map((tag, index) => (
                <span key={tag.text} className={`reason-tag reason-tag-${tag.kind}`}>
                  {index > 0 && <span aria-hidden="true"> · </span>}
                  {tag.text}
                </span>
              ))
            )}
          </span>
          {perRowSponsor && (
            <select
              className="break-sponsor"
              value={sponsorId}
              onChange={(event) => onAssign(slot.id, event.target.value)}
              aria-label={`Sponsor read at ${time}`}
            >
              {sponsors.map((sponsor, index) => (
                <option key={sponsor.id} value={sponsor.id}>
                  {sponsorLabel(sponsor, index)}
                </option>
              ))}
            </select>
          )}
        </span>
        <button
          type="button"
          className={`icon-button break-preview${previewBlocked ? " is-disabled" : ""}`}
          onClick={preview}
          disabled={previewBusy}
          aria-disabled={previewBlocked ? true : undefined}
          aria-label={`Preview the ad at ${time}`}
          title={previewBlocked ?? `Preview the ad at ${time}`}
        >
          <PlayIcon />
        </button>
        <button
          type="button"
          className="icon-button break-expand"
          onClick={() => setExpanded((open) => !open)}
          aria-expanded={expanded}
          aria-controls={detailsId}
          aria-label={`Why the break at ${time}`}
        >
          <ChevronIcon />
        </button>
      </div>

      {expanded && (
        <div className="break-details" id={detailsId}>
          {slot.source !== "manual" && (
            <ul className="break-signals">
              {slot.signals.length === 0 && <li>No signal was measured for this point.</li>}
              {slot.signals.map((signal, index) => (
                <li key={`${signal.label}-${index}`} className={`break-signal is-${signal.kind}`}>
                  {signal.kind === "supporting" ? <CheckIcon /> : <AlertIcon size={12} />}
                  <span>
                    <span className="visually-hidden">
                      {signal.kind === "supporting" ? "Supports: " : "Caution: "}
                    </span>
                    {signal.label}
                  </span>
                </li>
              ))}
            </ul>
          )}
          {slot.rationale && <p className="break-rationale">{slot.rationale}</p>}
          <p className="break-meta">
            {slot.rank !== null ? `rank ${slot.rank} · ` : ""}at {slot.time.toFixed(2)}s
          </p>
        </div>
      )}

      {showOutput && (
        <div className="break-output" aria-live="polite">
          {showBlocked && previewBlocked && <p className="break-note">{previewBlocked}</p>}
          {previewing && <p className="break-note">Generating a preview of the ad at {time}…</p>}
          {!previewing && note && (
            <p className={note.tone === "error" ? "status-error" : note.tone === "ok" ? "status-ok" : "break-note"}>
              {note.text}
            </p>
          )}
          {!previewing && previewUrl && (
            <audio
              className="break-audio"
              controls
              autoPlay
              src={previewUrl}
              aria-label={`Preview of the ad at ${time}`}
            />
          )}
        </div>
      )}
    </li>
  );
};

export default BreakRow;
