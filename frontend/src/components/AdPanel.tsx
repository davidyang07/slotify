import { useId } from "react";
import {
  LANGUAGES,
  MAX_SPONSORS,
  TONES,
  type GenerationNote,
  type Sponsor,
} from "../lib/workspace";
import "./AdPanel.css";

type AdPanelProps = {
  sponsors: Sponsor[];
  onUpdateSponsor: (id: string, patch: Partial<Sponsor>) => void;
  onAddSponsor: () => void;
  onRemoveSponsor: (id: string) => void;
  tone: string;
  onTone: (tone: string) => void;
  language: string;
  onLanguage: (language: string) => void;
  /** Said when the server cannot write scripts with OpenAI, else null. */
  scriptWriterNote: string | null;
  renderLabel: string;
  /** Why Render is off, or null when it is on. */
  renderBlocked: string | null;
  isRendering: boolean;
  onRender: () => void;
  /** What the last render said. */
  note: GenerationNote | null;
  renderUrl: string;
  downloadName: string;
  voiceCloneError: string;
  authHint: string | null;
};

/**
 * DESIGN.md `ad-panel`: everything about the ad, in order -- sponsors, tone and
 * language, then Render (the one amber button). After a render it shows the
 * result: an audio player, Download and "Render again".
 */
const AdPanel = ({
  sponsors,
  onUpdateSponsor,
  onAddSponsor,
  onRemoveSponsor,
  tone,
  onTone,
  language,
  onLanguage,
  scriptWriterNote,
  renderLabel,
  renderBlocked,
  isRendering,
  onRender,
  note,
  renderUrl,
  downloadName,
  voiceCloneError,
  authHint,
}: AdPanelProps) => {
  const uid = useId();
  const reasonId = `${uid}-render-reason`;
  const hasResult = Boolean(renderUrl) && !isRendering;
  const renderDisabled = isRendering || Boolean(renderBlocked);

  return (
    <aside className="ad-panel" aria-labelledby={`${uid}-title`}>
      <h2 className="ad-panel-title" id={`${uid}-title`}>
        Sponsor read
      </h2>

      <section className="ad-section" aria-labelledby={`${uid}-sponsors`}>
        <h3 className="eyebrow" id={`${uid}-sponsors`}>
          Sponsors
        </h3>
        <ol className="sponsor-list">
          {sponsors.map((sponsor, index) => (
            <li key={sponsor.id} className="sponsor">
              {sponsors.length > 1 && (
                <div className="sponsor-header">
                  <span className="sponsor-number">Sponsor {index + 1}</span>
                  <button
                    type="button"
                    className="ghost small"
                    onClick={() => onRemoveSponsor(sponsor.id)}
                    aria-label={`Remove sponsor ${index + 1}${sponsor.name.trim() ? `, ${sponsor.name.trim()}` : ""}`}
                  >
                    Remove
                  </button>
                </div>
              )}
              <div className="field">
                <label htmlFor={`${sponsor.id}-name`}>Name</label>
                <input
                  id={`${sponsor.id}-name`}
                  type="text"
                  placeholder="e.g. Morning Roast Coffee"
                  autoComplete="off"
                  value={sponsor.name}
                  onChange={(event) => onUpdateSponsor(sponsor.id, { name: event.target.value })}
                />
              </div>
              <div className="field">
                <label htmlFor={`${sponsor.id}-script`}>
                  Script <span className="optional">(optional)</span>
                </label>
                <textarea
                  id={`${sponsor.id}-script`}
                  rows={2}
                  placeholder="One sentence, about 10 seconds. Leave empty and Slotify writes it."
                  value={sponsor.script}
                  onChange={(event) => onUpdateSponsor(sponsor.id, { script: event.target.value })}
                />
              </div>
            </li>
          ))}
        </ol>
        <button
          type="button"
          className="secondary"
          onClick={onAddSponsor}
          disabled={sponsors.length >= MAX_SPONSORS}
        >
          Add sponsor
        </button>
        {sponsors.length >= MAX_SPONSORS && (
          <p className="helper">Up to {MAX_SPONSORS} sponsors per episode.</p>
        )}
        {sponsors.length > 1 && (
          <p className="helper">Pick each selected break&rsquo;s sponsor in its row.</p>
        )}
      </section>

      <section className="ad-section" aria-labelledby={`${uid}-voice`}>
        <h3 className="eyebrow" id={`${uid}-voice`}>
          Voice
        </h3>
        <div className="voice-grid">
          <div className="field">
            <label htmlFor="tone-select">Tone</label>
            <select id="tone-select" value={tone} onChange={(event) => onTone(event.target.value)}>
              {TONES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="language-select">Language</label>
            <select
              id="language-select"
              value={language}
              onChange={(event) => onLanguage(event.target.value)}
            >
              {LANGUAGES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>
        <p className="helper">
          Applies to scripts Slotify writes; a typed script is read as written.
          {scriptWriterNote ? ` ${scriptWriterNote}` : ""}
        </p>
      </section>

      <section className="ad-section ad-render" aria-label="Render">
        {hasResult && (
          <div className="render-result">
            <p className="status-ok">{note?.tone === "ok" ? note.text : "Render complete."}</p>
            <audio
              className="render-audio"
              controls
              autoPlay
              src={renderUrl}
              aria-label="The episode with its ad breaks"
            />
            <a className="secondary" href={renderUrl} download={downloadName}>
              Download
            </a>
          </div>
        )}
        <button
          type="button"
          className="primary wide"
          onClick={onRender}
          disabled={renderDisabled}
          aria-describedby={renderBlocked ? reasonId : undefined}
        >
          {hasResult && !renderBlocked ? "Render again" : renderLabel}
        </button>
        {renderBlocked && !isRendering && (
          <p className="helper" id={reasonId}>
            {renderBlocked}
          </p>
        )}
        {isRendering && (
          <p className="render-progress" role="status">
            Generating each ad read and merging it into the episode…
          </p>
        )}
        {note?.tone === "error" && (
          <p className="status-error" role="alert">
            {note.text}
          </p>
        )}
        {note?.tone === "info" && <p className="render-progress">{note.text}</p>}
        {voiceCloneError && voiceCloneError !== note?.text && (
          <p className="status-error">{voiceCloneError}</p>
        )}
        {authHint && <p className="status-error">{authHint}</p>}
      </section>
    </aside>
  );
};

export default AdPanel;
