import BarsAnimation from "./BarsAnimation";

type ProcessingOverlayProps = {
  onCancel: () => void;
};

/**
 * DESIGN.md `processing-overlay`: full screen, only for voice cloning, which
 * is short and must block. Analysis never uses it. A hung request must never
 * lock the screen, so it can always be cancelled.
 */
const ProcessingOverlay = ({ onCancel }: ProcessingOverlayProps) => (
  <div className="processing-screen" role="status" aria-live="polite" aria-label="Cloning voice">
    <BarsAnimation variant="overlay" />
    <div className="processing-text">
      <p className="processing-status">Cloning the host&rsquo;s voice</p>
      <p className="processing-substatus">
        Reading vocal characteristics from your audio. The ad read is generated next.
      </p>
      <button type="button" className="secondary processing-cancel" onClick={onCancel}>
        Cancel
      </button>
    </div>
  </div>
);

export default ProcessingOverlay;
