import { useEffect, useId } from "react";
import SlotifyLogo from "./SlotifyLogo";
import { AlertIcon } from "./icons";

type RightsModalProps = {
  accepted: boolean;
  onAcceptedChange: (accepted: boolean) => void;
  onCancel: () => void;
  onConfirm: () => void;
};

/**
 * The voice-rights confirmation in front of the first preview or render:
 * generating the ad clones the host's voice. Escape cancels.
 */
const RightsModal = ({ accepted, onAcceptedChange, onCancel, onConfirm }: RightsModalProps) => {
  const uid = useId();

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  return (
    <div className="modal-backdrop">
      <div
        className="modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${uid}-title`}
        aria-describedby={`${uid}-body`}
      >
        <div className="modal-header">
          <h2 className="modal-title" id={`${uid}-title`}>
            <AlertIcon size={18} className="modal-icon" />
            Voice rights
          </h2>
          <button type="button" className="icon-button" onClick={onCancel} aria-label="Close">
            ×
          </button>
        </div>
        <p className="modal-body" id={`${uid}-body`}>
          Break points were found without touching your voice. Generating the
          sponsor read clones it, so please confirm you have the rights.
        </p>
        <label className="modal-check">
          <input
            type="checkbox"
            className="checkbox"
            checked={accepted}
            autoFocus
            onChange={(event) => onAcceptedChange(event.target.checked)}
          />
          <span>
            I certify that I own the rights to this audio and voice. I understand
            that <SlotifyLogo size="sm" variant="light" /> will generate sponsor
            audio using voice cloning technology, and I confirm that no
            unauthorized voice impersonation is involved.
          </span>
        </label>
        <div className="modal-actions">
          <button type="button" className="ghost" onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="primary" disabled={!accepted} onClick={onConfirm}>
            Continue to generation
          </button>
        </div>
      </div>
    </div>
  );
};

export default RightsModal;
