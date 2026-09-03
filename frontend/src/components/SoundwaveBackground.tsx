// Imported rather than referenced by URL. A literal "/src/assets/..." path is
// served by the dev server and by nothing else, so the background silently 404s
// in a production build; importing it makes Vite emit and fingerprint the file.
import soundwaveMp4 from "../assets/slotify-soundwave.mp4";

type SoundwaveBackgroundProps = {
  isActive?: boolean;
};

const SoundwaveBackground = ({ isActive = false }: SoundwaveBackgroundProps) => {
  return (
    <div className="soundwave-bg" data-active={isActive} aria-hidden="true">
      <video
        className="soundwave-video"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
      >
        <source src={soundwaveMp4} type="video/mp4" />
      </video>
      <div className="soundwave-overlay" />
    </div>
  );
};

export default SoundwaveBackground;
