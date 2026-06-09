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
        <source src="/src/assets/slotify-soundwave.webm" type="video/webm" />
        <source src="/src/assets/slotify-soundwave.mp4" type="video/mp4" />
      </video>
      <div className="soundwave-overlay" />
    </div>
  );
};

export default SoundwaveBackground;
