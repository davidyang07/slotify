import { useEffect, useRef } from "react";

type SoundwaveBackgroundProps = {
  isActive: boolean;
};

const RESET_DELAY_MS = 150;

const SoundwaveBackground = ({ isActive }: SoundwaveBackgroundProps) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const resetTimeoutRef = useRef<number | null>(null);

  const clearResetTimeout = () => {
    if (resetTimeoutRef.current !== null) {
      window.clearTimeout(resetTimeoutRef.current);
      resetTimeoutRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      clearResetTimeout();
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (isActive) {
      clearResetTimeout();
      const playPromise = video.play();
      if (playPromise) {
        playPromise.catch(() => undefined);
      }
      return;
    }

    video.pause();
    clearResetTimeout();
    resetTimeoutRef.current = window.setTimeout(() => {
      if (videoRef.current) {
        videoRef.current.currentTime = 0;
      }
    }, RESET_DELAY_MS);
  }, [isActive]);

  const handleLoadedMetadata = () => {
    if (!videoRef.current) return;
    videoRef.current.pause();
    videoRef.current.currentTime = 0;
  };

  return (
    <div className="soundwave-bg" data-active={isActive} aria-hidden="true">
      <video
        ref={videoRef}
        className="soundwave-video"
        muted
        loop
        playsInline
        preload="metadata"
        onLoadedMetadata={handleLoadedMetadata}
      >
        <source src="/src/assets/slotify-soundwave.webm" type="video/webm" />
        <source src="/src/assets/slotify-soundwave.mp4" type="video/mp4" />
      </video>
      <div className="soundwave-overlay" />
    </div>
  );
};

export default SoundwaveBackground;
