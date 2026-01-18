import { useState } from "react";
import SoundwaveBackground from "./SoundwaveBackground";
import "./LandingHero.css";

type LandingHeroProps = {
  onPrimaryAction?: () => void;
  onSecondaryAction?: () => void;
};

const LandingHero = ({
  onPrimaryAction,
  onSecondaryAction,
}: LandingHeroProps) => {
  const [isTitleHover, setIsTitleHover] = useState(false);
  const [isButtonHover, setIsButtonHover] = useState(false);

  const isActive = isTitleHover || isButtonHover;

  return (
    <section className="landing-hero-section">
      <SoundwaveBackground isActive={isActive} />
      <div className="landing-hero-content">
        <div
          className="landing-hero-title-wrap"
          onMouseEnter={() => setIsTitleHover(true)}
          onMouseLeave={() => setIsTitleHover(false)}
        >
          <h1 className="landing-hero-title">Slotify</h1>
        </div>

        <p className="landing-hero-subtitle">
          Seamless sponsor inserts for audio.
        </p>
        <p className="landing-hero-body">
          Automate mid-roll placements with AI-generated reads that match the
          creator's voice and flow naturally inside each episode.
        </p>

        <div className="landing-hero-actions">
          <button
            type="button"
            className="landing-hero-cta landing-hero-primary"
            onMouseEnter={() => setIsButtonHover(true)}
            onMouseLeave={() => setIsButtonHover(false)}
            onClick={onPrimaryAction}
          >
            Try Slotify
          </button>
          <button
            type="button"
            className="landing-hero-cta landing-hero-secondary"
            onClick={onSecondaryAction}
          >
            View Demo
          </button>
        </div>
      </div>
    </section>
  );
};

export default LandingHero;
