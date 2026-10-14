import type { CSSProperties } from "react";

type BarsAnimationProps = {
  /** `overlay`: the voice-cloning screen. `inline`: analysis in the break list. */
  variant: "overlay" | "inline";
};

const shape = (count: number, peak: number) =>
  Array.from({ length: count }, (_, i) => {
    // Additive sines, so the bars look like speech rather than a sawtooth.
    const h =
      Math.abs(Math.sin(i / 5.2) * 0.35) +
      Math.abs(Math.sin(i / 2.1 + 1.3) * 0.23) +
      Math.abs(Math.sin(i / 8.7 + 0.5) * 0.2) +
      0.1;
    return Math.max(4, Math.floor(Math.min(1, h) * peak));
  });

const BARS = {
  overlay: shape(52, 80),
  inline: shape(24, 24),
};

/**
 * Wave bars pulsing in `--wave`, with one amber "active" bar sweeping across
 * (DESIGN.md `inline-progress` and `processing-overlay`).
 */
const BarsAnimation = ({ variant }: BarsAnimationProps) => (
  <div className={`bars bars-${variant}`} aria-hidden="true">
    {BARS[variant].map((h, i) => (
      <span
        key={i}
        className="bar"
        style={
          {
            // Custom properties rather than animationDelay/Duration, which
            // would also override the active-bar sweep.
            "--peak-h": `${h}px`,
            "--bar-delay": `${(i * 0.038) % 0.95}s`,
            "--bar-dur": `${0.55 + (i % 7) * 0.06}s`,
            "--bar-index": i,
          } as CSSProperties
        }
      />
    ))}
  </div>
);

export default BarsAnimation;
