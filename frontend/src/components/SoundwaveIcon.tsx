import React from "react";

interface SoundwaveIconProps {
  size?: "sm" | "md" | "lg";
  variant?: "light" | "dark";
  className?: string;
}

const SoundwaveIcon: React.FC<SoundwaveIconProps> = ({
  size = "md",
  variant = "dark",
  className = "",
}) => {
  // Size mappings for icon dimensions (all in em for scalability)
  const sizeConfig = {
    sm: {
      iconSize: "0.875rem", // ~14px base
      barHeight: "1.1em", // Main bar height
      sideBarHeight: "0.5em", // Side bars height
      barWidth: "0.12em", // Main bar width
      sideBarWidth: "0.08em", // Side bars width (thinner)
      gap: "0.1em", // Gap between center bar and side bars
      borderRadius: "0.06em",
    },
    md: {
      iconSize: "1.5rem", // ~24px base
      barHeight: "1.1em",
      sideBarHeight: "0.5em",
      barWidth: "0.12em",
      sideBarWidth: "0.08em",
      gap: "0.1em",
      borderRadius: "0.06em",
    },
    lg: {
      iconSize: "3rem", // ~48px base
      barHeight: "1.1em",
      sideBarHeight: "0.5em",
      barWidth: "0.12em",
      sideBarWidth: "0.08em",
      gap: "0.1em",
      borderRadius: "0.06em",
    },
  };

  const config = sizeConfig[size];
  const textColor = variant === "light" ? "#f6efe6" : "#231f1a";

  // Calculate positions: center bar with side bars flanking it
  const centerBarLeft = `calc(${config.sideBarWidth} + ${config.gap})`;
  const totalWidth = `calc(${config.sideBarWidth} + ${config.gap} + ${config.barWidth} + ${config.gap} + ${config.sideBarWidth})`;
  
  // Calculate vertical centering: center all bars
  const textCenter = "0.4em";
  const mainBarBottom = `calc(${textCenter} - ${config.barHeight} / 2)`;
  const sideBarBottom = `calc(${textCenter} - ${config.sideBarHeight} / 2)`;

  return (
    <span
      className={`soundwave-icon ${className}`}
      aria-label="Soundwave icon"
      style={{
        display: "inline-flex",
        alignItems: "center",
        fontSize: config.iconSize,
        lineHeight: 1,
        transform: "translateY(-0.1em)",
      }}
    >
      <span
        style={{
          position: "relative",
          display: "inline-block",
          width: totalWidth,
          height: config.barHeight,
        }}
      >
        {/* Left side bar */}
        <svg
          width={config.sideBarWidth}
          height={config.sideBarHeight}
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          style={{
            position: "absolute",
            left: 0,
            bottom: sideBarBottom,
          }}
          aria-hidden="true"
        >
          <rect
            width="100%"
            height="100%"
            rx={config.borderRadius}
            ry={config.borderRadius}
            fill={textColor}
          />
        </svg>

        {/* Main center bar */}
        <svg
          width={config.barWidth}
          height={config.barHeight}
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          style={{
            position: "absolute",
            left: centerBarLeft,
            bottom: mainBarBottom,
          }}
          aria-hidden="true"
        >
          <rect
            width="100%"
            height="100%"
            rx={config.borderRadius}
            ry={config.borderRadius}
            fill={textColor}
          />
        </svg>

        {/* Right side bar */}
        <svg
          width={config.sideBarWidth}
          height={config.sideBarHeight}
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          style={{
            position: "absolute",
            right: 0,
            bottom: sideBarBottom,
          }}
          aria-hidden="true"
        >
          <rect
            width="100%"
            height="100%"
            rx={config.borderRadius}
            ry={config.borderRadius}
            fill={textColor}
          />
        </svg>
      </span>
    </span>
  );
};

export default SoundwaveIcon;
