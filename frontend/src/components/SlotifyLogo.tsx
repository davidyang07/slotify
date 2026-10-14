import React from "react";

interface SlotifyLogoProps {
  size?: "sm" | "md" | "lg";
  variant?: "light" | "dark";
  className?: string;
}

const SlotifyLogo: React.FC<SlotifyLogoProps> = ({
  size = "md",
  variant = "dark",
  className = "",
}) => {
  // Size mappings for font-size
  const sizeConfig = {
    sm: {
      fontSize: "0.875rem", // 14px, inline in body copy
      letterSpacing: "-0.01em",
    },
    md: {
      fontSize: "1.125rem", // 18px, the top nav
      letterSpacing: "-0.02em",
    },
    lg: {
      fontSize: "3rem", // 48px
      letterSpacing: "-0.035em",
    },
  };

  const config = sizeConfig[size];
  // DESIGN.md: the wordmark is ink on the dark surfaces, never gradient text.
  // "dark" is for the rare light/amber background, where text is on-accent.
  const textColor = variant === "light" ? "var(--ink)" : "var(--on-accent)";

  return (
    <span
      className={`slotify-logo ${className}`}
      aria-label="Slotify"
      style={{
        display: "inline-flex",
        alignItems: "center",
        fontFamily: "var(--font-sans)",
        fontSize: config.fontSize,
        fontWeight: 600,
        color: textColor,
        letterSpacing: config.letterSpacing,
        lineHeight: 1,
      }}
    >
      Slotify
    </span>
  );
};

export default SlotifyLogo;
