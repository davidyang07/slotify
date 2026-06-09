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
      fontSize: "0.875rem", // ~14px base
    },
    md: {
      fontSize: "1.5rem", // ~24px base
    },
    lg: {
      fontSize: "3rem", // ~48px base
    },
  };

  const config = sizeConfig[size];
  const textColor = variant === "light" ? "#f6efe6" : "#231f1a";

  return (
    <span
      className={`slotify-logo ${className}`}
      aria-label="Slotify"
      style={{
        display: "inline-flex",
        alignItems: "center",
        fontFamily: '"Space Grotesk", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        fontSize: config.fontSize,
        fontWeight: 600,
        color: textColor,
        letterSpacing: "0.02em",
        lineHeight: 1,
      }}
    >
      Slotify
    </span>
  );
};

export default SlotifyLogo;
