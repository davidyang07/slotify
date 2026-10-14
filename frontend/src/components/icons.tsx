/**
 * Small inline icons. All are decorative (`aria-hidden`): every control that
 * uses one carries its own accessible label.
 */

type IconProps = { size?: number; className?: string };

const base = (size: number, className?: string) => ({
  width: size,
  height: size,
  className,
  "aria-hidden": true as const,
  focusable: "false" as const,
});

export const PlayIcon = ({ size = 14, className }: IconProps) => (
  <svg viewBox="0 0 16 16" {...base(size, className)}>
    <path d="M4.5 2.8v10.4c0 .5.6.8 1 .5l8-5.2a.6.6 0 0 0 0-1L5.5 2.3a.6.6 0 0 0-1 .5Z" fill="currentColor" />
  </svg>
);

export const PauseIcon = ({ size = 14, className }: IconProps) => (
  <svg viewBox="0 0 16 16" {...base(size, className)}>
    <rect x="3.5" y="2.5" width="3" height="11" rx="1" fill="currentColor" />
    <rect x="9.5" y="2.5" width="3" height="11" rx="1" fill="currentColor" />
  </svg>
);

export const ChevronIcon = ({ size = 14, className }: IconProps) => (
  <svg viewBox="0 0 16 16" fill="none" {...base(size, className)}>
    <path
      d="m4 6 4 4 4-4"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

export const CheckIcon = ({ size = 12, className }: IconProps) => (
  <svg viewBox="0 0 12 12" fill="none" {...base(size, className)}>
    <path
      d="M2.5 6.2 4.9 8.5 9.5 3.5"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

/** A caution or a caveat: ink-muted with this icon, never a warning colour. */
export const AlertIcon = ({ size = 14, className }: IconProps) => (
  <svg viewBox="0 0 16 16" fill="none" {...base(size, className)}>
    <circle cx="8" cy="8" r="6.25" stroke="currentColor" strokeWidth="1.3" />
    <path d="M8 4.8v3.7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    <circle cx="8" cy="11.1" r=".85" fill="currentColor" />
  </svg>
);

export const UploadIcon = ({ size = 24, className }: IconProps) => (
  <svg viewBox="0 0 24 24" {...base(size, className)}>
    <path
      d="M12 3a1 1 0 0 1 1 1v8.6l2.3-2.3a1 1 0 1 1 1.4 1.4l-4.01 4a1 1 0 0 1-1.38 0l-4.01-4a1 1 0 0 1 1.42-1.4L11 12.6V4a1 1 0 0 1 1-1z"
      fill="currentColor"
    />
    <path
      d="M5 15a1 1 0 0 1 1 1v2h12v-2a1 1 0 1 1 2 0v3a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1z"
      fill="currentColor"
    />
  </svg>
);

export const GitHubIcon = ({ size = 20, className }: IconProps) => (
  <svg viewBox="0 0 24 24" fill="currentColor" {...base(size, className)}>
    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
  </svg>
);
