import type { MouseEvent } from "react";
import { useEffect, useRef, useState } from "react";
import { formatTimestamp } from "../lib/format";
import type { Slot } from "../lib/recommendations";
import { PauseIcon, PlayIcon } from "./icons";
import "./WaveformPanel.css";

type WaveformPanelProps = {
  /** Decoded once per file by App; drawing never re-decodes. */
  peaks: Float32Array | null;
  decodeFailed: boolean;
  durationSeconds: number | null;
  slots: Slot[];
  selectedIds: string[];
  /** The break the insertion cursor sits on. */
  focusedId: string | null;
  /** Clicking a marker toggles its break and brings its row into view. */
  onMarker: (slotId: string) => void;
  /** The uploaded file, for listening to the original. */
  originalUrl: string;
};

const token = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

/** Paint the bars; `well` fills the background first (the base layer only). */
const paint = (
  canvas: HTMLCanvasElement,
  peaks: Float32Array | null,
  color: string,
  well: string | null,
) => {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (well) {
    ctx.fillStyle = well;
    ctx.fillRect(0, 0, width, height);
  }
  if (!peaks || peaks.length === 0) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  const mid = height / 2;
  const step = width / peaks.length;
  ctx.beginPath();
  for (let i = 0; i < peaks.length; i += 1) {
    const barHeight = Math.max(1, (peaks[i] ?? 0) * height * 0.9);
    const x = i * step + 0.5;
    ctx.moveTo(x, mid - barHeight / 2);
    ctx.lineTo(x, mid + barHeight / 2);
  }
  ctx.stroke();
};

const percent = (time: number, duration: number) =>
  `${Math.max(0, Math.min(1, time / duration)) * 100}%`;

/**
 * DESIGN.md `waveform-panel`: a canvas well in a 12px hairline frame. Bars in
 * --wave (--wave-strong under the playhead), unselected markers in
 * --wave-marker, selected markers amber, the insertion cursor in ink.
 * Markers and the cursor are DOM, so selecting a break never repaints the
 * canvas; it is painted when the peaks or its size change.
 */
const WaveformPanel = ({
  peaks,
  decodeFailed,
  durationSeconds,
  slots,
  selectedIds,
  focusedId,
  onMarker,
  originalUrl,
}: WaveformPanelProps) => {
  const wellRef = useRef<HTMLDivElement | null>(null);
  const baseRef = useRef<HTMLCanvasElement | null>(null);
  const playedRef = useRef<HTMLCanvasElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    const well = wellRef.current;
    if (!well) return;
    const draw = () => {
      // Colours come from the tokens in index.css, so the canvas and the CSS
      // around it cannot drift apart.
      if (baseRef.current) {
        paint(baseRef.current, peaks, token("--wave", "#8a8f98"), token("--canvas", "#010102"));
      }
      if (playedRef.current) {
        paint(playedRef.current, peaks, token("--wave-strong", "#d0d6e0"), null);
      }
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(well);
    return () => observer.disconnect();
  }, [peaks]);

  // Follow the playhead every frame while playing; timeupdate is only ~4 Hz.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      const audio = audioRef.current;
      if (audio) setCurrentTime(audio.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  const duration = durationSeconds && durationSeconds > 0 ? durationSeconds : null;
  const focused = slots.find((slot) => slot.id === focusedId) ?? null;

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      audio.play().catch(() => undefined);
    } else {
      audio.pause();
    }
  };

  /** Clicking the well (not a marker) moves the playhead there. */
  const seek = (event: MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    if (!audio || !duration) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    audio.currentTime = ratio * duration;
    setCurrentTime(audio.currentTime);
  };

  const playedPercent = duration ? (Math.min(currentTime, duration) / duration) * 100 : 0;

  return (
    <section className="waveform-panel" aria-label="Waveform and break markers">
      <div className="waveform-well" ref={wellRef} onClick={seek}>
        <canvas ref={baseRef} className="waveform-canvas" aria-hidden="true" />
        <canvas
          ref={playedRef}
          className="waveform-canvas waveform-canvas-played"
          aria-hidden="true"
          style={{ clipPath: `inset(0 ${100 - playedPercent}% 0 0)` }}
        />
        {!peaks && (
          <p className="waveform-message">
            {decodeFailed
              ? "This browser could not draw the waveform. Break points and rendering still work."
              : "Drawing the waveform…"}
          </p>
        )}
        {duration && currentTime > 0 && (
          <span
            className="waveform-playhead"
            style={{ left: percent(currentTime, duration) }}
            aria-hidden="true"
          />
        )}
        {duration && focused && (
          <span
            className="waveform-cursor"
            style={{ left: percent(focused.time, duration) }}
            aria-hidden="true"
          />
        )}
        {duration &&
          slots.map((slot) => {
            const selected = selectedIds.includes(slot.id);
            const label = formatTimestamp(slot.time);
            return (
              <button
                key={slot.id}
                type="button"
                className={`waveform-marker${selected ? " is-selected" : ""}`}
                style={{ left: percent(slot.time, duration) }}
                aria-pressed={selected}
                aria-label={`Break at ${label}`}
                title={
                  slot.placementScore === null
                    ? `${label} · added by you`
                    : `${label} · score ${slot.placementScore}`
                }
                onClick={(event) => {
                  event.stopPropagation();
                  onMarker(slot.id);
                }}
              />
            );
          })}
      </div>
      <div className="waveform-footer">
        <button
          type="button"
          className="icon-button waveform-play"
          onClick={togglePlay}
          disabled={!originalUrl}
          aria-label={playing ? "Pause the original audio" : "Play the original audio"}
        >
          {playing ? <PauseIcon /> : <PlayIcon />}
        </button>
        <span className="waveform-time">{formatTimestamp(currentTime)}</span>
        <span className="waveform-time waveform-time-end">
          {duration ? formatTimestamp(duration) : "-:--"}
        </span>
      </div>
      <audio
        ref={audioRef}
        src={originalUrl || undefined}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
        onEmptied={() => {
          setPlaying(false);
          setCurrentTime(0);
        }}
      />
    </section>
  );
};

export default WaveformPanel;
