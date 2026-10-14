export interface MergeFilter {
  filter: string;
  previewStart: number | null;
  previewEnd: number | null;
}

const FORMAT = "aformat=sample_rates=44100:channel_layouts=stereo";

// Fixed decimals: `${1e-7}` interpolates as "1e-7", which ffmpeg's filter
// parser does not read as a time.
const seconds = (value: number): string => value.toFixed(3);
const isAudible = (value: number): boolean => Number(seconds(value)) > 0;

/**
 * Build the ffmpeg `filter_complex` that splices the ad (input 1) into the main
 * audio (input 0) at `insertAt`. When `previewSeconds` is set, only a window
 * around the insertion point is rendered.
 *
 * `pause` seconds of silence go on each side of the ad, and the main audio
 * fades out over `crossfade` seconds into the break and back in after it.
 * These are fades rather than ffmpeg's overlapping `acrossfade`, which never
 * returns when one side is shorter than the fade (an insert at 0 s, say).
 */
export const buildMergeFilter = ({
  insertAt,
  previewSeconds,
  mainDuration,
  crossfade = 0,
  pause = 0,
}: {
  insertAt: number;
  previewSeconds: number;
  mainDuration: number | null;
  crossfade?: number;
  pause?: number;
}): MergeFilter => {
  const normalizedInsertAt = Math.max(0, insertAt);
  const isPreview = Boolean(previewSeconds && previewSeconds > 0);
  const previewStart = isPreview ? Math.max(0, normalizedInsertAt - previewSeconds) : null;
  const previewEnd = isPreview
    ? mainDuration
      ? Math.min(mainDuration, normalizedInsertAt + previewSeconds)
      : normalizedInsertAt + previewSeconds
    : null;

  const headStart = previewStart ?? 0;
  const headLength = normalizedInsertAt - headStart;
  const tailLength =
    previewEnd !== null
      ? previewEnd - normalizedInsertAt
      : mainDuration
        ? Math.max(0, mainDuration - normalizedInsertAt)
        : null;
  // A fade never runs longer than the audio it shapes.
  const fadeOut = Math.max(0, Math.min(crossfade, headLength));
  const fadeIn = Math.max(0, tailLength === null ? crossfade : Math.min(crossfade, tailLength));

  const head =
    `[0:a]${FORMAT},atrim=${seconds(headStart)}:${seconds(normalizedInsertAt)},` +
    "asetpts=PTS-STARTPTS" +
    (isAudible(fadeOut)
      ? `,afade=t=out:st=${seconds(headLength - fadeOut)}:d=${seconds(fadeOut)}`
      : "") +
    "[a0]";
  const tailTrim =
    previewEnd !== null
      ? `${seconds(normalizedInsertAt)}:${seconds(previewEnd)}`
      : seconds(normalizedInsertAt);
  const tail =
    `[0:a]${FORMAT},atrim=${tailTrim},asetpts=PTS-STARTPTS` +
    (isAudible(fadeIn) ? `,afade=t=in:st=0:d=${seconds(fadeIn)}` : "") +
    "[a1]";
  const ad = `[1:a]${FORMAT},asetpts=PTS-STARTPTS[ad]`;

  const parts = [head, tail, ad];
  let segments = ["[a0]", "[ad]", "[a1]"];
  if (isAudible(pause)) {
    const silence = `anullsrc=r=44100:cl=stereo,atrim=0:${seconds(pause)}`;
    parts.push(`${silence}[p0]`, `${silence}[p1]`);
    segments = ["[a0]", "[p0]", "[ad]", "[p1]", "[a1]"];
  }
  parts.push(`${segments.join("")}concat=n=${segments.length}:v=0:a=1[out]`);

  return { filter: parts.join(";"), previewStart, previewEnd };
};
