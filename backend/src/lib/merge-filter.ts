export interface MergeFilter {
  filter: string;
  previewStart: number | null;
  previewEnd: number | null;
}

/**
 * Build the ffmpeg `filter_complex` that splices the ad (input 1) into the main
 * audio (input 0) at `insertAt`. When `previewSeconds` is set, only a window
 * around the insertion point is rendered.
 */
export const buildMergeFilter = ({
  insertAt,
  previewSeconds,
  mainDuration,
}: {
  insertAt: number;
  previewSeconds: number;
  mainDuration: number | null;
}): MergeFilter => {
  const normalizedInsertAt = Math.max(0, insertAt);
  if (previewSeconds && previewSeconds > 0) {
    const previewStart = Math.max(0, normalizedInsertAt - previewSeconds);
    const previewEnd = mainDuration
      ? Math.min(mainDuration, normalizedInsertAt + previewSeconds)
      : normalizedInsertAt + previewSeconds;
    return {
      filter: [
        `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
          `atrim=${previewStart}:${normalizedInsertAt},asetpts=PTS-STARTPTS[a0]`,
        `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
          `atrim=${normalizedInsertAt}:${previewEnd},asetpts=PTS-STARTPTS[a1]`,
        `[1:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
          `asetpts=PTS-STARTPTS[ad]`,
        `[a0][ad][a1]concat=n=3:v=0:a=1[out]`,
      ].join(";"),
      previewStart,
      previewEnd,
    };
  }
  return {
    filter: [
      `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
        `atrim=0:${normalizedInsertAt},asetpts=PTS-STARTPTS[a0]`,
      `[0:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
        `atrim=${normalizedInsertAt},asetpts=PTS-STARTPTS[a1]`,
      `[1:a]aformat=sample_rates=44100:channel_layouts=stereo,` +
        `asetpts=PTS-STARTPTS[ad]`,
      `[a0][ad][a1]concat=n=3:v=0:a=1[out]`,
    ].join(";"),
    previewStart: null,
    previewEnd: null,
  };
};
