---
version: alpha
name: Slotify
description: "A near-black, precise audio workspace. Depth comes from a four-step surface ladder and 1px hairlines, not shadows or glows. Slotify amber (#c8a84a) is the single chromatic accent, kept for the brand mark, the primary action, focus, and the selected/active state. Inter carries every word; JetBrains Mono carries the numbers an editor checks — timestamps, scores, durations, file names. Adapted from Linear's design language (as documented in VoltAgent/awesome-design-md, MIT) with Slotify's own brand, accent and freely licensed fonts."

colors:
  accent: "#c8a84a"
  accent-hover: "#e2c876"
  accent-pressed: "#b39236"
  on-accent: "#010102"
  accent-line: "rgba(200, 168, 74, 0.55)"
  focus-ring: "#c8a84a"
  ink: "#f7f8f8"
  ink-muted: "#d0d6e0"
  ink-subtle: "#8a8f98"
  ink-tertiary: "#62666d"
  canvas: "#010102"
  surface-1: "#0f1011"
  surface-2: "#141516"
  surface-3: "#18191a"
  surface-4: "#191a1b"
  hairline: "#23252a"
  hairline-strong: "#34343a"
  hairline-tertiary: "#3e3e44"
  edge-highlight: "rgba(255, 255, 255, 0.04)"
  scrim: "rgba(0, 0, 0, 0.72)"
  success: "#27a644"
  danger: "#e5484d"
  wave: "#8a8f98"
  wave-strong: "#d0d6e0"
  wave-marker: "#62666d"

typography:
  display-xl:
    fontFamily: Inter
    fontSize: 80px
    fontWeight: 600
    lineHeight: 1.05
    letterSpacing: -3.0px
  display-md:
    fontFamily: Inter
    fontSize: 40px
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: -1.0px
  headline:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: -0.6px
  card-title:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: 500
    lineHeight: 1.3
    letterSpacing: -0.3px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: -0.1px
  body:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: -0.05px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0
  caption:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: 400
    lineHeight: 1.4
    letterSpacing: 0
  button:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: 500
    lineHeight: 1.2
    letterSpacing: 0
  eyebrow:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.3
    letterSpacing: 0.4px
  mono:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0
  mono-display:
    fontFamily: JetBrains Mono
    fontSize: 32px
    fontWeight: 500
    lineHeight: 1.1
    letterSpacing: -1.0px

rounded:
  xs: 4px
  sm: 6px
  md: 8px
  lg: 12px
  xl: 16px
  pill: 9999px

spacing:
  xxs: 4px
  xs: 8px
  sm: 12px
  md: 16px
  lg: 24px
  xl: 32px
  xxl: 48px
  section: 96px

components:
  app-header:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink-subtle}"
    typography: "{typography.body-sm}"
    height: 56px
    borderBottom: "1px {colors.hairline}"
  file-bar:
    textColor: "{colors.ink-muted}"
    typography: "{typography.mono}"
    gap: 12px
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    padding: 8px 14px
    minHeight: 36px
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
    textColor: "{colors.on-accent}"
  button-primary-pressed:
    backgroundColor: "{colors.accent-pressed}"
    textColor: "{colors.on-accent}"
  button-primary-wide:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    minHeight: 44px
  button-secondary:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.ink}"
    border: "1px {colors.hairline}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    padding: 8px 14px
  button-tertiary:
    backgroundColor: transparent
    textColor: "{colors.ink-muted}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    padding: 8px 14px
  button-disabled:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink-tertiary}"
    border: "1px {colors.hairline}"
  panel:
    backgroundColor: "{colors.surface-1}"
    border: "1px {colors.hairline}"
    rounded: "{rounded.xl}"
    padding: 32px
  card:
    backgroundColor: "{colors.surface-1}"
    border: "1px {colors.hairline}"
    rounded: "{rounded.lg}"
    padding: 24px
  break-list:
    backgroundColor: "{colors.surface-1}"
    border: "1px {colors.hairline}"
    rounded: "{rounded.lg}"
  break-row:
    backgroundColor: transparent
    borderBottom: "1px {colors.hairline}"
    typography: "{typography.body-sm}"
    minHeight: 52px
    padding: 10px 16px
  break-row-hover:
    backgroundColor: "{colors.surface-2}"
  break-row-selected:
    backgroundColor: "{colors.surface-2}"
    borderLeft: "2px {colors.accent}"
  score-bar:
    track: "{colors.surface-3}"
    fill: "{colors.ink-subtle}"
    height: 4px
    width: 64px
    rounded: "{rounded.pill}"
  reason-tag:
    textColor: "{colors.ink-subtle}"
    typography: "{typography.caption}"
  ad-panel:
    backgroundColor: "{colors.surface-1}"
    border: "1px {colors.hairline}"
    rounded: "{rounded.lg}"
    padding: 20px
    width: 360px
  score-badge:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink-muted}"
    typography: "{typography.mono}"
    rounded: "{rounded.pill}"
    padding: 2px 8px
  status-badge:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink-muted}"
    typography: "{typography.caption}"
    rounded: "{rounded.pill}"
    padding: 2px 8px
  provenance-chip:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink-muted}"
    border: "1px {colors.hairline}"
    typography: "{typography.mono}"
    rounded: "{rounded.xs}"
    padding: 2px 8px
  text-input:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.ink}"
    border: "1px {colors.hairline}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: 8px 12px
    minHeight: 40px
  text-input-focused:
    border: "1px {colors.hairline-strong}"
    outline: "2px {colors.focus-ring}"
  empty-state:
    textColor: "{colors.ink}"
    maxWidth: 560px
  dropzone:
    backgroundColor: "{colors.surface-1}"
    border: "1px dashed {colors.hairline-strong}"
    rounded: "{rounded.xl}"
    padding: 48px
  dropzone-active:
    backgroundColor: "{colors.surface-2}"
    border: "1px dashed {colors.accent-line}"
  waveform-panel:
    backgroundColor: "{colors.canvas}"
    border: "1px {colors.hairline}"
    rounded: "{rounded.lg}"
    padding: 16px
  inline-progress:
    textColor: "{colors.ink-muted}"
    typography: "{typography.body-sm}"
    bar: "{colors.wave}"
    barActive: "{colors.accent}"
  processing-overlay:
    backgroundColor: "rgba(1, 1, 2, 0.94)"
    textColor: "{colors.ink}"
  modal:
    backgroundColor: "{colors.surface-1}"
    border: "1px {colors.hairline-strong}"
    rounded: "{rounded.xl}"
    padding: 32px
  status-ok:
    textColor: "{colors.success}"
    typography: "{typography.body-sm}"
  status-error:
    textColor: "{colors.danger}"
    typography: "{typography.body-sm}"
---

## Overview

Slotify is a working tool on one screen: drop an episode, read a ranked list
of break points, pick some, render. The design gets out of the way of that
loop. The
page sits on `{colors.canvas}` (#010102, near-black with a faint cool tint),
and everything the user reads or touches lives on a lifted panel from a
four-step surface ladder (`{colors.surface-1}` → `{colors.surface-4}`) edged
with 1px hairlines (`{colors.hairline}` → `{colors.hairline-tertiary}`).
Hierarchy comes from lift and type weight, never from glow.

**Slotify amber** (`{colors.accent}` #c8a84a) is the only chromatic accent. It
marks exactly four things: the brand mark, the one primary action on a screen,
keyboard focus, and the selected state (a selected break row's edge and its
waveform marker). Because amber is light, text on
it is near-black (`{colors.on-accent}`), never white.

Inter carries every word, at weight 600 for display with measured negative
tracking and 400 for body. **JetBrains Mono** carries the numbers an editor
checks — timestamps, placement scores, durations, file names and provenance
labels — so they align and read as data.

The empty state keeps one cinematic element: the soundwave video, rendered
grayscale (its blue would be a second accent) and dimmed so the text keeps 4.5:1
over its brightest frames. Once a file is loaded the video is gone, and the
product's own surfaces — the waveform and the break list — are the visual.

**Key Characteristics:**
- Near-black canvas with a four-step surface ladder and 1px hairlines; no drop shadows on dark.
- One accent, amber, for brand mark, primary action, focus and selection only.
- Inter 600 display with negative tracking; Inter 400 body; JetBrains Mono for data.
- 8px buttons and inputs, 12px cards, 16px panels; pills only for badges.
- No gradients, glows, or gradient text in the app chrome.

## Colors

### Accent
- **Amber** (`{colors.accent}`): brand mark, the Render button, a selected break row's left edge, a chosen break's waveform marker, link emphasis.
- **Amber Hover** (`{colors.accent-hover}`) and **Amber Pressed** (`{colors.accent-pressed}`): states of the primary CTA only.
- **Accent Line** (`{colors.accent-line}`): the 1px border of a selected card or an active dropzone.
- **Focus Ring** (`{colors.focus-ring}`): 2px solid amber outline on any focused control. Solid, not translucent: amber at 50% alpha measures about 2.9:1 on these surfaces, under the 3:1 a focus indicator needs.

### Surface
- **Canvas** (`{colors.canvas}`): page background and the waveform well.
- **Surface 1** (`{colors.surface-1}`): panels, cards, inputs, the dropzone.
- **Surface 2** (`{colors.surface-2}`): selected and hovered cards, badges, the active nav link.
- **Surface 3 / 4** (`{colors.surface-3}`, `{colors.surface-4}`): dropdowns, nested panels inside a panel.
- **Hairline / Strong / Tertiary**: 1px borders, from default card edges to focused inputs and nested surfaces.
- **Edge Highlight** (`{colors.edge-highlight}`): an optional 1px inset top highlight on lifted panels.

### Text
- **Ink** (`{colors.ink}`): headings and primary text.
- **Ink Muted** (`{colors.ink-muted}`): secondary text, badge labels.
- **Ink Subtle** (`{colors.ink-subtle}`): helper text, inactive nav, captions. Meets 4.5:1 on every surface.
- **Ink Tertiary** (`{colors.ink-tertiary}`): disabled controls and non-essential footnotes only — it is about 3.3:1, so never for text a user must read.

### Semantic
- **Success** (`{colors.success}`): success status text ("Render complete").
- **Danger** (`{colors.danger}`): error text and destructive confirmations.
- No warning colour: a warning is ink-muted text with an icon, so amber keeps one meaning.

### Waveform
- **Wave** (`{colors.wave}`): the waveform bars.
- **Wave Strong** (`{colors.wave-strong}`): the portion under the playhead, when shown.
- **Wave Marker** (`{colors.wave-marker}`): an unselected break's marker line.
- A **selected** break's marker is `{colors.accent}`; the insertion cursor is `{colors.ink}`.

## Typography

### Font Family
- **Inter** (Google Fonts, weights 400/500/600) for display and text. Load once in `index.html`.
- **JetBrains Mono** (Google Fonts, weights 400/500) for data: timestamps (`0:08`), scores (`100/100`), durations, sizes, file names, ranker/provenance chips.

### Hierarchy

| Token | Size | Weight | Line Height | Tracking | Use |
|---|---|---|---|---|---|
| `{typography.display-xl}` | 80px | 600 | 1.05 | -3.0px | Landing wordmark only |
| `{typography.display-md}` | 40px | 600 | 1.15 | -1.0px | Landing subheading |
| `{typography.headline}` | 28px | 600 | 1.2 | -0.6px | Screen titles ("Recommended insertion points") |
| `{typography.card-title}` | 18px | 500 | 1.3 | -0.3px | Panel and card titles |
| `{typography.body-lg}` | 18px | 400 | 1.5 | -0.1px | Landing lead paragraph |
| `{typography.body}` | 15px | 400 | 1.5 | -0.05px | Default body, inputs |
| `{typography.body-sm}` | 14px | 400 | 1.5 | 0 | Break rows, panel body |
| `{typography.caption}` | 12px | 400 | 1.4 | 0 | Helper text, meta |
| `{typography.button}` | 14px | 500 | 1.2 | 0 | All button labels |
| `{typography.eyebrow}` | 12px | 500 | 1.3 | +0.4px | Section eyebrows, uppercase |
| `{typography.mono}` | 13px | 400 | 1.5 | 0 | Scores, sizes, chips |
| `{typography.mono-display}` | 32px | 500 | 1.1 | -1.0px | Reserved; break rows use `{typography.mono}` |

### Principles
- Display at 600, body at 400; nothing above 600.
- Negative tracking on display, positive on uppercase eyebrows.
- Numbers a user compares are mono; prose is never mono.
- No gradient text, including the wordmark.

## Layout

### Spacing
- 4px base unit: 4, 8, 12, 16, 24, 32, 48, 96.
- The ad panel pads 20px; break rows 10px × 16px; the empty state centres in the viewport.
- 16px between the waveform, the break list and the panel; 24px page gutters (16px on phones).

### The workspace
Slotify is **one screen**. There are no steps, no step links and no separate
landing page: what the user has decided is always visible.

- **Empty state** — no file yet: the `empty-state` centred in the viewport (a
  `display-md` line, one `body-lg` line in ink-subtle, then the `dropzone`), with
  the dimmed soundwave video behind it. Dropping or choosing a file starts
  analysis immediately; there is no Analyze button.
- **Workspace** — a file is loaded:
  - `app-header` with the wordmark left and the `file-bar` (`name · format · size · duration` in mono, then a `button-secondary` Replace) right.
  - Main column: the `waveform-panel` full width, then the `break-list`.
  - Right column (≥1024px): the `ad-panel`, 360px, sticky below the header.
  - Below 1024px the ad panel stacks under the break list.
- Max content width 1280px, centred.

### Whitespace
The canvas is the whitespace. Separate sections by lifting them onto panels,
not by adding colour bands.

## Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| 0 | Canvas, no border | Page, empty-state text |
| 1 | `{colors.surface-1}` + 1px `{colors.hairline}` | Break list, ad panel, inputs, dropzone |
| 2 | `{colors.surface-2}` | Hovered or selected break row, badges |
| 3 | `{colors.surface-3}` | Dropdowns, nested panels |
| Focus | 2px `{colors.focus-ring}` outline, 2px offset | Every focusable control |
| Overlay | `{colors.scrim}` behind a level-1 modal | Rights modal, voice-cloning overlay |

No drop shadows on dark surfaces, no coloured glows, no blur-heavy glass.

## Shapes

| Token | Value | Use |
|---|---|---|
| `{rounded.xs}` | 4px | Provenance chips, tiny tags |
| `{rounded.sm}` | 6px | Checkbox, inline tags |
| `{rounded.md}` | 8px | Buttons, inputs, selects |
| `{rounded.lg}` | 12px | Break list, ad panel, waveform panel |
| `{rounded.xl}` | 16px | Dropzone, modal |
| `{rounded.pill}` | 9999px | Score and status badges only |

## Components

### Header
**`app-header`** — 56px bar on canvas with a 1px bottom hairline: the wordmark
(amber mark + ink "Slotify") left; the `file-bar` and a GitHub icon link right.
No step links.

### Buttons
- **`button-primary`** — amber fill, near-black label, 8px radius. Exactly one on screen: **Render** in the ad panel ("Render 1 ad break", counting the selection). In the empty state there is no amber button — the dropzone is the action.
- **`button-primary-wide`** — Render, full width of the ad panel, 44px tall.
- **`button-secondary`** — surface-1 with a hairline border: Replace, Add sponsor, Add break, Download, Cancel.
- **`button-tertiary`** — text-only: remove a sponsor, expand a row's details.
- **`button-disabled`** — surface-2, ink-tertiary label, hairline border; never a dimmed amber. A disabled Render says why underneath in caption ink-subtle (e.g. "Needs an ElevenLabs key on the server").

### Empty state and file
- **`dropzone`** — surface-1, 1px dashed hairline-strong, 16px radius, keyboard-focusable; on drag-over **`dropzone-active`**. One caption line lists accepted formats and the size limit.
- **`file-bar`** — mono `ryan-test1.mp3 · MP3 · 358 KB · 0:22` in ink-muted, then Replace. It replaces the old dropzone + file-details card.

### Waveform
**`waveform-panel`** — canvas well inside a 12px hairline frame, full width of
the main column. Bars in `{colors.wave}`; unselected markers
`{colors.wave-marker}`; selected markers `{colors.accent}`; the insertion cursor
`{colors.ink}`. Clicking a marker selects or focuses its row. Start/end time in
mono caption.

### Break list
- **`break-list`** — a level-1 panel with a header row: "Break points" (card-title) left; the `provenance-chip` and, when it applies, the weak-label notice (caption ink-subtle) right. Never hide the notice.
- **`break-row`** — one row per break, in rank order, columns: checkbox · timestamp (mono, ink) · score (mono number + `score-bar`) · 2–3 `reason-tag`s joined by " · " ("pause 1.2s · mid-episode") · Preview (▶ icon button, `button-tertiary`, with an accessible label). Hover is `break-row-hover`; selected is `break-row-selected` (surface-2 + 2px amber left edge). Clicking a row toggles it; a chevron expands the full reasoning (every measured signal, including cautions) in body-sm ink-subtle.
- When there is more than one sponsor, a selected row shows a compact sponsor `select` in place of the reason tags' tail; with one sponsor there is no per-row control.
- The last row is "Add a break at mm:ss" — two small mono inputs and a `button-secondary` Add.
- No per-row "ranked by" labels and no restated prose: the header says what ranked them once.
- **Analysis in progress** — the list body shows `inline-progress`: a small bar animation (wave bars, active bar amber), one status line, and a `button-secondary` Cancel. The ad panel stays usable meanwhile.

### Ad panel
**`ad-panel`** — everything about the ad, in order: Sponsors (name +
optional script per sponsor, up to three, "Add sponsor"), Tone and Language
selects side by side (helper: "Applies to scripts Slotify writes; a typed
script is read as written"), then Render. After a render the panel shows the
result: an audio player, Download (`button-secondary`) and "Render again".
Preview results play inline in the row that produced them.

### Inputs
**`text-input`** for text fields, textareas and selects; focus adds the
hairline-strong border and the 2px amber focus ring. Labels are body-sm ink-muted
above the field; helper text is caption ink-subtle below.

### Overlays
- **`processing-overlay`** — fixed, full-screen, canvas at 94%; only for **voice cloning**, which is short and must block. Bar animation, headline, one status line, `button-secondary` Cancel. Analysis never uses it.
- **`modal`** — surface-1 with hairline-strong, 16px radius, over the scrim; primary and tertiary buttons right-aligned. Used for the voice-rights confirmation.

### Status text
`status-ok` in success, `status-error` in danger, both body-sm, always with
words — colour is never the only signal.

## Do's and Don'ts

### Do
- Keep `{colors.canvas}` as the anchor surface and lift content onto panels.
- Use amber only for the brand mark, the primary action, focus and selection.
- Put near-black text on amber.
- Set timestamps, scores and sizes in JetBrains Mono.
- Give every focusable control the amber focus ring.
- Say what was measured on each break in words, not only with colour.
- Keep the workspace to one screen; if something new needs a step, it belongs in the ad panel or a row's details.

### Don't
- Don't add a second accent (no copper CTA, no teal waveform, no green "Selected" button).
- Don't bring back steps, a separate landing page, or per-row repeated labels.
- Don't use gradients, glows, or gradient text in the app.
- Don't pill-round buttons; pills are for badges.
- Don't put white text on amber.
- Don't use `{colors.ink-tertiary}` for anything a user needs to read.
- Don't use pure #000000 as the canvas.

## Responsive Behavior

| Width | Changes |
|---|---|
| ≥1024px | Two columns: main (waveform + break list) and the 360px sticky ad panel |
| 640–1023px | One column: waveform, break list, then the ad panel |
| <640px | Break rows wrap reason tags to a second line; the file bar truncates the file name with an ellipsis; 16px gutters; the empty-state heading scales 40px → 28px |

- Buttons stay ≥36px tall (≥44px for the wide primary); inputs ≥40px.
- No horizontal scroll at 360px.

## Agent Prompt Guide

- Start from the tokens above; reference components by their `components:` name.
- Decide which surface level a new element sits on before choosing any colour.
- If a screen seems to need a second colour, it needs hierarchy instead: use the surface ladder and type weight.
- New numbers go in mono; new prose in Inter.
- Keep amber scarce — if more than one amber filled control is visible at once, one of them is wrong.
