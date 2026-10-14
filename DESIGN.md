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
  top-nav:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink-subtle}"
    typography: "{typography.body-sm}"
    height: 56px
    borderBottom: "1px {colors.hairline}"
  nav-link-active:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: 6px 12px
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
  slot-card:
    backgroundColor: "{colors.surface-1}"
    border: "1px {colors.hairline}"
    rounded: "{rounded.lg}"
    padding: 20px
  slot-card-selected:
    backgroundColor: "{colors.surface-2}"
    border: "1px {colors.accent-line}"
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
  step-indicator:
    textColor: "{colors.ink-subtle}"
    typography: "{typography.caption}"
  step-indicator-current:
    textColor: "{colors.ink}"
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

Slotify is a working tool: upload an episode, read a ranked list of break
points, pick some, render. The design gets out of the way of that loop. The
page sits on `{colors.canvas}` (#010102, near-black with a faint cool tint),
and everything the user reads or touches lives on a lifted panel from a
four-step surface ladder (`{colors.surface-1}` → `{colors.surface-4}`) edged
with 1px hairlines (`{colors.hairline}` → `{colors.hairline-tertiary}`).
Hierarchy comes from lift and type weight, never from glow.

**Slotify amber** (`{colors.accent}` #c8a84a) is the only chromatic accent. It
marks exactly four things: the brand mark, the one primary action on a screen,
keyboard focus, and the selected/active state (a selected slot, the current
step, the waveform marker of a chosen break). Because amber is light, text on
it is near-black (`{colors.on-accent}`), never white.

Inter carries every word, at weight 600 for display with measured negative
tracking and 400 for body. **JetBrains Mono** carries the numbers an editor
checks — timestamps, placement scores, durations, file names and provenance
labels — so they align and read as data.

The landing page keeps one cinematic element: the soundwave video, rendered
grayscale (its blue would be a second accent) and dimmed so the lead text keeps
4.5:1 over its brightest frames. Everywhere else the product's own surfaces — the waveform and
the slot list — are the visual.

**Key Characteristics:**
- Near-black canvas with a four-step surface ladder and 1px hairlines; no drop shadows on dark.
- One accent, amber, for brand mark, primary action, focus and selection only.
- Inter 600 display with negative tracking; Inter 400 body; JetBrains Mono for data.
- 8px buttons and inputs, 12px cards, 16px panels; pills only for badges.
- No gradients, glows, or gradient text in the app chrome.

## Colors

### Accent
- **Amber** (`{colors.accent}`): brand mark, primary CTA, current step, selected slot border, chosen-break marker on the waveform, link emphasis.
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
- **Success** (`{colors.success}`): the "measured" check marks on a slot and success status text.
- **Danger** (`{colors.danger}`): error text and destructive confirmations.
- No warning colour: a warning is ink-muted text with an icon, so amber keeps one meaning.

### Waveform
- **Wave** (`{colors.wave}`): the waveform bars.
- **Wave Strong** (`{colors.wave-strong}`): the portion under the playhead, when shown.
- **Wave Marker** (`{colors.wave-marker}`): an unselected slot's marker line.
- A **selected** slot's marker is `{colors.accent}`; the insertion cursor is `{colors.ink}`.

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
| `{typography.body-sm}` | 14px | 400 | 1.5 | 0 | Card body, slot reasons, nav |
| `{typography.caption}` | 12px | 400 | 1.4 | 0 | Helper text, meta |
| `{typography.button}` | 14px | 500 | 1.2 | 0 | All button labels |
| `{typography.eyebrow}` | 12px | 500 | 1.3 | +0.4px | Section eyebrows, uppercase |
| `{typography.mono}` | 13px | 400 | 1.5 | 0 | Scores, sizes, chips |
| `{typography.mono-display}` | 32px | 500 | 1.1 | -1.0px | A slot's timestamp |

### Principles
- Display at 600, body at 400; nothing above 600.
- Negative tracking on display, positive on uppercase eyebrows.
- Numbers a user compares are mono; prose is never mono.
- No gradient text, including the wordmark.

## Layout

### Spacing
- 4px base unit: 4, 8, 12, 16, 24, 32, 48, 96.
- Panels pad 32px (24px under 768px); cards 24px; slot cards 20px.
- 24px between blocks inside a panel; 48px between panels on a screen.

### Grid & Container
- App content max width 1120px, centred, 24px side gutters (16px on phones).
- Slot cards: 3-up from 1024px, 2-up from 640px, 1-up below.
- The waveform panel always spans the full content width.

### Whitespace
The canvas is the whitespace. Separate sections by lifting them onto panels,
not by adding colour bands.

## Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| 0 | Canvas, no border | Page, landing hero text |
| 1 | `{colors.surface-1}` + 1px `{colors.hairline}` | Panels, cards, inputs |
| 2 | `{colors.surface-2}` + 1px `{colors.hairline-strong}` | Hovered or selected card, active nav link |
| 3 | `{colors.surface-3}` | Dropdowns, nested panels |
| Focus | 2px `{colors.focus-ring}` outline, 2px offset | Every focusable control |
| Overlay | `{colors.scrim}` behind a level-1 modal | Rights modal, processing overlay |

No drop shadows on dark surfaces, no coloured glows, no blur-heavy glass.

## Shapes

| Token | Value | Use |
|---|---|---|
| `{rounded.xs}` | 4px | Provenance chips, tiny tags |
| `{rounded.sm}` | 6px | Checkbox, inline tags |
| `{rounded.md}` | 8px | Buttons, inputs, selects, nav links |
| `{rounded.lg}` | 12px | Cards, slot cards, waveform panel |
| `{rounded.xl}` | 16px | Screen panels, dropzone, modal |
| `{rounded.pill}` | 9999px | Score and status badges only |

## Components

### Navigation
**`top-nav`** — 56px bar on canvas with a 1px bottom hairline. Wordmark left
(amber mark + ink "Slotify"), step links right as `button-tertiary`; the
current step is **`nav-link-active`** (surface-2, ink).

**Step indicator** — three small labelled dots; completed steps ink-subtle with
a check, the current step ink with an amber dot, future steps an ink-tertiary
dot with an ink-subtle label (labels must stay readable at 4.5:1).

### Buttons
- **`button-primary`** — amber fill, near-black label, 8px radius. One per screen: Try Slotify, Analyze, Confirm selections, Render & export.
- **`button-primary-wide`** — the same, full width, 44px tall, for the screen's main action.
- **`button-secondary`** — surface-1 with a hairline border: Preview, Add another sponsor, Add custom slot, Back.
- **`button-tertiary`** — text-only: Cancel, nav links.
- **`button-disabled`** — surface-2, ink-tertiary label, hairline border; never a dimmed amber.

### Upload
**`dropzone`** — surface-1, 1px dashed hairline-strong, 16px radius; on drag
or once a file is chosen, **`dropzone-active`**. File details sit in a
`card` as a two-column definition grid, values in mono.

### Analysis
- **`waveform-panel`** — canvas well inside a 12px hairline frame. Bars in `{colors.wave}`; unselected markers `{colors.wave-marker}`; selected markers `{colors.accent}`; insertion cursor `{colors.ink}`. Time labels in mono caption.
- **`slot-card`** — surface-1, 12px. Header: rank as an eyebrow, timestamp in `{typography.mono-display}`, `score-badge` right. Reasons as a list with a small `{colors.success}` check. Actions: Preview (`button-secondary`), Select (`button-secondary`) / Selected (`slot-card-selected` + amber check label).
- **`provenance-chip`** — mono label naming the ranker ("Learned ranker (gated)"); the weak-label notice beside it is ink-subtle body-sm, never hidden.

### Inputs
**`text-input`** for text fields, textareas and selects; focus adds the
hairline-strong border and the 2px amber focus ring. Labels are body-sm ink-muted
above the field; helper text is caption ink-subtle below.

### Overlays
- **`processing-overlay`** — fixed, full-screen, canvas at 94%; a small bar animation in `{colors.wave}` with the active bar amber; headline, one line of status, and a `button-secondary` Cancel.
- **`modal`** — surface-1 with hairline-strong, 16px radius, over the scrim; primary and tertiary buttons right-aligned.

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
- Say what was measured on each slot in words, not only with colour.

### Don't
- Don't add a second accent (no copper CTA, no teal waveform, no green "Selected" button).
- Don't use gradients, glows, or gradient text in the app.
- Don't pill-round buttons; pills are for badges.
- Don't put white text on amber.
- Don't use `{colors.ink-tertiary}` for anything a user needs to read.
- Don't use pure #000000 as the canvas.

## Responsive Behavior

| Width | Changes |
|---|---|
| ≥1024px | Slot cards 3-up; panels pad 32px |
| 640–1023px | Slot cards 2-up |
| <768px | Nav step links stay as compact buttons (the export screen has no other way back); panels pad 24px |
| <640px | Slot cards 1-up; landing wordmark scales 80px → 44px; 16px side gutters |

- Buttons stay ≥36px tall (≥44px for the wide primary); inputs ≥40px.
- No horizontal scroll at 360px.

## Agent Prompt Guide

- Start from the tokens above; reference components by their `components:` name.
- Decide which surface level a new element sits on before choosing any colour.
- If a screen seems to need a second colour, it needs hierarchy instead: use the surface ladder and type weight.
- New numbers go in mono; new prose in Inter.
- Keep amber scarce — if more than one amber filled control is visible on a screen, one of them is wrong.
