# Slotify frontend

React 19 + TypeScript + Vite. One screen: drop an episode, read the ranked
ad-insertion points the backend returns, pick some, and (with ElevenLabs
configured on the server) render sponsor reads in the host's cloned voice.
`../DESIGN.md` is the design system it follows.

## Development

The frontend talks to the Node API in `../backend` (default
`http://localhost:3001`). Start that first, then:

```bash
npm install
npm run dev        # http://localhost:5173 (or the next free port)
```

The app calls same-origin `/api/...` URLs. The Vite dev server proxies `/api`
to the backend, so it keeps working if Vite picks a different port and the
backend's CORS list (`CORS_ORIGIN`) does not need to include it.

## Checks

```bash
npm run lint
npm test           # node:test files in src/**/*.test.ts, run with tsx
npm run build      # tsc -b, then vite build into dist/
npm run verify     # all three
```

## Production build

```bash
npm run build
npm run preview    # serves dist/ and proxies /api like the dev server
```

`dist/` is static. Serve it from the same origin as the API (for example,
behind a reverse proxy that routes `/api` to the backend), or build with
`VITE_API_BASE_URL` pointing at the API and add the site's origin to the
backend's `CORS_ORIGIN`.

## Environment variables

Set these in the shell or in `frontend/.env.local`. `VITE_*` values are
compiled into the bundle, so changing them needs a dev-server restart or a
rebuild.

| Variable | Used by | Default | Purpose |
| --- | --- | --- | --- |
| `VITE_API_BASE_URL` | app | empty (same origin) | Absolute API origin, e.g. `https://api.example.com`. Leave unset to use `/api` on the page's own origin. |
| `VITE_API_TOKEN` | app | unset | Sent as `Authorization: Bearer <token>` on every API request. Set it to the backend's `SLOTIFY_API_TOKEN` when that is set. It ships in the bundle, so anyone who can load the page can read it. |
| `VITE_PROXY_TARGET` | `vite` / `vite preview` | `http://localhost:3001` | Where the dev and preview servers proxy `/api`. |

## Layout

- `src/App.tsx`: the workspace's state and its API calls (analysis, voice clone, TTS, merge).
- `src/components/`: presentational pieces -- `EmptyState` (no file yet), `AppHeader` (wordmark and the file bar with Replace), `WaveformPanel`, `BreakList` / `BreakRow`, `AdPanel`, `RightsModal`, `ProcessingOverlay` (voice cloning only).
- `src/lib/format.ts`: timestamps, sizes and the file bar text.
- `src/lib/workspace.ts`: reason tags for a break row, "Add a break at mm:ss" parsing, the Render label and why it is off.
- `src/lib/api.ts`: the one fetch helper (base URL, bearer token, error messages).
- `src/lib/capabilities.ts`: what the backend can do, and why generation is off when it is.
- `src/lib/recommendations.ts`: parses `/api/insert-sections` without inventing slots or scores.
- `src/lib/tasks.ts`: per-file cancellation, so a previous upload's requests never write into the current one.
