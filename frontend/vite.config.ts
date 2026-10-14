import { defineConfig, loadEnv, type ProxyOptions } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
//
// The app calls same-origin /api/* URLs unless VITE_API_BASE_URL is set. Both
// `vite` and `vite preview` proxy /api to the backend, so the backend's CORS
// allow-list never needs to know which port Vite ended up on.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_PROXY_TARGET || 'http://localhost:3001'

  const apiProxy: Record<string, ProxyOptions> = {
    '/api': {
      target,
      changeOrigin: true,
      configure(proxy) {
        // Browsers send Origin on same-origin POSTs, and through the proxy that
        // is Vite's origin, which the backend may reject. A proxied request is
        // not cross-origin, so the header is dropped.
        proxy.on('proxyReq', (proxyReq) => {
          proxyReq.removeHeader('origin')
        })
        // Vite answers a dead backend with an empty 500; say what happened in
        // the `{error}` shape the UI shows.
        proxy.on('error', (err, _req, res) => {
          if (!('writeHead' in res) || res.headersSent) return
          const reason =
            err.message || (err as NodeJS.ErrnoException).code || 'no response'
          res.writeHead(502, { 'Content-Type': 'application/json' })
          res.end(
            JSON.stringify({
              error: `The Slotify API is not reachable at ${target} (${reason}). Start the backend, or set VITE_PROXY_TARGET.`,
            }),
          )
        })
      },
    },
  }

  return {
    plugins: [react()],
    server: { proxy: apiProxy },
    preview: { proxy: apiProxy },
  }
})
