import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const BACKEND_ORIGIN = process.env.BACKEND_ORIGIN ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
server: {
      port: 5173,
      // Fail loudly instead of silently moving to 5174: the backend CORS allowlist
      // and any bookmarks both name 5173 explicitly.
      strictPort: true,
      watch: {
        // Uploaded files land in uploads/, and the backend and its virtualenv sit
        // beside the app. Watching those made every save and every upload churn
        // the dev server, and an upload could reload the page out from under the
        // user mid-chat. None of it affects the frontend bundle.
        ignored: [
          '**/uploads/**',
          '**/backend/**',
          '**/venv/**',
          '**/chroma_rag/**',
          '**/chroma_memory/**',
          '**/frontend-legacy/**',
          '**/.git/**',
        ],
      },
      proxy: {
      // Same-origin in dev, so no CORS preflight and no cookie/SameSite issues.
      '/api': {
        target: BACKEND_ORIGIN,
        changeOrigin: true,
        // Vite forwards the original path, so the /api prefix has to be
        // stripped or the backend 404s on a route it does not define.
        rewrite: (path) => path.replace(/^\/api/, ''),
        // SSE must not be buffered by the proxy or tokens arrive in one burst.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              proxyRes.headers['x-accel-buffering'] = 'no'
            }
          })
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})