import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const backend = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5188,
    proxy: {
      '/api': { target: backend, rewrite: (p) => p.replace(/^\/api/, '') },
      '/media': backend,
      '/samples': backend,
    },
  },
})
