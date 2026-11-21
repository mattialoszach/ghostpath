import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const target = process.env.GHOSTPATH_BACKEND || 'http://127.0.0.1:8765'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': target, '/ws': { target, ws: true } } },
  preview: { proxy: { '/api': target, '/ws': { target, ws: true } } },
})
