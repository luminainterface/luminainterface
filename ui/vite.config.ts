import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

const apiUrl = process.env.VITE_API_URL || 'http://localhost:8201'
const eventsUrl = process.env.VITE_EVENTS_URL || 'http://localhost:8101'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    proxy: {
      '/api': {
        target: apiUrl,
        changeOrigin: true,
        secure: false
      },
      '/ws': {
        target: eventsUrl.replace(/^http/, 'ws'),
        ws: true,
        changeOrigin: true,
        secure: false
      },
      '/health': {
        target: apiUrl,
        changeOrigin: true,
        secure: false
      }
    }
  }
}) 